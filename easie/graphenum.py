import itertools
from typing import List
import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem
import re
from .utils.draw import draw_bipartite_route
from .utils.graph import (
    graph_from_reaction_smiles,
    graph_from_reaction_smarts,
)
from .utils.utils import (
    bulk_similarity,
    bulk_plausibility,
    generate_product,
    run_reaction_node,
    get_atom_ids_for_substruct_match,
    generalize_query,
    flatten_list,
    merge_generalized_queries,
    get_substruct_match_idxs,
    get_strict_symbols_for_mol,
    instantiate_mol
)

class RxnGraphEnumerator:
    def __init__(
        self,
        reaction_smiles: List[str],
        forward_filter=None,
        mapper=None,
        reaction_smarts=None,
    ):
        self.reaction_smiles = reaction_smiles
        self.reaction_smarts = reaction_smarts
        if not self.reaction_smarts:
            print ('Constructing graph with reaction SMILES...')
            self.graph = graph_from_reaction_smiles(self.reaction_smiles, mapper)
            print ('Done constructing graph with reaction SMILES')
        else:
            print ('Constructing graph with reaction SMARTS...')
            self.graph = graph_from_reaction_smarts(
                self.reaction_smiles, self.reaction_smarts
            )
            print ('Done constructing graph with reaction SMARTS')
        
        self.root = self.get_root()
        self.root_smiles = self.graph.nodes[self.root]["smiles"]
        self.leaves = []
        for leaf_node in self.get_leaves():
            smiles = self.graph.nodes[leaf_node]["smiles"]
            self.leaves.append(
                {
                    "id": leaf_node,
                    "smiles": smiles,
                    "options": [smiles],
                    "query": [],
                    "queried_atoms": [],
                }
            )
        self.reaction_nodes = sorted(
            [
                node
                for node in self.graph.nodes
                if self.graph.nodes[node]["type"] == "reaction"
            ],
            key=lambda x: nx.shortest_path_length(
                self.graph, source=self.root, target=x
            ),
            reverse=True,
        )
        self.bb_labeled_graph = self.get_labeled_intermediates_by_building_block()
        self.get_building_block_queries()
        self.forward_filter = forward_filter

    @staticmethod
    def load_fast_filter(fast_filter_scorer):
        """Load fast filter model and return scoring function.

        This is a tensorflow model, so be careful if using multiprocessing.
        """
        model = fast_filter_scorer()
        model.load()
        return model.predict

    def get_labeled_intermediates_by_building_block(self):
        bb_map = {}
        reaction_nodes = self.reaction_nodes
        # label each leaf node with unique isotope numbers
        atom_uid = 1
        self.atom_uid_to_leaf = {}
        for leaf in self.leaves:
            leaf_idx = leaf["id"]
            smiles = self.graph.nodes[leaf_idx]["smiles"]
            mol = Chem.MolFromSmiles(smiles)
            for a in mol.GetAtoms():
                a.SetIsotope(atom_uid)
                self.atom_uid_to_leaf[atom_uid] = leaf_idx
                atom_uid += 1
            bb_map[leaf_idx] = Chem.MolToSmiles(mol)

        # run through the reaction graph to get isotope labeled products
        g = self.graph.copy()
        nx.set_node_attributes(g, bb_map, name="smiles")
        for reaction_node in reaction_nodes:
            product = run_reaction_node(g, reaction_node, constrain_smiles=True)
            parent = list(g.pred[reaction_node])[0]
            nx.set_node_attributes(g, {parent: product}, name="smiles")

        return g

    def get_building_block_queries(self):
        g = self.bb_labeled_graph
        for r in self.reaction_nodes:
            rxn_smarts = g.nodes[r]["template"]
            orig_reactant_smarts = rxn_smarts.split(">>")[-1].split(".")

            # get the successors for the reaction node
            reactants = [Chem.MolFromSmiles(g.nodes[s]["smiles"]) for s in g.succ[r]]
            reactant_smarts = [
                Chem.MolFromSmarts(s) for s in rxn_smarts.split(">>")[-1].split(".")
            ]

            reactant_substruct_idxs, reactant_substruct_idxs_w_lg = get_substruct_match_idxs (reactant_smarts, reactants)
            
            # Get the combinations of atoms idxs matched by each reactant
            r_match_combos = list(itertools.product(*reactant_substruct_idxs))
            r_match_combos_w_lg = list(itertools.product(*reactant_substruct_idxs_w_lg))
            
            product = Chem.MolFromSmiles(g.nodes[list(g.pred[r])[0]]["smiles"])
            product_smarts = Chem.MolFromSmarts(rxn_smarts.split(">>")[0])
            map_nums = [a.GetAtomMapNum() for a in product_smarts.GetAtoms()]
            product_substruct_matches = product.GetSubstructMatches(product_smarts, uniquify=False)
            product_matched_ids = [
                get_atom_ids_for_substruct_match(product, match, map_nums)
                for match in product_substruct_matches
            ]

            if len(r_match_combos):
                reactant_matched_ids = []
                for combo in r_match_combos:
                    combo_matched_ids = set.union(*combo)
                    reactant_matched_ids.append(combo_matched_ids)
                r_idx = None
                for i, r_ids in enumerate(reactant_matched_ids):
                    for _, p_ids in enumerate(product_matched_ids):
                        if r_ids == p_ids:
                            r_idx = i
                            break
                try:
                    r_combo = r_match_combos_w_lg[r_idx]
                except:
                    print ("No matches found")
                    # pdb.set_trace()
                    raise

                for atom_idxs, smarts in zip(r_combo, orig_reactant_smarts):
                    rdk_smarts = Chem.MolFromSmarts(smarts)
                    for leaf in self.leaves:
                        leaf_smiles = g.nodes[leaf["id"]]["smiles"]
                        leaf_mol = Chem.MolFromSmiles(leaf_smiles)
                        # leaf_atom_idxs = set(
                        #     [a.GetIsotope() for a in leaf_mol.GetAtoms()]
                        # )
                        leaf_match = leaf_mol.GetSubstructMatches(rdk_smarts)
                        leaf_matched_ids = [get_atom_ids_for_substruct_match(leaf_mol, match) for match in leaf_match]
                        for leaf_atom_idxs in leaf_matched_ids:
                            if leaf_atom_idxs.intersection(atom_idxs) == atom_idxs:
                                leaf["query"].append(smarts)
                                leaf["queried_atoms"].append(atom_idxs)

        for leaf in self.leaves:
            query_list = leaf["query"]
            queried_atoms = leaf["queried_atoms"]
            leaf_mol = Chem.MolFromSmiles(leaf["smiles"])
            
            #add in leaving group patterns
            iso_labeled_mol = Chem.MolFromSmiles(g.nodes[leaf['id']]['smiles'])
            unaccounted_lg_atom_idxs = []
            unaccounted_lg_atom_isos = []
            accounted_for = set.union(*leaf['queried_atoms'])
            root_atoms = [a.GetIsotope() for a in Chem.MolFromSmiles(g.nodes[self.root]['smiles']).GetAtoms()]
            lg_atoms = [a.GetIsotope() for a in iso_labeled_mol.GetAtoms() if a.GetIsotope() not in root_atoms]
            symbols = get_strict_symbols_for_mol(iso_labeled_mol)
            iso_patt = re.compile("(?<=\[)[0-9]+(?=[A-Z,a-z])")
            neighbors_iso = set()
            for atom in iso_labeled_mol.GetAtoms():
                iso = atom.GetIsotope()
                if iso in lg_atoms and iso not in accounted_for:
                    unaccounted_lg_atom_idxs.append(atom.GetIdx())
                    unaccounted_lg_atom_isos.append(iso)
                    for n in atom.GetNeighbors():
                        neighbors_iso.add(n.GetIsotope()) 
            
            #merge neighbors of the pattern into the lg pattern
            if len(unaccounted_lg_atom_idxs):
                for q, prev_match in zip(query_list, queried_atoms):
                    if len(prev_match.intersection(neighbors_iso)) > 0:
                        query_list.remove(q)
                        queried_atoms.remove(prev_match)
                        unaccounted_lg_atom_idxs.extend([a.GetIdx() for a in iso_labeled_mol.GetAtoms() if a.GetIsotope() in prev_match])
                        unaccounted_lg_atom_isos.extend(prev_match)

                lg_query = Chem.MolFragmentToSmiles(iso_labeled_mol, unaccounted_lg_atom_idxs, 
                                                    isomericSmiles=True, atomSymbols=symbols, 
                                                    allBondsExplicit=True, allHsExplicit=True)
                lg_query = re.sub(iso_patt, "", lg_query)
                query_list.append(lg_query)
                queried_atoms.append(set(unaccounted_lg_atom_isos))
                # pdb.set_trace()
            
            
            if len(set.union(*queried_atoms)) == sum([len(s) for s in queried_atoms]):
                # no overlap between query patterns
                query_string = ".".join(query_list)
                query_obj = Chem.MolFromSmarts(query_string)
            
            else:
                # convert overlapping portions of query SMARTS to recursive SMARTS
                duplicated_idxs = []
                for idx in set.union(*queried_atoms):
                    if sum([idx in s for s in queried_atoms]) > 1:
                        duplicated_idxs.append(idx)
                need_to_generalize = []
                for a in queried_atoms:
                    if len(set(a).intersection(duplicated_idxs)) > 0:
                        need_to_generalize.append(True)
                    else:
                        need_to_generalize.append(False)
                for i, n in enumerate(need_to_generalize):
                    if n:
                        query_list[i] = generalize_query(query_list[i])
                        query_string = ".".join(query_list)
                        query_obj = Chem.MolFromSmarts(query_string)
                        queried_atoms[i] = get_atom_ids_for_substruct_match(leaf_mol, leaf_mol.GetSubstructMatch(query_obj))#set([])
                        if leaf_mol.HasSubstructMatch(query_obj) and len(
                            set.union(*queried_atoms)
                        ) == sum([len(s) for s in queried_atoms]):
                            break
            
            if not leaf_mol.HasSubstructMatch(query_obj):
                matches = [flatten_list(leaf_mol.GetSubstructMatches(Chem.MolFromSmarts(q))) for q in query_list]
                for i, m1 in enumerate(matches):
                    if len(m1) == 1 and len(query_list[i]):
                        queries_to_combine = []
                        for j, m2 in enumerate(matches):
                            if j > i and m1==m2:
                                queries_to_combine.append(query_list[j])
                                query_list[j] = ""
                        query_list[i] = merge_generalized_queries([query_list[i]]+queries_to_combine)
                query_string = ".".join([q for q in query_list if len(q)])

            try:
                query_obj = Chem.MolFromSmarts(query_string)
                assert leaf_mol.HasSubstructMatch(query_obj)
            except:
                print ("WARNING: Query string does not match building block")
                print ("building block:", leaf['smiles'])
                print ("Query:", query_string)
                # pdb.set_trace()
            leaf["query"] = query_string
            del leaf["queried_atoms"]


    def search_building_blocks(self, pricer, price_cutoff=1000):
        for leaf in self.leaves:
            query = leaf.get("query")
            options = pricer.lookup_smarts(query)
            options = [bb["smiles"] for bb in options if bb["ppg"]<=price_cutoff]
            leaf["options"] = np.unique([leaf.get("smiles")] + options).tolist()

    def limit_query_matches(self):
        for leaf in self.leaves:
            for query in leaf["query"].split("."):
                q = Chem.MolFromSmarts(query)
                leaf["options"] = list(
                    filter(
                        lambda x: len(Chem.MolFromSmiles(x).GetSubstructMatches(q))
                        == 1,
                        leaf["options"],
                    )
                )

    def filter_by_similarity(
        self,
        threshold: float = 0.5,
        cutoff: float = None,
        fraction: float = None,
        num: int = None,
        keep_most_similar: bool = True,
    ) -> List[str]:
        """
        Filter possible building block options by similarity to the original
        building block.

        Parameters:
            threshold: minimum similarity
            cutoff: maximum similarity
            fraction: keeps a set fraction of the building block options
            keep_most_similar: whether to keep the most similar or the most dissimilar fraction of the options

        """
        for leaf in self.leaves:
            smiles = leaf["smiles"]
            options = leaf["options"]
            sim_df = pd.DataFrame()
            sim_df["smiles"] = options
            sim_df["similarity"] = bulk_similarity(smiles, options)
            sim_df = sim_df.sort_values("similarity", ascending=(not keep_most_similar))
            if fraction:
                num_to_keep = int(fraction * len(sim_df))
            elif num:
                num_to_keep = num
            else:
                num_to_keep = len(sim_df)
            sim_df = sim_df.iloc[:num_to_keep].sort_values(
                "similarity", ascending=False
            )
            if threshold:
                sim_df = sim_df[sim_df["similarity"] >= threshold]
            if cutoff:
                sim_df = sim_df[sim_df["similarity"] <= cutoff]
            leaf["options"] = sim_df["smiles"].values.tolist()

    def filter_by_plausibility(self, threshold: float = 0.5, fraction: float = None):
        """
        Filter possible building block options by plausibility that they will
        perform the desired reaction.

        Parameters:
            threshold: minimum similarity
            fraction: keeps a set fraction of the building block options
        """
        if self.forward_filter is None:
            raise ValueError("Plausibility Filter not loaded correctly")

        for leaf in self.leaves:
            smiles = leaf["smiles"]
            options = leaf["options"]
            reaction_node = list(self.graph.predecessors(leaf["id"]))[0]
            reaction_smarts = self.graph.nodes[reaction_node]["template_reverse"]
            fixed_smiles = ".".join(
                [
                    self.graph.nodes[reactant_node]["smiles"]
                    for reactant_node in self.graph.successors(reaction_node)
                    if self.graph.nodes[reactant_node]["smiles"] != smiles
                ]
            )  # other reactants
            sim_df = pd.DataFrame()
            sim_df["smiles"] = options
            sim_df["plausibility"] = bulk_plausibility(
                fixed_smiles, reaction_smarts, options, self.forward_filter
            )
            sim_df = sim_df.sort_values("plausibility", ascending=False)

            sim_df = sim_df[sim_df["plausibility"] >= threshold]
            leaf["options"] = sim_df["smiles"].values.tolist()

    def bb_combinations(self):
        return itertools.product(*[leaf["options"] for leaf in self.leaves])

    def get_root(self):
        for node, deg in self.graph.in_degree:
            if deg == 0:
                return node
        return

    def get_leaves(self):
        return [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]

    def draw(self):
        draw_bipartite_route(self.graph)

    def get_pg_leaves(self):
        """
        Identifies leaf nodes in the graph that represent protecting group molecules
        e.g. molecules from which no atoms end up in the final product.
        """
        prod = self.bb_labeled_graph.nodes[self.root]["smiles"]
        prod_mol = Chem.MolFromSmiles(prod)
        isotope_nums = set([])
        for a in prod_mol.GetAtoms():
            if a.GetIsotope() != 0:
                isotope_nums.add(self.atom_uid_to_leaf[a.GetIsotope()])

        for leaf in self.leaves:
            if leaf["id"] in isotope_nums:
                leaf["pg"] = False
            else:
                leaf["pg"] = True

    def count_combinations(self, deduplicate=True):
        """
        Returns the number of combinations of building blocks
        Parameters:
            deduplicate (bool): if True, do not count nodes identified as protecting groups.
        """
        if deduplicate:
            self.get_pg_leaves()
            return np.prod(
                [len(leaf["options"]) for leaf in self.leaves if not leaf["pg"]]
            )
        else:
            return np.prod([len(leaf["options"]) for leaf in self.leaves])

    def library_generator(self):
        for bbs in self.bb_combinations():
            bb_map = {leaf.get("id"): bb for leaf, bb in zip(self.leaves, bbs)}
            yield generate_product(self.graph, self.reaction_nodes, bb_map)

    def bb_map(self):
        """
        Returns a mapping from original building block smiles to analog building
        block smiles for every combination of building block analogs
        """
        for bbs in self.bb_combinations():
            bb_map = {leaf.get("id"): bb for leaf, bb in zip(self.leaves, bbs)}
            yield bb_map

    def bb_map_filtered(self, filter_func):
        for bbs in self.bb_combinations():
            if filter_func(bbs):
                bb_map = {leaf.get("id"): bb for leaf, bb in zip(self.leaves, bbs)}
                yield bb_map

    def generate_library(self, nproc=1):
        return Parallel(n_jobs=nproc, verbose=1)(
            delayed(generate_product)(self.graph, self.reaction_nodes, bb_map)
            for bb_map in list(self.bb_map())
        )

    def generate_library_filtered(self, filter_func, nproc=1):
        return Parallel(n_jobs=nproc, verbose=1)(
            delayed(generate_product)(self.graph, self.reaction_nodes, bb_map)
            for bb_map in list(self.bb_map_filtered(filter_func))
        )

    def apply_pharma_filters(self, pharma_filter):
        """
        Remove building blocks that would lead to problematic structures in the enumerated product
        pharma_filter (rdkit.Chem.rdfiltercatalog.FilterCatalog): a rdkit filter catalog
        """

        for leaf in self.leaves:
            options = leaf['options']
            query_no_lg = Chem.RWMol(Chem.MolFromSmarts(leaf['query']))
            atoms_to_rm = [a.GetIdx() for a in query_no_lg.GetAtoms() if a.GetAtomMapNum()==0]
            for a in sorted(atoms_to_rm, reverse=True):
                query_no_lg.RemoveAtom(a)
            no_lg_string = Chem.MolToSmarts(query_no_lg)

            if len(no_lg_string):
                rm_rxn = AllChem.ReactionFromSmarts("("+leaf['query'] + ')>>' + Chem.MolToSmarts(query_no_lg))
                
                #ensure that structures that are needed for reactivity are not filtered ou
                no_lg_leaf = rm_rxn.RunReactants([Chem.MolFromSmiles(leaf['smiles'])])[0][0]
                instantiate_mol(no_lg_leaf)
                
                filter_match = pharma_filter.GetFirstMatch(no_lg_leaf)
                while filter_match is not None:
                    pharma_filter.RemoveEntry(filter_match)
                    filter_match = pharma_filter.GetFirstMatch(no_lg_leaf)
                
                clean_options = []
                for option in options:
                    #remove the part of the molecule that will not make it to the product
                    mol = Chem.MolFromSmiles(option)
                    clean_mol = rm_rxn.RunReactants([mol])[0][0]
                    instantiate_mol(clean_mol)
                    if not pharma_filter.HasMatch(clean_mol):
                        clean_options.append(option)
                leaf['options'] = clean_options
    
    def has_regioselectivity_issues (self):
        g = self.graph.copy()
        for reaction_node in self.reaction_nodes:
            products = run_reaction_node(g, reaction_node, return_all_results=True)
            if len(np.unique(products)) > 1:
                return True
        return False




    
