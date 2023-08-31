from typing import List
import networkx as nx
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from rdchiral.main import rdchiralRun
from .template_extractor_local import get_strict_smarts_for_atom

VERBOSE = False


def mol2fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, useChirality=True)


def smi2fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol2fp(mol)


def bulk_similarity(smiles: str, query_smiles: List) -> List[float]:
    fp = smi2fp(smiles)
    fps = [smi2fp(smi) for smi in query_smiles]
    return Chem.DataStructs.BulkTanimotoSimilarity(fp, fps)


def bulk_plausibility(
    fixed_reactant_smiles: str, rxn, variable_smiles_list: List[str], forward_filter
):
    rxn = rdchiralReaction(rxn)
    # for now only using fast_filtering
    if len(fixed_reactant_smiles) > 0:
        joined_reactants = [
            fixed_reactant_smiles + "." + smiles for smiles in variable_smiles_list
        ]
    # single reactant
    else:
        joined_reactants = variable_smiles_list

    rdchiral_reactants = [rdchiralReactants(smiles) for smiles in joined_reactants]
    products = []
    for reactants in rdchiral_reactants:
        res = rdchiralRun(rxn, reactants)
        if len(res):
            products.append(res[0])
        else:
            products.append("")
            print("WARNING: Error applying template to reactants. Skipping reactants.")
    plausibilities = [
        forward_filter(reactants, products)
        for reactants, products in zip(joined_reactants, products)
    ]
    return plausibilities


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    _ = [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)


def reverse_template(smarts):
    left, _, right = smarts.split(">")
    if "." in right and not right.startswith("("):
        right = f"({right})"
    return f"{right}>>{left}"


def run_reaction_node(graph, node, constrain_smiles=False, return_all_results=False):
    template = graph.nodes[node].get("template_reverse")
    rxn = rdchiralReaction(template)
    reactant_node_ids = list(graph.succ[node])
    reactants = [graph.nodes[i]["smiles"] for i in reactant_node_ids]
    if any([len(x) == 0 for x in reactants]):
        return ""
    reactant_smiles = ".".join(reactants)
    reactants = rdchiralReactants(reactant_smiles)
    result = rdchiralRun(rxn, reactants)
    result = list(sorted(result, key=lambda x: len(x), reverse=True))
    ### REMOVE PRODUCTS THAT ARE SPLIT (e.g. intramolecular reaction)
    result = [r for r in result if '.' not in r]
    if len(result) == 0:
        if VERBOSE:
            print(
                "ERROR: found no product results for reactants: {} with reaction: {}".format(
                    reactant_smiles, template
                )
            )
        else:
            print("ERROR: found no product results")
        # pdb.set_trace()
        return ""
    elif not constrain_smiles:
        if return_all_results:
            return result
        if len(result) > 1:
            if VERBOSE:
                print("WARNING: multiple products found, using first product found")
        return result[0]
    else:
        if VERBOSE:
            print("Constraining product to match original")
        known_prod_node = list(graph.pred[node])[0]
        known_prod = graph.nodes[known_prod_node]["smiles"]
        for res in result:
            m = Chem.MolFromSmiles(res)
            if m:
                for a in m.GetAtoms():
                    a.SetIsotope(0)
                if Chem.MolToSmiles(m) == known_prod:
                    if VERBOSE:
                        print("Found match!")
                    return res
        return result[0]


def generate_product(graph, reaction_nodes, bb_map: dict = {}, constrain_smiles=False):
    g = graph.copy()
    nx.set_node_attributes(g, bb_map, name="smiles")
    for reaction_node in reaction_nodes:
        product = run_reaction_node(g, reaction_node, constrain_smiles=constrain_smiles)
        parent = list(g.pred[reaction_node])[0]
        nx.set_node_attributes(g, {parent: product}, name="smiles")
    result = {"smiles": product}
    result.update(bb_map)
    return result


def generalize_query(query_string):
    """
    Move designated atoms from the query molecule into a recursive SMARTS
    pattern
    """
    return "[*;$({})]".format(query_string)

def merge_generalized_queries(list_of_queries):
    """
    Merge two generalized queries into a single query
    Generalized queries must have the form: '[*;$(<pattern>)]'
    """
    patts = [';$('+q[5:-2]+')' for q in list_of_queries]
    merged = "[*"+"".join(patts)+"]"
    return merged

def rm_unmapped_atoms_from_rwmol(rwmol):
    atoms_to_rm = []
    for atom in rwmol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atoms_to_rm.append(atom.GetIdx())
    for idx in sorted(atoms_to_rm, reverse=True):
        rwmol.RemoveAtom(idx)
    return rwmol


def rm_unmapped_atoms_from_reaction_smarts(smarts):
    products = smarts.split(">>")[1].split(".")
    reactants = smarts.split(">>")[0].split(".")

    products = [Chem.RWMol(Chem.MolFromSmarts(p)) for p in products]
    reactants = [Chem.RWMol(Chem.MolFromSmarts(r)) for r in reactants]

    products = [Chem.MolToSmarts(rm_unmapped_atoms_from_rwmol(p)) for p in products]
    reactants = [Chem.MolToSmarts(rm_unmapped_atoms_from_rwmol(r)) for r in reactants]

    smarts = ".".join(reactants) + ">>" + ".".join(products)
    smarts = re.sub("&D[0-9]", "", smarts)
    return smarts


def get_atom_ids_for_substruct_match(mol, match, map_nums=None, keep_unmapped=False):
    atom_ids = set()
    if map_nums is None:
        map_nums = [1] * len(match)
    assert len(match) == len(map_nums)
    for idx, map_num in zip(match, map_nums):
        isotope_num = mol.GetAtomWithIdx(idx).GetIsotope()
        if map_num != 0 or keep_unmapped:
            atom_ids.add(isotope_num)
    return atom_ids

def flatten_list(list_of_lists):
    return [x for y in list_of_lists for x in y]

def get_substruct_match_idxs (reactant_smarts, reactants):
    reactant_substruct_idxs = []
    reactant_substruct_idxs_w_lg = []
    for smarts in reactant_smarts:
        map_nums = [a.GetAtomMapNum() for a in smarts.GetAtoms()]
        possible_match_idxs = []
        possible_match_idxs_w_lg = []
        for reactant in reactants:
            matches = reactant.GetSubstructMatches(smarts, uniquify=False)
            for match in matches:
                possible_match_idxs.append(
                    get_atom_ids_for_substruct_match(reactant, match,  map_nums)
                )
                possible_match_idxs_w_lg.append(
                    get_atom_ids_for_substruct_match(
                        reactant, match, map_nums, keep_unmapped=True
                    )
                )
        reactant_substruct_idxs.append(possible_match_idxs)
        reactant_substruct_idxs_w_lg.append(possible_match_idxs_w_lg)
    return reactant_substruct_idxs, reactant_substruct_idxs_w_lg

def get_strict_symbols_for_mol (mol, rm_iso=True):
    symbols = []
    iso_patt = re.compile("(?<=\[)[0-9]+(?=[A-Z,a-z])")
    for atom in mol.GetAtoms():
        strict_symbol = get_strict_smarts_for_atom(atom)
        if rm_iso:
            symbols.append(re.sub(iso_patt,"", strict_symbol))
        else:
            symbols.append(strict_symbol)
    return symbols

def instantiate_mol (mol):
    mol.UpdatePropertyCache()
    Chem.GetSymmSSSR(mol)
    mol.GetRingInfo().NumRings()
