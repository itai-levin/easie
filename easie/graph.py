import networkx as nx
from rdkit import Chem
from typing import List
from .utils import canonicalize_smiles, reverse_template
from .template_extractor_local import extract_from_reaction

def graph_from_reaction_smiles(reaction_smiles: List[str], mapper=None) -> nx.DiGraph:
    graph = nx.DiGraph()
    for rsmi in reaction_smiles:
        if not is_mapped(rsmi) and mapper:
            rsmi = mapper(rsmi)
        reactants, spectators, products = rsmi.split(">")
        reactants, products =  remove_unmatched_mappings(reactants, products)
        template = extract_from_reaction(
            {"_id": None, "reactants": products, "products": reactants}
        )
        if template.get("reaction_smarts") is not None:
            tp, tr = template['reaction_smarts'].split('>>')
            template['reaction_smarts'] = '>>'.join([tr, tp])
        else:
            continue
        reactants = list(map(canonicalize_smiles, reactants.split(".")))
        products = canonicalize_smiles(products)
        canonicalized_rsmi = ">".join([".".join(reactants), spectators, products])
        graph.add_node(
            canonicalized_rsmi,
            smiles=canonicalized_rsmi,
            template=template["reaction_smarts"],
            template_reverse=reverse_template(template["reaction_smarts"]),
            spectators=spectators,
            type="reaction",
        )
        for react in reactants:
            graph.add_edge(canonicalized_rsmi, react)
            graph.nodes[react]["smiles"] = react
            graph.nodes[react]["type"] = "chemical"
        graph.add_edge(products, canonicalized_rsmi)
        graph.nodes[products]["smiles"] = products
        graph.nodes[products]["type"] = "chemical"

    smiles_to_uid = {s: u + 1 for u, s in enumerate(graph.nodes)}
    nx.relabel_nodes(graph, smiles_to_uid, copy=False)
    split_repeated_nodes(graph)
    return graph


def graph_from_reaction_smarts(
    reaction_smiles: List[str], reaction_smarts: List[str]
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for rsmi, rsmarts in zip(reaction_smiles, reaction_smarts):
        reactants, spectators, products = rsmi.split(">")
        reactants = list(map(canonicalize_smiles, reactants.split(".")))
        products = canonicalize_smiles(products)
        canonicalized_rsmi = ">".join([".".join(reactants), spectators, products])
        graph.add_node(
            canonicalized_rsmi,
            smiles=canonicalized_rsmi,
            template=rsmarts,
            template_reverse=reverse_template(rsmarts),
            spectators=spectators,
            type="reaction",
        )
        for react in reactants:
            graph.add_edge(canonicalized_rsmi, react)
            graph.nodes[react]["smiles"] = react
            graph.nodes[react]["type"] = "chemical"
        graph.add_edge(products, canonicalized_rsmi)
        graph.nodes[products]["smiles"] = products
        graph.nodes[products]["type"] = "chemical"

    smiles_to_uid = {s: u + 1 for u, s in enumerate(graph.nodes)}
    nx.relabel_nodes(graph, smiles_to_uid, copy=False)
    split_repeated_nodes(graph)
    return graph


def split_repeated_nodes(graph):
    """
    Splits leaf nodes that are used in muliple reactions into multiple nodes
    Note: it may be important to perform a similar correction on a whole sub-tree
        if an intermediate node is similarly used in multiple reactions
    """
    uid = max(graph.nodes) + 1
    for n, d in dict(graph.in_degree).items():
        if d > 1 and graph.nodes[n]["type"] == "chemical" and graph.out_degree(n) == 0:
            print("Splitting reactant node", n)
            orig_node = graph.nodes[n]
            for r in graph.predecessors(n):
                graph.add_node(uid, **orig_node)
                graph.add_edge(r, uid)
                uid += 1
            graph.remove_node(n)


def is_mapped(reaction_smiles):
    r, _, p = reaction_smiles.split(">")
    r_mol = Chem.MolFromSmiles(r)
    p_mol = Chem.MolFromSmiles(p)
    if all([a.GetAtomMapNum() == 0 for a in r_mol.GetAtoms()]) and all(
        [a.GetAtomMapNum() == 0 for a in p_mol.GetAtoms()]
    ):
        return False

    return True

def remove_unmatched_mappings(r, p):
    r = Chem.MolFromSmiles(r)
    p = Chem.MolFromSmiles(p)
    r_maps = [a.GetAtomMapNum() for a in r.GetAtoms()]
    p_maps = [a.GetAtomMapNum() for a in p.GetAtoms()]
    [a.SetAtomMapNum(0) for a in r.GetAtoms() if a.GetAtomMapNum() not in p_maps]
    [a.SetAtomMapNum(0) for a in p.GetAtoms() if a.GetAtomMapNum() not in r_maps]
    r = Chem.MolToSmiles(r)
    p = Chem.MolToSmiles(p)
    return r, p

def get_reaction_path(graph, root_node, leaf_node):
    paths = list(nx.shortest_simple_paths(graph, root_node, leaf_node))
    if len(paths):
        return [
            graph.nodes.get(n_id).get("template")
            for n_id in paths[0]
            if graph.nodes[n_id]["type"] == "reaction"
        ]
    return
