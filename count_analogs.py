import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from rxnmapper import RXNMapper
import argparse
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import *

from easie.utils.prop_conv_utils import *
from easie.building_blocks.file_pricer import FilePricer
from easie.graphenum import RxnGraphEnumerator
from easie.utils import askcos_utils

def get_args():
    options = argparse.ArgumentParser()
    options.add_argument("--path", dest="path_file", type=str, required=True)
    options.add_argument("--out", dest="out_prefix", type=str, required=True)
    options.add_argument("--leaf-out", dest="leaf_out_file", type=str, default=None)
    options.add_argument("--building-blocks", dest="building_blocks", type=str,
                        default="easie/building_blocks/buyables.json.gz") 
    options.add_argument("--mw-cutoff", dest='mw_cutoff', type=float, default=None)
    options.add_argument("--sim-thresh", dest='sim_thresh', type=float, default=None)
    return options.parse_args()


if __name__=='__main__':
 
    
    args = get_args()
    rxn_mapper = RXNMapper()
    with open(args.path_file, "r") as f:
        reaction_smiles = list(json.load(f))
    print (reaction_smiles)
    mapped_rsmi = rxn_mapper.get_attention_guided_atom_maps(reaction_smiles)
    mapped_rsmi = [m["mapped_rxn"] for m in mapped_rsmi]
    graph = RxnGraphEnumerator(mapped_rsmi)
  



    pricer = FilePricer()
    pricer.load(args.building_blocks, precompute_mols=True)
    graph.search_building_blocks(pricer)
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)
    graph.apply_pharma_filters(catalog)
   
    # get mw for each leaf
    leaf_mw_dict = {}
    for leaf in graph.leaves:
        for option in leaf['options']:
            mol = Chem.MolFromSmiles(option)
            mw = Descriptors.ExactMolWt(mol)
            leaf_mw_dict[option] = mw

    if args.sim_thresh is not None:
        graph.filter_by_similarity(threshold=args.sim_thresh)

    if args.mw_cutoff is not None:
        bb_mw_dists = [[leaf_mw_dict[o] for o in l['options']] for l in graph.leaves]
        prod_mw_dist = convolve_n_prop_lists(bb_mw_dists, 1)
        num_analogs = sum([y for y,x in zip(*prod_mw_dist ) if x <= args.mw_cutoff])
    
    if args.leaf_out_file is not None:
        with open(args.leaf_out_file, 'w') as f:
            print ("Writing leaf info to:", args.leaf_out_file)
            f.write(json.dumps(graph.leaves))

    else:
        num_analogs = graph.count_combinations()
    
    results = {'Similarity threshold:':float(args.sim_thresh),'MW cutoff:':int(args.mw_cutoff), 'Number of analogs:':int(num_analogs)}
    with open(args.out_prefix+'_enumeration_counting_summary.json', "w") as f:
        json.dump(results, f)
