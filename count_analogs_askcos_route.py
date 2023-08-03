import sys
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from easie.utils import askcos_utils
from rxnmapper import RXNMapper
from rdkit import Chem
import argparse
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import *
from easie.utils.prop_conv_utils import *
from easie.building_blocks.file_pricer import FilePricer
from easie.graphenum import RxnGraphEnumerator

def get_args():
    options = argparse.ArgumentParser()
    options.add_argument("--askcos-output", dest="askcos_output", type=str)
    options.add_argument("--building-blocks", dest="building_blocks", type=str, 
                        default="easie/building_blocks/buyables.json.gz")
    options.add_argument("--out", dest="out_prefix", 
                         help="Prefix for path for analysis summary output", type=str)
    options.add_argument("--mw-cutoff", dest='mw_cutoff', type=float, default=None)
    options.add_argument("--sim-thresh", dest='sim_thresh', type=float, default=None)
    options.add_argument("--brenk-filters", dest="brenk_filters", action="store_true", default=False)
    return options.parse_args()


args = get_args()

rxn_mapper = RXNMapper()
sep = "\t"
TP_KEY = "[{'template_set': 'reaxys'}]"
pricer = FilePricer()

print ("Initializing buyables database")
pricer.load(args.building_blocks, precompute_mols=True)
print ("Done initializing buyables database")

with open(args.askcos_output, "r") as f:
    askcos_results = f.readlines()

for result in askcos_results:
    result = json.loads(result)
    smiles = list(result.keys())[0]
    if "output" in result[smiles][TP_KEY].keys():
        paths = result[smiles][TP_KEY]["output"]
        print("{} paths found for {}".format(len(paths), smiles))
    else:
        print("No routes found for {}".format(smiles))

    clean_smiles = smiles.replace("/", "").replace("\\", "")
    out_file = "{}_{}_pathways_summaries.csv".format(
        args.out_prefix,
        clean_smiles
    )
    leaf_out_file = "{}_{}_leaves_summaries.json".format(
        args.out_prefix,
        clean_smiles
    )
    reaction_fields_of_interest = ["template_score", "plausibility", "scscore"]
    chemical_fields_of_interest = ["ppg"]
    header = "route_id" + sep
    header += sep.join(reaction_fields_of_interest) + sep
    header += sep.join(chemical_fields_of_interest)
    header += sep + "regioselectivity_concern" + sep + "path_length" + sep + "num_analogs\n"
    if len(paths): 
        leaf_mw_dict = {}
        with open(out_file, "w") as f:
            f.write(header)
        with open(leaf_out_file, "w") as f:
            f.write("")
        for path_id, path in tqdm(enumerate(paths)):
            output_line = [str(path_id)]

            reaction_prop_dict = {}
            chemical_prop_dict = {}

            for field in reaction_fields_of_interest:
                reaction_prop_dict[field] = askcos_utils.traverse_pathway(path, field)
                output_line.append(str(reaction_prop_dict[field]))

            path_length = len(askcos_utils.traverse_pathway(path, field))

            for field in chemical_fields_of_interest:
                chemical_prop_dict[field] = askcos_utils.traverse_pathway(path, field, False)
                output_line.append(str(chemical_prop_dict[field]))

            
            # get num analogs
            rsmi = askcos_utils.traverse_pathway(path, "smiles")
            mapped_rsmi = rxn_mapper.get_attention_guided_atom_maps(rsmi)
            mapped_rsmi = [m["mapped_rxn"] for m in mapped_rsmi]

            cleaned_mapped_rsmi = []
            for r in mapped_rsmi:
                reactants, products = r.split(">>")
                reactants = Chem.MolFromSmiles(reactants)
                products = Chem.MolFromSmiles(products)

                r_atom_map_numbers = [a.GetAtomMapNum() for a in reactants.GetAtoms()]
                for a in products.GetAtoms():
                    if a.GetAtomMapNum() not in r_atom_map_numbers:
                        a.SetAtomMapNum(0)

                cleaned_mapped_rsmi.append(
                    Chem.MolToSmiles(reactants) + ">>" + Chem.MolToSmiles(products)
                )
#             try:
            graph = RxnGraphEnumerator(cleaned_mapped_rsmi)
            graph.search_building_blocks(pricer)
            issue = graph.has_regioselectivity_issues()
            if args.brenk_filters:
                params = FilterCatalogParams()
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
                catalog = FilterCatalog(params)
                graph.apply_pharma_filters(catalog)
            if args.sim_thresh is not None:
                graph.filter_by_similarity(threshold=args.sim_thresh)
            if args.mw_cutoff is not None:
                for leaf in graph.leaves:
                    for option in leaf['options']:
                        if option not in leaf_mw_dict.keys():
                            mol = Chem.MolFromSmiles(option)
                            mw = Descriptors.ExactMolWt(mol)
                            leaf_mw_dict[option] = mw
                bb_mw_dists = [[leaf_mw_dict[o] for o in l['options']] for l in graph.leaves]
                orig_bb_mws = sum([leaf_mw_dict[l['smiles']] for l in graph.leaves])
                orig_p_mw = Descriptors.ExactMolWt(Chem.MolFromSmiles(graph.root_smiles))
                mw_correction = orig_p_mw - orig_bb_mws
                prod_mw_dist = convolve_n_prop_lists(bb_mw_dists, 1)
                num_analogs = sum([y for y,x in zip(*prod_mw_dist ) if x+mw_correction <= args.mw_cutoff])
            else:
                num_analogs = graph.count_combinations()


            output_line.append(str(issue))
            output_line.append(str(path_length))
            output_line.append(str(num_analogs))
#             except:
#                 print("Could not run enumeration counting for path {}".format(path_id))
#                 output_line.append("0")

            with open(out_file, "a") as f:
                f.write(sep.join(output_line) + "\n")

            with open(leaf_out_file, "a") as f:
                f.write(json.dumps(graph.leaves) + "\n")

    else:
        print("No paths found")
