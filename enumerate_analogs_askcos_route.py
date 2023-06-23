import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from rxnmapper import RXNMapper
import argparse
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import *

from easie.utils import askcos_utils
from easie.utils.prop_conv_utils import *
from easie.building_blocks.file_pricer import FilePricer
from easie.graphenum import RxnGraphEnumerator

def get_args():
    options = argparse.ArgumentParser()
    options.add_argument("--askcos-output", dest="askcos_output", type=str)
    options.add_argument("--building-blocks", dest="building_blocks", type=str, 
                        default="easie/building_blocks/buyables.json.gz")
    options.add_argument("--out", dest="out_prefix", type=str,)
    options.add_argument("--sim-thresh", dest="sim", type=float, default=0)
    options.add_argument("--prop-filters", dest="prop_filters", action="store_true", default=False)
    options.add_argument("--nprocs", dest="nprocs", type=int, default=64)
    return options.parse_args()

args = get_args()

rxn_mapper = RXNMapper()
sep = "\t"
TP_KEY = "[{'template_set': 'reaxys'}]"
pricer = FilePricer()

pricer.load(args.building_blocks, precompute_mols=True)

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

    out_path = "{}{}_enumeration_results".format(args.out_prefix, clean_smiles)
    if not os.path.isdir(out_path):
        with open(out_path, 'w') as f:
            f.write("")
          
    else:
        print ('File {} already exists. Confirm that you want to overwrite by deleting it.'.format(out_path))
        raise 

    if len(paths):
        # get num analogs
        for path_id, path in tqdm(enumerate(paths)):
            
            # Get atom-mapped SMILES
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
            
            try:
                # Construct enumeration graph
                graph = RxnGraphEnumerator(cleaned_mapped_rsmi)
                graph.search_building_blocks(pricer)
                print ("Setting a similarity threshold on building blocks of", args.sim)
                graph.filter_by_similarity(threshold=args.sim)

                # filter    
                params = FilterCatalogParams()
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
                catalog = FilterCatalog(params)
                graph.apply_pharma_filters(catalog)

                if args.prop_filters:
                    print ("Applying additional pharmaceutical filters")
                    orig_reactants = [Chem.MolFromSmiles(leaf['smiles']) for leaf in graph.leaves]
                    p_mol = Chem.MolFromSmiles(graph.graph.nodes[graph.root]['smiles'])
                    
                    all_options = set.union(*[set(leaf['options']) for leaf in graph.leaves])
                    prop_dict = {s:{'mol':Chem.MolFromSmiles(s)} for s in all_options}

                    for s in prop_dict:
                        mol = prop_dict[s]['mol']
                        prop_dict[s]['tpsa'] = Descriptors.TPSA(mol) 
                        prop_dict[s]['rot_bonds'] = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
                        prop_dict[s]['MW'] = Chem.rdMolDescriptors.CalcExactMolWt(mol)
                        prop_dict[s]['h_donors'] = Lipinski.NumHDonors(mol) 
                        prop_dict[s]['h_acceptors'] = Lipinski.NumHAcceptors(mol)
                        prop_dict[s]['logp'] = Descriptors.MolLogP(mol) 

                    h_donors_correction = Lipinski.NumHDonors(p_mol) - np.sum([Lipinski.NumHDonors(x) for x in orig_reactants])
                    h_acceptors_correction = Lipinski.NumHAcceptors(p_mol) - np.sum([Lipinski.NumHAcceptors(x) for x in orig_reactants])
                    logp_correction = Descriptors.MolLogP(p_mol) - np.sum([Descriptors.MolLogP(x) for x in orig_reactants])
                    mw_correction =  Chem.rdMolDescriptors.CalcExactMolWt(p_mol) - np.sum([Chem.rdMolDescriptors.CalcExactMolWt(x) for x in orig_reactants])
                    rot_bonds_correction = Chem.rdMolDescriptors.CalcNumRotatableBonds(p_mol) - np.sum([Chem.rdMolDescriptors.CalcNumRotatableBonds(x) for x in orig_reactants])
                    tpsa_correction = Descriptors.TPSA(p_mol) - np.sum([Descriptors.TPSA(x) for x in orig_reactants])    

                    properties = ['tpsa', 'rot_bonds', 'MW', 'h_donors', 'h_acceptors', 'logp']
                    property_filters = [tpsa_correction, rot_bonds_correction, mw_correction, h_donors_correction, h_acceptors_correction, logp_correction]
                    property_ranges = [(0,150), (0, 10), (0,600), (0,6), (0,11), (-np.inf, 5)]

                    def prop_filter(list_of_building_blocks, property_dict, list_of_properties, list_of_corrections, list_of_ranges):
                        for prop, prop_correction, prop_range in zip(list_of_properties, list_of_corrections, list_of_ranges):
                            prop_sum = np.sum([property_dict[bb][prop] for bb in list_of_building_blocks]) + prop_correction
                            if prop_sum < prop_range[0] or prop_sum > prop_range[1]:
                                return False
                        return True
                    print ("Enumerating with filters")
                    filter_func = lambda x : prop_filter(x, prop_dict, properties, property_filters, property_ranges)
                    num_analogs = graph.count_combinations()
                    print ("Estimated number of analogs to enumerate: {}".format(num_analogs))
                    lib = graph.generate_library_filtered(nproc=args.nprocs, filter_func=filter_func)
                else:
                    lib = graph.generate_library(nproc=args.nprocs)
                with open (out_path, "a") as f:
                    json.dump(lib, f)
                    f.write("\n")
                        
            except:
              print("Could not run enumeration for path {}".format(path_id))

