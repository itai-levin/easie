import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

def prop_list_to_dist (prop_list, bin_size, conv_min=None):
    """
    Convert a list of values into a discretized property distribution with a given bin size
    Note: currently properties must be positive
    """
    if conv_min is None:
        conv_min = np.min(prop_list)
    max_bin = bin_size * np.ceil(np.max(prop_list) / bin_size)
    conv_min = bin_size * np.floor(conv_min / bin_size)
    bin_num = int((max_bin-conv_min) / bin_size)
    
    
    hist = np.histogram(prop_list, bins=bin_num, range=(conv_min,max_bin))
    
    return hist
    
def convolve_2_prop_dists (prop_dist_1, prop_dist_2, bin_size):
    #align the two probability distributions
    prop_probs_1, prop_vals_1 = prop_dist_1
    prop_probs_2, prop_vals_2 = prop_dist_2
    
    min_val = min(np.min(prop_vals_1), np.min(prop_vals_2))
    
    conv_min = np.min(prop_vals_1) + np.min(prop_vals_2)
    
    pad_length_1 = int(np.ceil((np.min(prop_vals_1) - min_val)/bin_size))
    pad_length_2 = int(np.ceil((np.min(prop_vals_2) - min_val)/bin_size))
    
    if pad_length_1 > 0:
        prop_probs_1 = np.pad(prop_probs_1, (pad_length_1, 0))
        prop_vals_1 = np.pad(prop_vals_1, (pad_length_1, 0), mode="linear_ramp", end_values=min_val)

    if pad_length_2 > 0:
        prop_probs_2 = np.pad(prop_probs_2, (pad_length_2, 0))    
        prop_vals_2 = np.pad(prop_vals_2, (pad_length_2, 0), mode="linear_ramp", end_values=min_val)
    
    #convolve
    conv_prop_probs = np.convolve(prop_probs_1, prop_probs_2)
    conv_max = np.max(prop_vals_1)+np.max(prop_vals_2)
    conv_min = np.min(prop_vals_1)+np.min(prop_vals_2)
    conv_prop_vals = np.arange(conv_min, conv_max, bin_size)[:len(prop_vals_1)+len(prop_vals_2)-2]
    return (conv_prop_probs, conv_prop_vals)

def convolve_2_prop_lists (prop_list_1, prop_list_2, bin_size):
    prop_dist_1  = prop_list_to_dist(prop_list_1, bin_size, np.min(prop_list_1))
    prop_dist_2  = prop_list_to_dist(prop_list_2, bin_size, np.min(prop_list_2))
    return convolve_2_prop_dists(prop_dist_1, prop_dist_2, bin_size)

def convolve_n_prop_lists (prop_lists, bin_size):
    
    assert len(prop_lists) >= 2
    
    prop_dists = [prop_list_to_dist(prop_list, bin_size) for prop_list in prop_lists]
    
    accumulator = convolve_2_prop_dists(prop_dists[0], prop_dists[1], bin_size)
    
    for prop_dist in prop_dists[2:]:
        accumulator = convolve_2_prop_dists(accumulator, prop_dist, bin_size)
    return accumulator
    

def initialize_mol_props (mol):
    mol.UpdatePropertyCache()
    Chem.GetSymmSSSR(mol)
    mol.GetRingInfo().NumRings()

def get_morgan_fingerprint(mol, radius=2, bits=2048):
    fp_arr = np.zeros(bits)
    initialize_mol_props(mol)
    return AllChem.GetMorganFingerprintAsBitVect(mol,radius)

def keep_only_lg (smarts_query_obj):
    rwmol = Chem.RWMol(smarts_query_obj)
    atoms = [a for a in rwmol.GetAtoms()]
    for a in atoms:
        if a.GetAtomMapNum()!=0:
            rwmol.RemoveAtom(a.GetIdx())
    return rwmol

def rm_lg (smarts_query_obj):
    rwmol = Chem.RWMol(smarts_query_obj)
    atoms = [a for a in rwmol.GetAtoms()]
    for a in atoms:
        if a.GetAtomMapNum()==0:
            rwmol.RemoveAtom(a.GetIdx())
    return rwmol

def get_reactant_smarts (reaction_smarts):
    r_smarts_strings = reaction_smarts.split('>>')[0].split('.')
    return [Chem.MolFromSmarts(r) for r in r_smarts_strings]

def remove_LGs (reactant, smarts_patterns):
    """
    Removes the parts of the molecule that will be removed
    when run through the synthesis tree
    """
    all_matches = set()
    for patt in smarts_patterns:
        match = set(reactant.GetSubstructMatch(patt))
        all_matches = all_matches.union(match)
    atoms_to_include = [a.GetIdx() for a in reactant.GetAtoms() if a.GetIdx() not in all_matches]
    return Chem.MolFragmentToSmiles(reactant, atoms_to_include)
    
    
def get_predicted_histogram(reactant_lists, reactant_prop_dict, property_name, step_size=1, correction_factor=0):
    r_prop_values = []
    for smiles_list in reactant_lists:
        r_prop_values.append(np.array([reactant_prop_dict[s][property_name] for s in smiles_list]))

    r_prop_hists = [prop_list_to_dist(v, step_size) for v in r_prop_values]
    r_prop_values[0] = r_prop_values[0] + correction_factor
    pred_prop_hist = convolve_n_prop_lists(r_prop_values, step_size)
    return pred_prop_hist


def plot_prop_hist (prop_hist, normalize=False, new_figure=False, **kwargs):
    if new_figure:
        plt.figure()
    if normalize:
        plt.bar(prop_hist[1][:-1], prop_hist[0]/np.sum(pred_prop_hist[0]), **kwargs)
    else:
        plt.bar(prop_hist[1][:-1], prop_hist[0], **kwargs)
    

def compare_conv_and_enum (reactant_lists, reactant_prop_dict, property_name, step_size=1, 
                           normalize=False, correction_factor=0,show_reactants=False, **kwargs):
    pred_prop_hist = get_predicted_histogram(reactant_lists, reactant_prop_dict, property_name, step_size)
    plot_prop_hist (pred_prop_hist, normalize, label='Predicted Product', width=step_size, **kwargs)
    
    
    
def plot_dists_with_highlighted (leaves, prop_dict, prop, step_size, correction_factor, lb, ub, cdf_thresh=0.005, 
                                 cdf_cutoff=0.995, bb_prop_filter = None, figsize=(2,2), **kwargs):
    options = [leaf['options'] for leaf in leaves]
    if bb_prop_filter is not None:
        options = [[x for x in leaf_options if bb_prop_filter(prop_dict[x][prop])] for leaf_options in options]
    hist = get_predicted_histogram(options, prop_dict, prop, step_size=step_size, correction_factor=correction_factor)
    hist_within = [hist[0][i] if (hist[1][i]>=lb and hist[1][i]<=ub) else 0 for i in range(len(hist[0]))]
    hist_without = [hist[0][i] if (hist[1][i]<lb or hist[1][i]>ub) else 0 for i in range(len(hist[0]))]
    print ("Total size of space:", np.sum(hist[0]))
    print ("Molecules within bounds:", np.sum(hist_within))
    print ("Molecules out of bounds:", np.sum(hist_without))
    
    cdf = [np.sum(hist[0][:i])/np.sum(hist[0]) for i in range(len(hist[0]))]
    for i, c in enumerate(cdf):
        if c > cdf_thresh:
            xstart = hist[1][i]
            break
    for i, c in enumerate(cdf):
        if c > cdf_cutoff:
            xcutoff = hist[1][i]
            break

    plt.figure(figsize=figsize)
    plot_prop_hist([hist_within, hist[1]], new_figure=False,  width=step_size, color='red', **kwargs)
    plot_prop_hist([hist_without, hist[1]], new_figure=False,  width=step_size, color='gray', **kwargs)
    plt.tick_params('y', direction='in')
    plt.xlim(xstart, xcutoff)
    
    
    
def plot_outline (x, y ,step_size, color="black"):
    ys = []
    xs = []
    xs.append(x[0] - 0.5*step_size)
    ys.append(0)
    for i in range(len(x)):
        xs.append(x[i] - 0.5*step_size)
        xs.append(x[i] + 0.5*step_size)
        ys.append(y[i])
        ys.append(y[i])
    xs.append(x[-1] + 0.5*step_size)
    ys.append(0)
    plt.plot(xs, ys, color=color, linewidth=0.5)
    
def plot_dists (leaves, prop_dict, prop, step_size, correction_factor, lb, ub, cdf_thresh=0.005, 
                    cdf_cutoff=0.995, bb_prop_filter = None, figsize=(2,2), new_figure=False, col='blue', 
                    alpha=1, **kwargs):
    options = [leaf['options'] for leaf in leaves]
    if bb_prop_filter is not None:
        options = [[x for x in leaf_options if bb_prop_filter(prop_dict[x][prop])] for leaf_options in options]
    hist = get_predicted_histogram(options, prop_dict, prop, step_size=step_size, correction_factor=correction_factor)
    hist_within = [hist[0][i] if (hist[1][i]>=lb and hist[1][i]<=ub) else 0 for i in range(len(hist[0]))]
    hist_without = [hist[0][i] if (hist[1][i]<lb or hist[1][i]>ub) else 0 for i in range(len(hist[0]))]
    print ("Total size of space:", np.sum(hist[0]))
    print ("Molecules within bounds:", np.sum(hist_within))
    print ("Molecules out of bounds:", np.sum(hist_without))
    
    cdf = [np.sum(hist[0][:i])/np.sum(hist[0]) for i in range(len(hist[0]))]
    for i, c in enumerate(cdf):
        if c > cdf_thresh:
            xstart = hist[1][i]
            break
    for i, c in enumerate(cdf):
        if c > cdf_cutoff:
            xcutoff = hist[1][i]
            break
    if new_figure:
        plt.figure(figsize=figsize)
    plot_prop_hist([hist[0], hist[1]], new_figure=False, color=col, width=step_size,  alpha=alpha, **kwargs)
    plot_outline(hist[1][:-1],hist[0], step_size)
    plt.tick_params('y', direction='in')
    return xstart, xcutoff