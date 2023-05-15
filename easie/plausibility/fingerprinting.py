#####################################
#File adapted from ASKCOS v. 2023.01#
#####################################

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np



def mol_smi_to_morgan_fp(
    smiles,
    radius=2,
    length=2048,
    as_column=False,
    raise_exceptions=False,
    dtype="float32",
    **kwargs,
):
    """
    Create Morgan Fingerprint from molecule SMILES.
    Returns correctly shaped zero vector on errors.

    Args:
        smiles (str): input molecule SMILES
        radius (int, optional): fingerprint radius, default 2
        length (int, optional): fingerprint length, default 2048
        as_column (bool, optional): return fingerprint as column vector
        raise_exceptions (bool, optional): raise exceptions instead of returning zero vector
        dtype (str, optional): data type of the generated fingerprint array
        **kwargs: passed to GetMorganFingerprintAsBitVect

    Returns:
        np.array of shape (length,) or (1, length) if as_column = True
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print (
            f"Unable to parse SMILES {smiles}: {e!s}"
        )
        if raise_exceptions:
            raise
        fp = np.zeros(length, dtype=dtype)
    else:
        try:
            fp_bit = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=radius,
                nBits=length,
                **kwargs,
            )
            fp = np.empty(length, dtype=dtype)
            DataStructs.ConvertToNumpyArray(fp_bit, fp)
        except Exception as e:
            print (
                f"Unable to generate fingerprint for {smiles}: {e!s}"
            )
            if raise_exceptions:
                raise
            fp = np.zeros(length, dtype=dtype)

    if as_column:
        return fp.reshape(1, -1)
    else:
        return fp


def reac_prod_smi_to_morgan_fp(
    rsmi,
    psmi,
    radius=2,
    length=2048,
    as_column=False,
    raise_exceptions=False,
    **kwargs,
):
    """
    Create Morgan Fingerprints from reactant and product SMILES separately.

    Args:
        rsmi (str): reactant molecule SMILES
        psmi (str): product molecule SMILES
        radius (int, optional): fingerprint radius, default 2
        length (int, optional): fingerprint length, default 2048
        as_column (bool, optional): return fingerprints as column vector
        raise_exceptions (bool, optional): raise exceptions instead of returning zero vector
        **kwargs: passed to GetMorganFingerprintAsBitVect

    Returns:
        product np.array of shape (length,) or (1, length) if as_column = True
        reactant np.array of shape (length,) or (1, length) if as_column = True
    """
    rfp = mol_smi_to_morgan_fp(
        rsmi,
        radius=radius,
        length=length,
        as_column=as_column,
        raise_exceptions=raise_exceptions,
        **kwargs,
    )
    pfp = mol_smi_to_morgan_fp(
        psmi,
        radius=radius,
        length=length,
        as_column=as_column,
        raise_exceptions=raise_exceptions,
        **kwargs,
    )
    return pfp, rfp
