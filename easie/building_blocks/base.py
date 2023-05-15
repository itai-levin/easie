#####################################
#File adapted from ASKCOS v. 2023.01#
#####################################

from abc import ABC, abstractmethod

import rdkit.Chem as Chem


class BasePricer(ABC):
    """
    Abstract base class for building block data.
    """

    @staticmethod
    def canonicalize(smiles, isomeric_smiles=True):
        """
        Canonicalize the input SMILES.

        Returns:
            str: canonicalized SMILES or empty str on failure
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)
        except Exception:
            return ""
        else:
            return smiles

    def get_price(self, smiles, *args, **kwargs):
        """
        Lookup the lowest price for the requested SMILES.

        Returns:
            float: lowest price for the requested SMILES, 0 if not found
        """
        result = self.lookup_smiles(smiles, *args, **kwargs)
        if result is not None:
            return result["ppg"]
        else:
            return 0.0

    @abstractmethod
    def lookup_smiles(self, smiles, *args, **kwargs):
        """
        Lookup data for the requested SMILES, based on lowest price.

        Returns:
            dict: data for the requested SMILES, None if not found
        """
        raise NotImplementedError

    @abstractmethod
    def lookup_smiles_list(self, smiles_list, *args, **kwargs):
        """
        Lookup data for a list of SMILES, based on lowest price for each.

        Returns:
            dict: mapping from input SMILES to data dict
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, *args, **kwargs):
        """
        Search the database based on the specified criteria.

        Returns:
            list: full documents of all buyables matching the criteria
        """
        raise NotImplementedError

    @abstractmethod
    def list_sources(self):
        """
        Retrieve all available source names.

        Returns:
            list: list of source names
        """
        raise NotImplementedError

    @abstractmethod
    def list_properties(self):
        """
        Retrieve all available property names.

        Note: Not all documents may have all properties defined.

        Returns:
            list: list of property names
        """
        raise NotImplementedError

    def get(self, _id):
        """
        Get a single entry by its _id.
        """
        raise NotImplementedError

    def update(self, _id, new_doc):
        """
        Update a single entry by its _id.
        """
        raise NotImplementedError

    def delete(self, _id):
        """
        Delete a single entry by its _id.
        """
        raise NotImplementedError

    def add(self, new_doc, allow_overwrite=True):
        """
        Add a new entry to the database.
        """
        raise NotImplementedError

    def add_many(self, new_docs, allow_overwrite=True):
        """
        Add a list of new entries to the database.
        """
        raise NotImplementedError
