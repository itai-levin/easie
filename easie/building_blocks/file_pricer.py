#####################################
#File adapted from ASKCOS v. 2023.01#
#####################################

import os

import pandas as pd
from rdkit import Chem

from easie.building_blocks.base import BasePricer

pricer_loc = "file_pricer"


class FilePricer(BasePricer):
    """
    Building block database using local file as data source.
    """

    def __init__(self):
        """
        Initialize FilePricer instance.
        """
        self.path = None
        self.data = None
        self.smarts_query_index = {}

    def load(self, path, precompute_mols=False):
        """
        Load price data from local file.
        """
        if os.path.isfile(path):
            self.path = path
            self.data = pd.read_json(
                path,
                orient="records",
                dtype={"smiles": "object", "source": "object", "ppg": "float"},
                compression="gzip",
            )
            print("Loaded prices from flat file")
            self.indexed_queries = {}
        else:
            print(
                "Buyables file does not exist: {}".format(path), pricer_loc
            )

        if precompute_mols:
            self.data["mols"] = [Chem.MolFromSmiles(x) for x in self.data["smiles"]]

    def save(self, path):
        """
        Write price data to a local file.
        """
        self.data.to_json(path, orient="records", compression="gzip")

    def lookup_smarts(self, smarts, precomputed_mols=False):

        if smarts in self.smarts_query_index.keys():
            matches = self.smarts_query_index[smarts]

        elif precomputed_mols:
            pattern = Chem.MolFromSmarts(smarts)
            matches = self.data["mols"].apply(lambda x: x.HasSubstructMatch(pattern))
            self.smarts_query_index[smarts] = matches

        else:
            pattern = Chem.MolFromSmarts(smarts)
            matches = self.data["smiles"].apply(
                lambda x: Chem.MolFromSmiles(x).HasSubstructMatch(pattern)
            )
            self.smarts_query_index[smarts] = matches

        return self.data[matches].to_dict(orient="records")

    def lookup_smiles(
        self, smiles, source=None, canonicalize=True, isomeric_smiles=True
    ):
        """
        Lookup data for the requested SMILES, based on lowest price.

        Args:
            smiles (str): SMILES string to look up
            source (list or str, optional): buyables sources to consider;
                if ``None`` (default), include all sources, otherwise
                must be single source or list of sources to consider;
            canonicalize (bool, optional): whether to canonicalize SMILES string
            isomeric_smiles (bool, optional): whether to generate isomeric
                SMILES string when performing canonicalization

        Returns:
            dict: data for the requested SMILES, None if not found
        """
        if canonicalize:
            smiles = (
                self.canonicalize(smiles, isomeric_smiles=isomeric_smiles) or smiles
            )

        if source == []:
            # If no sources are allowed, there is no need to perform lookup
            # Empty list is checked explicitly here, since None means source
            # will not be included in query, and '' is a valid source value
            return None

        if self.data is not None:
            query = self.data["smiles"] == smiles

            if source is not None:
                if isinstance(source, list):
                    query = query & (self.data["source"].isin(source))
                else:
                    query = query & (self.data["source"] == source)

            results = self.data.loc[query]
            if len(results.index):
                idxmin = results["ppg"].idxmin()
                return results.loc[idxmin].to_dict()
            else:
                return None
        else:
            return None

    def lookup_smiles_list(self, smiles_list, *args, **kwargs):
        """
        Lookup prices for a list of SMILES.

        Returns:
            dict: mapping from input SMILES to price
        """
        raise NotImplementedError

    def search(self, *args, **kwargs):
        """
        Search the database based on the specified criteria.

        Returns:
            list: full documents of all buyables matching the criteria
        """
        raise NotImplementedError

    def list_sources(self):
        """
        Retrieve all available source names.

        Returns:
            list: list of source names
        """
        raise NotImplementedError

    def list_properties(self):
        """
        Retrieve all available property names.

        Note: Not all documents may have all properties defined.

        Returns:
            list: list of property names
        """
        raise NotImplementedError
