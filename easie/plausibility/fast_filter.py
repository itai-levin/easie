#####################################
#File adapted from ASKCOS v. 2023.01#
#####################################

from tensorflow.keras.models import load_model

from .scorer import Scorer
from .fingerprinting import reac_prod_smi_to_morgan_fp

class FastFilterScorer(Scorer):
    """Fast filter to evaluate likelihood of reactions.

    Attributes:
        model (keras.engine.training.Model): Model used to evaluate reactions.
    """

    def __init__(self):
        """Initializes FastFilterScorer."""
        self.model = None

    def load(self, model_path):
        """Loads model from a file.

        Args:
            model_path (str): Path to file specifying model.
        """
        print ("Starting to load fast filter")
        self.model = load_model(model_path)
        print ("Done loading fast filter")

    def smiles_to_fp(self, reactant_smiles, target):
        """Generates fingerprints for the input reactant and target SMILES.

        Args:
            reactant_smiles (str): SMILES string of reactants.
            target (str): SMILES string of target product.

        Returns:
            np.ndarray, np.ndarray: product and reaction fingerprints with dtype of int8
        """
        pfp, rfp = reac_prod_smi_to_morgan_fp(
            reactant_smiles,
            target,
            length=2048,
            as_column=True,
            useFeatures=False,
            useChirality=False,
        )
        rxnfp = pfp - rfp
        return pfp, rxnfp

    def predict(self, reactant_smiles, target):
        """Run model prediction on the input reactant and target SMILES.

        Args:
            reactant_smiles (str): SMILES string of reactants.
            target (str): SMILES string of target product.

        Returns:
            float: output score from model
        """
        pfp, rxnfp = self.smiles_to_fp(reactant_smiles, target)
        return self.model([pfp, rxnfp]).numpy().item(0)

    def evaluate(self, reactant_smiles, target, **kwargs):
        """Evaluates likelihood of given reaction.

        Args:
            reactant_smiles (str): SMILES string of reactants.
            target (str): SMILES string of target product.
            **kwargs: Unused.

        Returns:
            A list of reaction outcomes.
        """
        score = self.predict(reactant_smiles, target)
        outcome = {"smiles": target, "template_ids": [], "num_examples": 0}
        all_outcomes = []
        all_outcomes.append(
            [
                {
                    "rank": 1.0,
                    "outcome": outcome,
                    "score": score,
                    "prob": score,
                }
            ]
        )
        return all_outcomes

    def evaluate_reaction_score(self, reaction_smiles, **kwargs):
        """
        Given a reaction SMILES string, return the score.
        Convenience wrapper for the ``evaluate`` method.
        """
        reactants, products = reaction_smiles.split(">>")
        all_outcomes = self.evaluate(reactants, products)
        return all_outcomes[0][0]["score"]

    def filter_with_threshold(self, reactant_smiles, target, threshold):
        """Filters reactions based on a score threshold.

        Args:
            reactant_smiles (str): SMILES string of reactants.
            target (str): SMILES string of target product.
            threshold (float): Value scores must be above to pass filter.

        Returns:
            2-tuple of (np.ndarray of np.ndarray of np.bool, float): Whether the
                reaction passed the filer and the score of the reaction.
        """
        score = self.predict(reactant_smiles, target)
        filter_flag = score > threshold
        return filter_flag, float(score)


if __name__ == "__main__":

    ff = FastFilterScorer()
    ff.load(model_path=gc.FAST_FILTER_MODEL["model_path"])
    score = ff.evaluate("CCO.CC(=O)O", "CCOC(=O)C")
    print(score)
    score = ff.evaluate(
        "[CH3:1][C:2](=[O:3])[O:4][CH:5]1[CH:6]([O:7][C:8]([CH3:9])=[O:10])[CH:11]([CH2:12][O:13][C:14]([CH3:15])=[O:16])[O:17][CH:18]([O:19][CH2:20][CH2:21][CH2:22][CH2:23][CH2:24][CH2:25][CH2:26][CH2:27][CH2:28][CH3:29])[CH:30]1[O:31][C:32]([CH3:33])=[O:34].[CH3:35][O-:36].[CH3:38][OH:39].[Na+:37]",
        "CCCCCCCCCCOC1OC(CO)C(O)C(O)C1O",
    )
    print(score)
    score = ff.evaluate(
        "CNC.Cc1ccc(S(=O)(=O)OCCOC(c2ccccc2)c2ccccc2)cc1", "CN(C)CCOC(c1ccccc1)c2ccccc2"
    )
    print(score)

    flag, sco = ff.filter_with_threshold("CCO.CC(=O)O", "CCOC(=O)C", 0.75)
    print(flag)
    print(sco)
