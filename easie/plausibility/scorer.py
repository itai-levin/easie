#####################################
#File adapted from ASKCOS v. 2023.01#
#####################################
from abc import ABC, abstractmethod


class Scorer(ABC):
    """Interface for scorer classes."""

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Scores reactions with different contexts.

        Implemented method should return:

        * A list of results (one for each context).

          Each result should contain:

          * A list of outcomes, of which each outcome is a dictionnary
            containing:

            * rank
            * forward result
            * score
            * probability
        """
        raise NotImplementedError
