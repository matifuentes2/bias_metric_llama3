
from typing import Callable, Union, List, Optional, Dict
import abc
import dataclasses


@dataclasses.dataclass
class ScoreData:
    """Data structure containing task evaluation results.
    `low_score` and `high_score` will be used to help calibrate scores when
    comparing or aggregating scores across tasks. They do not need to be
    lower and upper bounds on the score. It is OK if the task returns a score
    outside of this range.
    Fields:
      `score_dict`: a dictionary of (str) : (float) entries that report the scores
        for this task e.g., `"custom_score" , "bleu" , "rouge" , "exact_str_match"
        , "perplexity" , "multiple_choice_grade"`.
      `preferred_score`: the preferred key of score_dict to use for evaluation,
        e.g., `"bleu"`.
      `low_score`: a score corresponding to poor, or chance level, performance on
        the task, for the score stored as the `preferred_score entry` in `score_dict`.
      `high_score`: a score corresponding to extremely good, potentially superhuman,
        performance on the task, for the score stored as the `preferred_score` entry
        in `score_dict`.
    """
    score_dict: Dict[str, float]
    preferred_score: str
    low_score: float
    high_score: float
    score_name: str


@dataclasses.dataclass
class MetricMetadata:
    """Data structure containing task meta-data such as name and description.
    Fields:
      `name`: a short, human-readable name for your metric.
      `description`: a one to two sentence English language description of the metric.
        if you include a novel scoring scheme in the dictionary of scores returned
        by your task, please describe it here briefly.
      `keywords`: a list of strings, where each string contains a separate keyword
        describing the metric.
      `paper`: paper in pdf
    """

    name: str
    description: str
    keywords: List[str]
    paper: str


class Metric(abc.ABC):
    """The base class for defining a metric.
        Extend this class to implement a custum bias metric.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_metric_details(self) -> MetricMetadata:
        """Provide meta-information describing the metric.
        The meta-information returned describes the metric.
        Returns:
          metadata: A MetricMetadata dataclass with all fields populated.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def evaluate_model(self, model, tokenizer) -> Union[ScoreData, List[ScoreData]]:
        """Evaluate a given language model and return the computed score(s).

        Args:
            model: A Model that allows the task to interact with the language model.
        Returns:
            `ScoreData` or `List[ScoreData]` : single or multiple scores in return 
        """
        raise NotImplementedError


