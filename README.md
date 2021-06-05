# BIAS_METRIC_RANKING
Compare different bias related metrics 

## Installation
```
pip install -r requirements.txt
```

## Run
Currently, we implemented four metrics, which includes multiple test sets. 
The four metrics are:

- SEAT: On Measuring Social Biases in Sentence Encoders ([LINK](https://arxiv.org/pdf/1903.10561.pdf))
- SteroSet: StereoSet: Measuring stereotypical bias in pretrained language models ([LINK](https://arxiv.org/pdf/2004.09456.pdf))
- Stero-Skew: Stereotype and Skew: Quantifying Gender Bias in Pre-trained and Fine-tuned Language Models ([LINK](https://arxiv.org/pdf/2101.09688.pdf))
- Embedding Stero-Skew: our-own

To run all this metric on different models use:

```
python main.py
```

## Metric class
To add a new metric, implement the following class:
```
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

```
Check an example at: ```metrics/SEAT.py```
