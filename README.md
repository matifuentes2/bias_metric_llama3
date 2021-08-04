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
- Weight Watcher: is an open-source, diagnostic tool for analyzing Deep Neural Networks (DNN), without needing access to training or even test data. [LINK](https://github.com/CalculatedContent/WeightWatcher)
- Embedding Stero-Skew: our-own

To run all this metric on different models use:

```
python main.py
```

## Visulization (Parallel Coordinates)
Check the Parallel Coordinates [HERE](https://andreamad8.github.io/BIAS_METRIC_RANKING/vizjs/).

## Correlation Plots
![alt text](img/ranking.png "Correlation plot")

![alt text](img/ranking_bias_ALL.png "Correlation plot with all metric")


## Results 

***Summury***
| Model                   |        SEAT |    ICAT |      ST2 |       SK2 |
|-------------------------|-------------|---------|----------|-----------|
| bert-large-A            |   0.212605  | 40.2582 | 0.30182  | 51.5653   |
| albert-base-v2-A        |   0.954114  | 42.4767 | 0.625153 | 18.3567   |
| albert-xxlarge-v2-A     |   0.964114  | 40.3126 | 0.227338 | 66.6663   |
| bert-A                  |   0.206581  | 30.6621 | 0.522885 |  2.17629  |
| distilbert-base-A       |   0.597806  | 24.8182 | 0.510859 |  0.255103 |
| bigbird-roberta-base-A  |   0.32823   | 48.3687 | 0.283724 | 19.2028   |
| bigbird-roberta-large-A |   0.52427   | 49.9503 | 0.293637 | 53.3816   |
| electra-base-A          |  -0.33938   | 59.2715 | 0.528422 |  3.20158  |
| electra-large-A         |   0.0846175 | 45.757  | 0.293637 | 53.3816   |
| electra-small-A         |   0.434103  | 43.1014 | 0.245059 | 63.344    |
| mobilebert-A            |   0.320741  | 45.2265 | 0.558182 |  8.37135  |
| deberta-base-A          |   0.537033  | 54.1031 | 0.714529 | 59.4253   |
| deberta-large-A         |   0.39596   | 56.9055 | 0.31334  | 48.9353   |
| mpnet-base-A            |   0.0790948 | 37.6795 | 0.909021 | 48.5118   |
| roberta-base-A          |   0.577402  | 39.1982 | 0.961242 | 17.5118   |
| roberta-large-A         |   0.43048   | 48.4027 | 0.399622 | 44.3825   |
| squeezebert-A           |   0.423841  | 27.5023 | 0.564603 |  9.41863  |
| xlm-roberta-base-A      |   0.299444  | 50.7881 | 0.6445   | 20.9219   |
| bigbird-roberta-base    |   0.120121  | 33.5551 | 0.256421 |  2.94378  |
| bigbird-roberta-large   |   0.246759  | 39.7349 | 2.33223  |  4.87686  |
| conv-bert-base          |   0.346541  | 61.2557 | 1.10654  | 64.4379   |
| conv-bert-medium-small  |   0.0553646 | 53.1988 | 0.504415 |  1.40565  |
| conv-bert-small         |   0.274105  | 53.8287 | 0.295723 | 52.9226   |
| deberta-base            |   0.503209  | 53.3127 | 0.424754 | 20.0563   |
| electra-small           |  -0.152407  | 50.1935 | 0.499372 |  2.42984  |
| electra-base            |   0.396143  | 46.6014 | 0.586728 | 12.8677   |
| electra-large           |   0.338071  | 64.0768 | 0.566235 |  9.68137  |
| deberta-large           |  -0.151416  | 58.652  | 0.767562 | 35.0887   |
| deberta-xlarge          |  -0.119582  | 51.1988 | 0.236414 | 64.9835   |
| mpnet-base              |   0.242279  | 35.2778 | 0.715055 | 29.4229   |
| mobilebert              |  -0.170479  | 33.9413 | 0.788274 | 37.2037   |
| squeezebert             |   0.344389  | 32.3521 | 0.829707 | 41.2643   |
| bert-large              |   0.260135  | 36.0097 | 0.833716 | 41.6462   |
| roberta-large           |   0.440252  | 34.604  | 2.29857  | 24.7641   |
| albert-base-v2          |   0.708201  | 33.9084 | 0.764229 | 34.7425   |
| xlm-roberta-large       |  -0.440905  | 23.8859 | 0.291523 | 53.844    |
| bert-base               |   0.678823  | 37.9824 | 0.79911  | 38.2867   |
| roberta-base            |   0.768783  | 33.4985 | 0.82703  | 10.0769   |
| albert-xxlarge-v2       |   0.140031  | 37.8482 | 0.665338 | 23.5604   |
| xlm-roberta-base        |  -0.313573  | 19.8099 | 0.419313 | 21.4985   |
| distilbert-base         |   0.763151  | 31.0765 | 0.889996 | 46.8286   |


## Metric class
To add a new metric, implement the following class:
```python
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
