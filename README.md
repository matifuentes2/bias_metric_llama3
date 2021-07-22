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
| Model                   |        SEAT |    ICAT |      ST1 |      ST2 |       NT1 |      NT2 |   $\alpha$ |
|-------------------------|-------------|---------|----------|----------|-----------|----------|------------|
| bert-large-A            |   0.212605  | 40.2582 | 0        | 0.30182  | 0.578283  | 0.494949 |    4.31179 |
| albert-base-v2-A        |   0.954114  | 42.4767 | 0        | 0.625153 | 0.247475  | 0.161616 |    5.35072 |
| albert-xxlarge-v2-A     |   0.964114  | 40.3126 | 0        | 0.227338 | 0.522727  | 0.489899 |    9.00281 |
| bert-A                  |   0.206581  | 30.6621 | 0        | 0.522885 | 0.171717  | 0.204545 |    4.05595 |
| distilbert-base-A       |   0.597806  | 24.8182 | 0        | 0.510859 | 0.767677  | 0.641414 |    4.23127 |
| bigbird-roberta-base-A  |   0.32823   | 48.3687 | 0.309168 | 0.283724 | 0.472222  | 0.507576 |    4.67274 |
| bigbird-roberta-large-A |   0.52427   | 49.9503 | 0        | 0.293637 | 0.666667  | 0.669192 |    4.60873 |
| electra-base-A          |  -0.33938   | 59.2715 | 0        | 0.528422 | 0.184343  | 0.401515 |    4.61282 |
| electra-large-A         |   0.0846175 | 45.757  | 0        | 0.293637 | 0.525253  | 0.502525 |    4.57602 |
| electra-small-A         |   0.434103  | 43.1014 | 0        | 0.245059 | 0.426768  | 0.565657 |    4.89688 |
| mobilebert-A            |   0.320741  | 45.2265 | 0        | 0.558182 | 0.388889  | 0.340909 |    6.6069  |
| deberta-base-A          |   0.537033  | 54.1031 | 0.30847  | 0.714529 | 0.782828  | 0.823232 |    3.29192 |
| deberta-large-A         |   0.39596   | 56.9055 | 0        | 0.31334  | 0.636364  | 0.676768 |    3.54856 |
| mpnet-base-A            |   0.0790948 | 37.6795 | 0        | 0.909021 | 0.558081  | 0.583333 |    5.03652 |
| roberta-base-A          |   0.577402  | 39.1982 | 0        | 0.961242 | 0.883838  | 0.780303 |    4.37836 |
| roberta-large-A         |   0.43048   | 48.4027 | 0        | 0.399622 | 0.540404  | 0.666667 |    4.27942 |
| squeezebert-A           |   0.423841  | 27.5023 | 0        | 0.564603 | 0.0631313 | 0.121212 |    1.6226  |
| xlm-roberta-base-A      |   0.299444  | 50.7881 | 0        | 0.6445   | 0.474747  | 0.277778 |    4.4435  |
| bigbird-roberta-base    |   0.120121  | 33.5551 | 1.87972  | 0.256421 | 0.333333  | 0.361111 |    4.66235 |
| bigbird-roberta-large   |   0.246759  | 39.7349 | 0.850865 | 2.33223  | 0.585859  | 0.659091 |    4.61276 |
| conv-bert-base          |   0.346541  | 61.2557 | 0        | 1.10654  | 0.770202  | 0.830808 |    4.87079 |
| conv-bert-medium-small  |   0.0553646 | 53.1988 | 0        | 0.504415 | 0.555556  | 0.790404 |    5.48689 |
| conv-bert-small         |   0.274105  | 53.8287 | 0        | 0.295723 | 0.338384  | 0.343434 |    5.14023 |
| deberta-base            |   0.503209  | 53.3127 | 0.294503 | 0.424754 | 0.636364  | 0.775253 |    3.29402 |
| electra-small           |  -0.152407  | 50.1935 | 0        | 0.499372 | 0.542929  | 0.717172 |    4.85704 |
| electra-base            |   0.396143  | 46.6014 | 0        | 0.586728 | 0.719697  | 0.752525 |    4.52969 |
| electra-large           |   0.338071  | 64.0768 | 0        | 0.566235 | 0.739899  | 0.646465 |    4.6576  |
| deberta-large           |  -0.151416  | 58.652  | 0.634049 | 0.767562 | 0.671717  | 0.739899 |    3.54834 |
| deberta-xlarge          |  -0.119582  | 51.1988 | 1.01572  | 0.236414 | 0.661616  | 0.737374 |    3.48598 |
| mpnet-base              |   0.242279  | 35.2778 | 0        | 0.715055 | 0.590909  | 0.45202  |    5.05202 |
| mobilebert              |  -0.170479  | 33.9413 | 0        | 0.788274 | 0.512626  | 0.522727 |    6.57335 |
| squeezebert             |   0.344389  | 32.3521 | 0        | 0.829707 | 0.340909  | 0.277778 |    1.6226  |
| bert-large              |   0.260135  | 36.0097 | 0        | 0.833716 | 0.487374  | 0.457071 |    4.20567 |
| roberta-large           |   0.440252  | 34.604  | 2.78896  | 2.29857  | 0.747475  | 0.694444 |    4.29148 |
| albert-base-v2          |   0.708201  | 33.9084 | 0        | 0.764229 | 0.765152  | 0.767677 |    4.11182 |
| xlm-roberta-large       |  -0.440905  | 23.8859 | 0        | 0.291523 | 0.520202  | 0.492424 |    4.49264 |
| bert-base               |   0.678823  | 37.9824 | 0        | 0.79911  | 0.727273  | 0.507576 |    4.0507  |
| roberta-base            |   0.768783  | 33.4985 | 2.26141  | 0.82703  | 0.575758  | 0.5      |    4.85783 |
| albert-xxlarge-v2       |   0.140031  | 37.8482 | 0        | 0.665338 | 0.679293  | 0.65404  |    4.37851 |
| xlm-roberta-base        |  -0.313573  | 19.8099 | 0        | 0.419313 | 0.719697  | 0.573232 |    4.37837 |
| distilbert-base         |   0.763151  | 31.0765 | 0        | 0.889996 | 0.792929  | 0.787879 |    4.25338 |


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
