import metrics.metric as metric
import weightwatcher as ww
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weightwatcher") 
logger.setLevel(logging.WARNING)

class WW(metric.Metric):
    """WeightWatcher"""

    def __init__(self, cuda = True, seed=42, verbose=False):
        pass

    def get_metric_details(self):
        raise metric.TaskMetadata(
            name="WeightWatcher",
            description="",
            keywords=["generalization", "stero", "embedding"],
            paper="https://pypi.org/project/weightwatcher/"
            )


    def evaluate_model(self, model, tokenizer):
        watcher = ww.WeightWatcher(model=model)
        details = watcher.analyze()
        summary = watcher.get_summary(details)
        
        return [metric.ScoreData(score_dict = summary,
                    preferred_score ='alpha',
                    low_score = 0,
                    high_score = 1,
                    score_name = f"WW")]