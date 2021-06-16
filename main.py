from metrics.stero_skrew_winobias import SteroSkrewWino
from metrics.steroset import Steroset
from metrics.embed_wino import EmbeddingWino
from metrics.SEAT import SEAT
from metrics.ww import WW
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import glob
import os
import json
from tabulate import tabulate  
from collections import defaultdict


def score_metric(model, tokenizer, metr):
    res = metr.evaluate_model(model,tokenizer)
    dic = {}
    for r in res:
        dic.update(r.score_dict)
    return dic

if os.path.exists(".data/results_re.json"):
    table = json.load(open(".data/results_re.json"))
else:
    table = defaultdict(lambda: defaultdict())


models = ['distilbert-base-uncased','google/bigbird-roberta-base','google/bigbird-roberta-large','YituTech/conv-bert-base','YituTech/conv-bert-medium-small','YituTech/conv-bert-small','microsoft/deberta-base','google/electra-small-discriminator','google/electra-base-discriminator','google/electra-large-discriminator','microsoft/deberta-large','microsoft/deberta-xlarge','microsoft/mpnet-base', 'google/mobilebert-uncased','squeezebert/squeezebert-uncased','bert-large-uncased', 'roberta-large','albert-base-v2', 'xlm-roberta-large','bert-base-uncased', 'roberta-base','albert-xxlarge-v2', 'xlm-roberta-base']

models_debias = ["custom_models/bert-large-A","custom_models/albert-base-v2-A","custom_models/albert-xxlarge-v2-A","custom_models/bert-A","custom_models/distilbert-base-uncased-A","custom_models/google/bigbird-roberta-base-A","custom_models/google/bigbird-roberta-large-A","custom_models/google/electra-base-discriminator-A","custom_models/google/electra-large-discriminator-A","custom_models/google/electra-small-discriminator-A","custom_models/google/mobilebert-uncased-A","custom_models/microsoft/deberta-base-A","custom_models/microsoft/deberta-large-A","custom_models/microsoft/mpnet-base-A","custom_models/roberta-base-A","custom_models/roberta-large-A","custom_models/squeezebert-A","custom_models/xlm-roberta-base-A"]

metric_to_eval = ["SEAT", "Stero", "SteroSkrewWino", "EmbeddingWino","WW"]

for model_name in models+models_debias:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model_MLM = AutoModelForMaskedLM.from_pretrained(model_name)
    for metric in metric_to_eval:
        if model_name in table and  metric in table[model_name]:
            print(f"Skipping {model_name} for {metric} since it has been already eval")
            continue

        print(f"Evaluating {model_name}")


        if metric == "WW":
            dic_results = score_metric(model, tokenizer, WW())
        elif metric == "SEAT":
            dic_results = score_metric(model_MLM, tokenizer, SEAT())
        elif metric == "Stero":
            dic_results = score_metric(model_MLM, tokenizer, Steroset(batch_size=1))
        elif metric == "SteroSkrewWino":
            dic_results = score_metric(model_MLM, tokenizer, SteroSkrewWino())
        elif metric == "EmbeddingWino":
            dic_results = score_metric(model_MLM, tokenizer, EmbeddingWino())


        table[model_name][metric] = dic_results

        with open('.data/results_re.json', 'w') as outfile:
            json.dump(table, outfile, indent=4)


        ## delete the evaluated model to avoid too large memory useage
        ## UNCOMMENT THIS IF YOU DON'T WANT CACHE THE CHECKPOINTS
        print("DELETING FILES")
        files = glob.glob('/home/andrea/.cache/huggingface/transformers/*', recursive=True)
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))