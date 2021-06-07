from metrics.stero_skrew_winobias import SteroSkrewWino
from metrics.steroset import Steroset
from metrics.embed_wino import EmbeddingWino
from metrics.SEAT import SEAT
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import glob
import os
import json
from tabulate import tabulate  



def score_metric(model, tokenizer, metr, dic):
    res = metr.evaluate_model(model,tokenizer)
    for r in res:
        dic.update(r.score_dict)
    return dic

if os.path.exists(".data/results.json"):
    table = json.load(open(".data/results.json"))
    names_done = [t["Model"] for t in table]
else:
    table = []
    names_done = []


models = ['distilbert-base-uncased','google/bigbird-roberta-base','google/bigbird-roberta-large','YituTech/conv-bert-base','YituTech/conv-bert-medium-small','YituTech/conv-bert-small','microsoft/deberta-base','google/electra-small-discriminator','google/electra-base-discriminator','google/electra-large-discriminator','microsoft/deberta-large','microsoft/deberta-xlarge','microsoft/mpnet-base', 'google/mobilebert-uncased','squeezebert/squeezebert-uncased','bert-large-uncased', 'roberta-large','albert-base-v2', 'xlm-roberta-large','bert-base-uncased', 'roberta-base','albert-xxlarge-v2', 'xlm-roberta-base']

models_debias = ["custom_models/bert-large-A","custom_models/albert-base-v2-A","custom_models/albert-xxlarge-v2-A","custom_models/bert-A","custom_models/distilbert-base-uncased-A","custom_models/google/bigbird-roberta-base-A","custom_models/google/bigbird-roberta-large-A","custom_models/google/electra-base-discriminator-A","custom_models/google/electra-large-discriminator-A","custom_models/google/electra-small-discriminator-A","custom_models/google/mobilebert-uncased-A","custom_models/microsoft/deberta-base-A","custom_models/microsoft/deberta-large-A","custom_models/microsoft/mpnet-base-A","custom_models/roberta-base-A","custom_models/roberta-large-A","custom_models/squeezebert-A","custom_models/xlm-roberta-base-A"]

for model_name in models_debias+models:
    if model_name in names_done:
        print(f"Skipping {model_name} since it has been already eval")
        continue

    print(f"Evaluating {model_name}")
    dic_results = {"Model":model_name}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    dic_results = score_metric(model, tokenizer, SEAT(), dic_results)
    dic_results = score_metric(model, tokenizer, Steroset(batch_size=1), dic_results)
    dic_results = score_metric(model, tokenizer, SteroSkrewWino(), dic_results)
    dic_results = score_metric(model, tokenizer, EmbeddingWino(), dic_results)


    table.append(dic_results)
    print(tabulate(table,tablefmt="github",headers="keys"))

    with open('.data/results.json', 'w') as outfile:
        json.dump(table, outfile, indent=4)


    ## delete the evaluated model to avoid too large memory useage
    ## UNCOMMENT THIS IF YOU DON'T WANT CACHE THE CHECKPOINTS
    # print("DELETING FILES")
    # files = glob.glob('/home/andrea/.cache/huggingface/transformers/*', recursive=True)
    # for f in files:
    #     try:
    #         os.remove(f)
    #     except OSError as e:
    #         print("Error: %s : %s" % (f, e.strerror))