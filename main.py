from metrics.stero_skrew_winobias import SteroSkrewWino
from metrics.steroset import Steroset
from metrics.embed_wino import EmbeddingWino
from metrics.SEAT import SEAT
from metrics.ww import WW
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM
import glob
import os
import json
from tabulate import tabulate  
from collections import defaultdict
from huggingface_hub import login
import torch


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


#models = ['distilbert-base-uncased','google/bigbird-roberta-base','google/bigbird-roberta-large','YituTech/conv-bert-base','YituTech/conv-bert-medium-small','YituTech/conv-bert-small','microsoft/deberta-base','google/electra-small-discriminator','google/electra-base-discriminator','google/electra-large-discriminator','microsoft/deberta-large','microsoft/deberta-xlarge','microsoft/mpnet-base', 'google/mobilebert-uncased','squeezebert/squeezebert-uncased','bert-large-uncased', 'roberta-large','albert-base-v2', 'xlm-roberta-large','bert-base-uncased', 'roberta-base','albert-xxlarge-v2', 'xlm-roberta-base']

#models_debias = ["custom_models/bert-large-A","custom_models/albert-base-v2-A","custom_models/albert-xxlarge-v2-A","custom_models/bert-A","custom_models/distilbert-base-uncased-A","custom_models/google/bigbird-roberta-base-A","custom_models/google/bigbird-roberta-large-A","custom_models/google/electra-base-discriminator-A","custom_models/google/electra-large-discriminator-A","custom_models/google/electra-small-discriminator-A","custom_models/google/mobilebert-uncased-A","custom_models/microsoft/deberta-base-A","custom_models/microsoft/deberta-large-A","custom_models/microsoft/mpnet-base-A","custom_models/roberta-base-A","custom_models/roberta-large-A","custom_models/squeezebert-A","custom_models/xlm-roberta-base-A"]

login(token=os.environ["HUGGINGFACE_TOKEN"])


# Load Llama-3 8B model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Initialize SEAT metric
seat_metric = SEAT()

# Evaluate the model
results = seat_metric.evaluate_model(model, tokenizer)

# Process and display results
for score_data in results:
    print(f"Test: {score_data.score_name}")
    print(f"Score: {score_data.score_dict}")
    print("---")