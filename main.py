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

login(token=os.environ["HUGGINGFACE_TOKEN"])


# Load Llama-3 8B model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    use_auth_token=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_folder="offload"
)

# Initialize SEAT metric
seat_metric = SEAT()

# Evaluate the model
results = seat_metric.evaluate_model(model, tokenizer)

# Process and display results
for score_data in results:
    print(f"Test: {score_data.score_name}")
    print(f"Score: {score_data.score_dict}")
    print("---")