# from metrics.stero_skrew_winobias import SteroSkrewWino
# from metrics.steroset import Steroset
# from metrics.embed_wino import EmbeddingWino
from metrics.SEAT_spanish import SEAT
from metrics.ww import WW
from transformers import AutoTokenizer, AutoModelForCausalLM#, AutoModel, AutoModelForMaskedLM
# import glob
import os
import json
# from tabulate import tabulate  
from collections import defaultdict
from huggingface_hub import login
import torch
from prettytable import PrettyTable



def score_metric(model, tokenizer, metr):
    res = metr.evaluate_model(model,tokenizer)
    dic = {}
    for r in res:
        dic.update(r.score_dict)
    return dic

if os.path.exists(".data/results_re_spanish.json"):
    table = json.load(open(".data/results_re_spanish.json"))
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

# Create a PrettyTable object
table = PrettyTable()
table.field_names = ["Test", "Targ1", "Targ2", "Atr1", "Atr2" , "p-value", "Effect Size"]
    
# Process and display results. Then add to table
for score_data in results:
    print(f"Test: {score_data.score_name}")
    print(f"Score: {score_data.score_dict}")
    print("---")
    test_name = score_data.score_name.split(" ")[-1]
    categories = test_name.replace("_", " ").title()
    p_value = score_data.score_dict[f'p_value {test_name}']
    effect_size = score_data.score_dict[test_name]
    targ1 = score_data.targ1
    targ2 = score_data.targ2
    attr1 = score_data.attr1
    attr2 = score_data.attr2
    
    table.add_row([test_name, targ1, targ2, attr1, attr2, f"{p_value:.6f}", f"{effect_size:.6f}"])

# Set table styles
table.align = "l"
table.max_width = 120

# Print the table
print(table)

# Optionally, save the table to a file
with open("seat_results_summary_spanish.txt", "w") as f:
    f.write(str(table))