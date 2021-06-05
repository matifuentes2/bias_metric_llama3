import metrics.metric as metric
import os 
import string
import torch
import numpy as np
from tqdm import tqdm
import re


class EmbeddingWino(metric.Metric):
    """Stereotype and Skew: Quantifying Gender Bias in Pre-trained and Fine-tuned Language Models"""

    def __init__(self, cuda = True, seed=42, verbose=False):
        self.download_data()
        self.protest1 = "corefBias/WinoBias/wino/data/pro_stereotyped_type1.txt.test"
        self.antitest1 = "corefBias/WinoBias/wino/data/anti_stereotyped_type1.txt.test"
        self.protest2 = "corefBias/WinoBias/wino/data/pro_stereotyped_type2.txt.test"
        self.antitest2 = "corefBias/WinoBias/wino/data/anti_stereotyped_type2.txt.test"

        self.test_1 = self.load_data(self.protest1,self.antitest1)
        self.test_2 = self.load_data(self.protest2,self.antitest2)

        self.cuda = cuda
        self.device = "cuda" if self.cuda else "cpu"
        self.pdist = torch.nn.PairwiseDistance(p=2)
        
    def get_embedding_avg(self, sentence, model, tokenizer, device):
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs,output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        # print(last_hidden_states)
        return torch.mean(last_hidden_states,dim=1)

    def pre_proc(self,li, make_neutral=False):
        new_li = []
        for _,line in enumerate(li):
            line = " ".join(line.split()[1:])
            sex = re.findall('\[.*?\]',line)
            if make_neutral:
                line = line.replace(sex[0],"[the person]")
            if("[he]" in sex or "[his]" in sex or "[He]" in sex or "[His]" in sex):
                new_li.append([line.replace("[","").replace("]","").strip(),"M"])
            else:
                new_li.append([line.replace("[","").replace("]","").strip(),"F"])
        return new_li

    def load_data(self,pro_filename,anti_filename):
        f = open(pro_filename, "r") 
        lines = f.readlines()
        f.close()
        f = open(anti_filename, "r") 
        lines_anti = f.readlines()
        f.close()

        pro = self.pre_proc(lines, make_neutral=False)
        pro_neu = self.pre_proc(lines, make_neutral=True)
        
        anti = self.pre_proc(lines_anti, make_neutral=False)
        anti_neu = self.pre_proc(lines_anti, make_neutral=True)
        
        return {"pro":pro,"pro_neu":pro_neu,"anti":anti, "anti_neu":anti_neu}

    def download_data(self):
        if not os.path.isdir('corefBias'):
            os.system("git clone https://github.com/uclanlp/corefBias.git")
        
    def get_metric_details(self):
        raise metric.TaskMetadata(
            name="Embedding distance for WinoBias",
            description="",
            keywords=["gender bias", "stero", "embedding", "occupation"],
            paper="NONE"
            )

    def evaluate_model(self, model, tokenizer):
        model.to(self.device)
        model.eval()
        score = []
        for i, test_set in enumerate([self.test_1,self.test_2]):
            dist_original = []
            arr_original = []
            for [sent_pro, _],[sent_anti, _] in tqdm(zip(test_set['pro'], test_set['anti']),total=len(test_set['pro'])):
                emb_pro = self.get_embedding_avg(sent_pro,model,tokenizer,self.device)
                emb_anti = self.get_embedding_avg(sent_anti,model,tokenizer,self.device)
                distance = self.pdist(emb_pro,emb_anti).item()
                dist_original.append(distance)
                arr_original.append([sent_pro,distance])

            dist_neutral = []
            arr_neutral = []
            for [sent_pro, _],[sent_anti, _] in tqdm(zip(test_set['pro_neu'], test_set['anti_neu']),total=len(test_set['pro_neu'])):
                emb_pro = self.get_embedding_avg(sent_pro,model,tokenizer,self.device)
                emb_anti = self.get_embedding_avg(sent_anti,model,tokenizer,self.device)
                distance = self.pdist(emb_pro,emb_anti).item()
                dist_neutral.append(distance)
                arr_neutral.append([sent_pro,distance])


            cnt = []
            for [sent_neu, score_neu], [sent_orig, score_orig] in zip(arr_neutral, arr_original):
                if score_orig> score_neu:
                    cnt.append(1)
                else:
                    cnt.append(0)
            score.append(metric.ScoreData(score_dict = {f'dist T{i+1}':np.mean(dist_original),f'dist_neutral T{i+1}':np.mean(dist_neutral), f"neutral_score T{i+1}":np.mean(cnt)},
                                    preferred_score ='neutral_score',
                                    low_score = 0,
                                    high_score = 1,
                                    score_name = f"EmbeddingWino T{i+1}"))
        return score
        