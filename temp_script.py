import json
from collections import defaultdict

SEAT = [
        "angry_black_woman_stereotype",
        "p_value angry_black_woman_stereotype",
        "angry_black_woman_stereotype_b",
        "p_value angry_black_woman_stereotype_b",
        "heilman_double_bind_competent_1",
        "p_value heilman_double_bind_competent_1",
        "heilman_double_bind_competent_1+3-",
        "p_value heilman_double_bind_competent_1+3-",
        "heilman_double_bind_competent_1-",
        "p_value heilman_double_bind_competent_1-",
        "heilman_double_bind_competent_one_sentence",
        "p_value heilman_double_bind_competent_one_sentence",
        "heilman_double_bind_competent_one_word",
        "p_value heilman_double_bind_competent_one_word",
        "heilman_double_bind_likable_1",
        "p_value heilman_double_bind_likable_1",
        "heilman_double_bind_likable_1+3-",
        "p_value heilman_double_bind_likable_1+3-",
        "heilman_double_bind_likable_1-",
        "p_value heilman_double_bind_likable_1-",
        "heilman_double_bind_likable_one_sentence",
        "p_value heilman_double_bind_likable_one_sentence",
        "heilman_double_bind_likable_one_word",
        "p_value heilman_double_bind_likable_one_word",
        "sent-angry_black_woman_stereotype",
        "p_value sent-angry_black_woman_stereotype",
        "sent-angry_black_woman_stereotype_b",
        "p_value sent-angry_black_woman_stereotype_b",
        "sent-heilman_double_bind_competent_one_word",
        "p_value sent-heilman_double_bind_competent_one_word",
        "sent-heilman_double_bind_likable_one_word",
        "p_value sent-heilman_double_bind_likable_one_word",
        "sent-weat1",
        "p_value sent-weat1",
        "sent-weat2",
        "p_value sent-weat2",
        "sent-weat3",
        "p_value sent-weat3",
        "sent-weat3b",
        "p_value sent-weat3b",
        "sent-weat4",
        "p_value sent-weat4",
        "sent-weat5",
        "p_value sent-weat5",
        "sent-weat5b",
        "p_value sent-weat5b",
        "sent-weat6",
        "p_value sent-weat6",
        "sent-weat6b",
        "p_value sent-weat6b",
        "sent-weat7",
        "p_value sent-weat7",
        "sent-weat7b",
        "p_value sent-weat7b",
        "sent-weat8",
        "p_value sent-weat8",
        "sent-weat8b",
        "p_value sent-weat8b",
        "sent-weat9",
        "p_value sent-weat9",
        "sent-weat10",
        "p_value sent-weat10",
        "weat1",
        "p_value weat1",
        "weat2",
        "p_value weat2",
        "weat3",
        "p_value weat3",
        "weat3b",
        "p_value weat3b",
        "weat4",
        "p_value weat4",
        "weat5",
        "p_value weat5",
        "weat5b",
        "p_value weat5b",
        "weat6",
        "p_value weat6",
        "weat6b",
        "p_value weat6b",
        "weat7",
        "p_value weat7",
        "weat7b",
        "p_value weat7b",
        "weat8",
        "p_value weat8",
        "weat8b",
        "p_value weat8b",
        "weat9",
        "p_value weat9",
        "weat10",
        "p_value weat10",
]
Stero = [      
        "Count",  
        "ICAT Score",
        "LM Score",
        "SS Score",
] 
SteroSkrewWino = [
        "stero T1",
        "skew T1",
        "stero T2",
        "skew T2",
]

EmbeddingWino = [
        "dist T1",
        "dist_neutral T1",
        "neutral_score T1",
        "dist T2",
        "dist_neutral T2",
        "neutral_score T2",
]


table = json.load(open(".data/results.json"))
table_new = {}
for row in table:
    dic = {"SEAT":defaultdict(), "Stero":defaultdict(), "SteroSkrewWino":defaultdict(), "EmbeddingWino":defaultdict()}
    for c_name, c_value in row.items():
        if c_name in SEAT:
            dic["SEAT"][c_name] = c_value
        elif c_name in Stero:
            dic["Stero"][c_name] = c_value
        elif c_name in SteroSkrewWino:
            dic["SteroSkrewWino"][c_name] = c_value
        elif c_name in EmbeddingWino:
            dic["EmbeddingWino"][c_name] = c_value     
    table_new[row["Model"]] = dic 
with open('.data/results_re.json', 'w') as outfile:
    json.dump(table_new, outfile, indent=4)
