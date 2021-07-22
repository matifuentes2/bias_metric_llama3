from tabulate import tabulate  
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import json
import numpy as np
import csv



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", vmin=-4, vmax=4, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    mask = 1- np.tri(data.shape[0], k=0)
    data = np.ma.array(data, mask=mask) 
    # Want diagonal elements as well

    im = ax.imshow(data,vmin=vmin, vmax=vmax,**kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=90)#, ha="right",rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)
    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['top'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.xticks(rotation=45)
    return im, None


def annotate_heatmap(im, temp_stat, data=None, valfmt="{x:.2f}",
                     textcolors=("black","grey"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if temp_stat[i][j]:
                # kw.update(weight="bold")#color=textcolors[int(im.norm(data[i, j]) > threshold)])
                kw.update(weight="bold",color=textcolors[1])
            else:
                kw.update(color=textcolors[0])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts




table = json.load(open(".data/results_re.json"))

SEAT = [
        "angry_black_woman_stereotype",
        "angry_black_woman_stereotype_b",
        "heilman_double_bind_competent_1",
        "heilman_double_bind_competent_1+3-",
        "heilman_double_bind_competent_1-",
        "heilman_double_bind_competent_one_sentence",
        "heilman_double_bind_competent_one_word",
        "heilman_double_bind_likable_1",
        "heilman_double_bind_likable_1+3-",
        "heilman_double_bind_likable_1-",
        "heilman_double_bind_likable_one_sentence",
        "heilman_double_bind_likable_one_word",
        "sent-angry_black_woman_stereotype",
        "sent-angry_black_woman_stereotype_b",
        "sent-heilman_double_bind_competent_one_word",
        "sent-heilman_double_bind_likable_one_word",
        "sent-weat6",
        "sent-weat6b",
        "sent-weat7",
        "sent-weat7b",
        "sent-weat8",
        "sent-weat8b",

        # "sent-weat1",
        # "sent-weat2",
        # "sent-weat3",
        # "sent-weat3b",
        # "sent-weat4",
        # "sent-weat5",
        # "sent-weat5b",
        # "sent-weat9",
        # "sent-weat10",
        # "weat1",
        # "weat2",
        # "weat3",
        # "weat3b",
        # "weat4",
        # "weat5",
        # "weat5b",
        # "weat6",
        # "weat6b",
        # "weat7",
        # "weat7b",
        # "weat8",
        # "weat8b",
        # "weat9",
        # "weat10",
]
Stero = [      
        "ICAT Score",
        # "Count",  
        # "LM Score",
        # "SS Score",
] 
SteroSkrewWino = [
        "stero T1",
        "stero T2",
        # "skew T1",
        # "skew T2",
]

EmbeddingWino = [
        # "dist T1",
        # "dist_neutral T1",
        "neutral_score T1",
        # "dist T2",
        # "dist_neutral T2",
        "neutral_score T2",
]
WW = [
    # "log_norm",
    "alpha",
    # "alpha_weighted",
    # "log_alpha_norm",
    # "log_spectral_norm",
    # "stable_rank",
]

map_names = {
    "alpha":r"$\alpha$",
    "neutral_score T1":"NT1",
    "neutral_score T2":"NT2",
    "stero T1":"ST1",
    "stero T2":"ST2",
    "skew T2":"SK2",
    "skew T1":"SK1",
    "ICAT Score":"ICAT",
    "LM Score":"LM-ICAT",
    "SS Score":"SS-ICAT",
}
to_avg = [
        "angry_black_woman_stereotype",
        "angry_black_woman_stereotype_b",
        "heilman_double_bind_competent_1",
        "heilman_double_bind_competent_1+3-",
        "heilman_double_bind_competent_1-",
        "heilman_double_bind_competent_one_sentence",
        "heilman_double_bind_competent_one_word",
        "heilman_double_bind_likable_1",
        "heilman_double_bind_likable_1+3-",
        "heilman_double_bind_likable_1-",
        "heilman_double_bind_likable_one_sentence",
        "heilman_double_bind_likable_one_word",
        "sent-angry_black_woman_stereotype",
        "sent-angry_black_woman_stereotype_b",
        "sent-heilman_double_bind_competent_one_word",
        "sent-heilman_double_bind_likable_one_word",
        "sent-weat6",
        "sent-weat6b",
        "sent-weat7",
        "sent-weat7b",
        "sent-weat8",
        "sent-weat8b",
]
table_new = []
arr_numbers = []
metric_to_eval = ["SEAT", "Stero", "SteroSkrewWino", "EmbeddingWino","WW"]
for model_name, t in table.items():
    dic = {}
    dic["Model"]= model_name.replace("custom_models/","").replace("google/","").replace("microsoft/","").replace("YituTech/","").replace("-discriminator","").replace("-uncased","").replace("squeezebert/","")
    print(model_name)
    for metric in metric_to_eval:
        if metric in t:
            if metric == "SEAT" and len(to_avg)>0:
                seat = []
                for c in to_avg:
                    seat.append(t[metric][c])
                dic["SEAT"] = np.mean(seat)
            elif metric == "SEAT" and len(to_avg)==0:
                for c in SEAT:
                    dic[c] = t[metric][c]
            elif metric == "Stero":
                for c in Stero:
                    if c == "ICAT Score" or c == "LM Score":
                        dic[map_names[c]] = 100-t[metric][c]
                    else:
                        dic[map_names[c]] = t[metric][c]
            elif metric == "SteroSkrewWino":
                for c in SteroSkrewWino:
                    dic[map_names[c]] = t[metric][c]
            elif metric == "EmbeddingWino":
                for c in EmbeddingWino:
                    dic[map_names[c]] = t[metric][c]
            elif metric == "WW":
                for c in WW:
                    dic[map_names[c]] = t[metric][c]
        else:
            if metric == "SEAT":
                dic["SEAT"] = 0.0
            elif metric == "Stero":
                for c in Stero:
                    dic[c] = 0.0
            elif metric == "SteroSkrewWino":
                for c in SteroSkrewWino:
                    dic[c] = 0.0
            elif metric == "EmbeddingWino":
                for c in EmbeddingWino:
                    dic[c] = 0.0
            elif metric == "WW":
                for c in WW:
                    dic[c] = 0.0

    table_new.append(dic)
    # del dic['Model']
    arr_numbers.append(list(v for k, v in dic.items() if k!= "Model" ))

print(tabulate(table_new,tablefmt="github",headers="keys"))
with open('vizjs/results.csv', mode='w') as fi:
    writer = csv.writer(fi, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(list(table_new[0].keys())+["group"])
    for row in table_new:
        if "-A" in row['Model']:
            writer.writerow(list(row.values())+[1])
        else:
            writer.writerow(list(row.values())+[0])

# plt.rcParams.update({'font.size': 8})
names = list(table_new[0].keys())
names.remove("Model")

arr = np.array(arr_numbers).transpose()
arr = np.nan_to_num(arr, nan=0.0)

n = len(names)
corr_mat = []
stat_mat = []
for i in range(n):
    temp = []
    temp_stat = []
    for j in range(n):
        corr, p_val  = pearsonr(arr[i], arr[j])
        temp.append(corr)
        temp_stat.append(True if p_val<0.05 else False)

    corr_mat.append(temp)
    stat_mat.append(temp_stat)
corr_mat = np.array(corr_mat)



fig, ax = plt.subplots()#figsize=(15,18)
# fig, ax = plt.subplots(figsize=(15,18))
names = [n.replace("_","\_")for n in names]
im, cbar = heatmap(corr_mat, names, names, ax=ax, vmin=-1, vmax=1, cmap="Spectral", cbarlabel="Pearson Correlation")
texts = annotate_heatmap(im,stat_mat, valfmt="{x:.1f}")

fig.tight_layout()

plt.savefig("img/ranking.png",dpi=400)


