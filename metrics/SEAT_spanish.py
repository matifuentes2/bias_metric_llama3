import metrics.metric as metric
import logging as log
import math
import itertools as it
import numpy as np
import scipy.special
import scipy.stats
import os 
import string
import json
import torch
import numpy as np
from tqdm import tqdm
import re


class SEAT(metric.Metric):
    """On Measuring Social Biases in Sentence Encoders"""

    def __init__(self, cuda = True, seed=42, verbose=False):
        self.download_data()
        self.data_dir = ".data/sent-bias-spanish/tests"
        self.all_tests = sorted(
            [
                entry[:-len('.jsonl')]
                for entry in os.listdir(self.data_dir)
                if not entry.startswith('.') and entry.endswith('.jsonl')
            ],
            key=test_sort_key
        )

        self.cuda = cuda
        self.device = "cuda" if self.cuda else "cpu"

    def get_metric_details(self):
        raise metric.TaskMetadata(
            name="On Measuring Social Biases in Sentence Encoders",
            description="",
            keywords=["social bias", "stero", "embedding"],
            paper="https://arxiv.org/pdf/1903.10561.pdf"
            )

    def download_data(self):
        if not os.path.isdir('.data/sent-bias-spanish'):
            os.system("git clone https://github.com/matifuentes2/sent-bias-spanish.git .data/sent-bias-spanish")
         
    # ORIGINAL ENCODE 
    # def encode(model, tokenizer, texts):
    #     ''' Use tokenizer and model to encode texts '''
    #     encs = {}
    #     for text in texts:
    #         tokenized = tokenizer.tokenize(text)
    #         indexed = tokenizer.convert_tokens_to_ids(tokenized)
    #         segment_idxs = [0] * len(tokenized)
    #         tokens_tensor = torch.tensor([indexed])
    #         segments_tensor = torch.tensor([segment_idxs])
    #         enc, _ = model(tokens_tensor, segments_tensor, output_all_encoded_layers=False)

    #         enc = enc[:, 0, :]  # extract the last rep of the first input
    #         encs[text] = enc.detach().view(-1).numpy()
    #     return encs

    def get_embedding_avg(self, sentences, model, tokenizer):
      model.eval()
      encs = {}
      for sentence in sentences:
          inputs = tokenizer(sentence, return_tensors="pt")
          # Move inputs to the same device as the model
          inputs = {k: v.to(model.device) for k, v in inputs.items()}
          with torch.no_grad():
              outputs = model(**inputs, output_hidden_states=True)
          last_hidden_states = outputs.hidden_states[-1]
          encs[sentence] = torch.mean(last_hidden_states, dim=1).view(-1).cpu().numpy()
      return encs



    def evaluate_model(self, model, tokenizer):
      model.eval()
      score = []
      for test in self.all_tests:
      #for test in self.all_tests[:3]:
          encs = load_json(os.path.join(self.data_dir, f"{test}.jsonl"))
          encs["targ1"]["encs"] = self.get_embedding_avg(encs["targ1"]["examples"], model, tokenizer)
          encs["targ2"]["encs"] = self.get_embedding_avg(encs["targ2"]["examples"], model, tokenizer)
          encs["attr1"]["encs"] = self.get_embedding_avg(encs["attr1"]["examples"], model, tokenizer)
          encs["attr2"]["encs"] = self.get_embedding_avg(encs["attr2"]["examples"], model, tokenizer)
          enc = [e for e in encs["targ1"]['encs'].values()][0]
          d_rep = enc.size if isinstance(enc, np.ndarray) else len(enc)
          esize, pval = run_test(encs, n_samples=100000, parametric=False)
          score.append(metric.ScoreData(score_dict = {f'{test}':esize,f'p_value {test}':pval},
                      preferred_score =f'{test}',
                      low_score = 0,
                      high_score = 1,
                      score_name = f"SEAT {test}",
                      targ1 = encs["targ1"]["category"],
                      targ2 = encs["targ2"]["category"],
                      attr1 = encs["attr1"]["category"],
                      attr2 = encs["attr2"]["category"]))
      return score

''' Implements the WEAT tests '''
# X and Y are two sets of target words of equal size.
# A and B are two sets of attribute words.

def split_comma_and_check(arg_str, allowed_set, item_type):
    ''' Given a comma-separated string of items,
    split on commas and check if all items are in allowed_set.
    item_type is just for the assert message. '''
    items = arg_str.split(',')
    for item in items:
        if item not in allowed_set:
            raise ValueError("Unknown %s: %s!" % (item_type, item))
    return items

def load_json(sent_file):
    ''' Load from json. We expect a certain format later, so do some post processing '''
    log.info("Loading %s..." % sent_file)
    all_data = json.load(open(sent_file, 'r'))
    data = {}
    for k, v in all_data.items():
        examples = v["examples"]
        data[k] = examples
        v["examples"] = examples
    return all_data  # data

def cossim(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    epsilon = 1e-8  # Small value to prevent division by zero
    return np.dot(x, y) / (norm_x * norm_y + epsilon)


def construct_cossim_lookup(XY, AB):
    """
    XY: mapping from target string to target vector (either in X or Y)
    AB: mapping from attribute string to attribute vectore (either in A or B)
    Returns an array of size (len(XY), len(AB)) containing cosine similarities
    between items in XY and items in AB.
    """

    cossims = np.zeros((len(XY), len(AB)))
    for xy in XY:
        for ab in AB:
            cossims[xy, ab] = cossim(XY[xy], AB[ab])
    return cossims

def test_sort_key(test):
    '''
    Return tuple to be used as a sort key for the specified test name.
   Break test name into pieces consisting of the integers in the name
    and the strings in between them.
    '''
    key = ()
    prev_end = 0
    for match in re.finditer(r'\d+', test):
        key = key + (test[prev_end:match.start()], int(match.group(0)))
        prev_end = match.end()
    key = key + (test[prev_end:],)

    return key

def s_wAB(A, B, cossims):
    """
    Return vector of s(w, A, B) across w, where
        s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    """
    return cossims[:, A].mean(axis=1) - cossims[:, B].mean(axis=1)


def s_XAB(X, s_wAB_memo):
    r"""
    Given indices of target concept X and precomputed s_wAB values,
    return slightly more computationally efficient version of WEAT
    statistic for p-value computation.

    Caliskan defines the WEAT statistic s(X, Y, A, B) as
        sum_{x in X} s(x, A, B) - sum_{y in Y} s(y, A, B)
    where s(w, A, B) is defined as
        mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    The p-value is computed using a permutation test on (X, Y) over all
    partitions (X', Y') of X union Y with |X'| = |Y'|.

    However, for all partitions (X', Y') of X union Y,
        s(X', Y', A, B)
      = sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
      = C,
    a constant.  Thus
        sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
      = sum_{x in X'} s(x, A, B) + (C - sum_{x in X'} s(x, A, B))
      = C + 2 sum_{x in X'} s(x, A, B).

    By monotonicity,
        s(X', Y', A, B) > s(X, Y, A, B)
    if and only if
        [s(X', Y', A, B) - C] / 2 > [s(X, Y, A, B) - C] / 2,
    that is,
        sum_{x in X'} s(x, A, B) > sum_{x in X} s(x, A, B).
    Thus we only need use the first component of s(X, Y, A, B) as our
    test statistic.
    """
    return s_wAB_memo[X].sum()


def s_XYAB(X, Y, s_wAB_memo):
    r"""
    Given indices of target concept X and precomputed s_wAB values,
    the WEAT test statistic for p-value computation.
    """
    return s_XAB(X, s_wAB_memo) - s_XAB(Y, s_wAB_memo)


def p_val_permutation_test(X, Y, A, B, n_samples, cossims, parametric=False):
    ''' Compute the p-val for the permutation test, which is defined as
        the probability that a random even partition X_i, Y_i of X u Y
        satisfies P[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
    '''
    X = np.array(list(X), dtype=np.int64)  # Change np.int to np.int64
    Y = np.array(list(Y), dtype=np.int64)  # Change np.int to np.int64
    A = np.array(list(A), dtype=np.int64)  # Change np.int to np.int64
    B = np.array(list(B), dtype=np.int64)  # Change np.int to np.int64

    assert len(X) == len(Y)
    size = len(X)
    s_wAB_memo = s_wAB(A, B, cossims=cossims)
    XY = np.concatenate((X, Y))

    if parametric:
        log.info('Using parametric test')
        s = s_XYAB(X, Y, s_wAB_memo)

        log.info('Drawing {} samples'.format(n_samples))
        samples = []
        for _ in range(n_samples):
            np.random.shuffle(XY)
            Xi = XY[:size]
            Yi = XY[size:]
            assert len(Xi) == len(Yi)
            si = s_XYAB(Xi, Yi, s_wAB_memo)
            samples.append(si)

        # Compute sample standard deviation and compute p-value by
        # assuming normality of null distribution
        log.info('Inferring p-value based on normal distribution')
        (shapiro_test_stat, shapiro_p_val) = scipy.stats.shapiro(samples)
        log.info('Shapiro-Wilk normality test statistic: {:.2g}, p-value: {:.2g}'.format(
            shapiro_test_stat, shapiro_p_val))
        sample_mean = np.mean(samples)
        sample_std = np.std(samples, ddof=1)
        log.info('Sample mean: {:.2g}, sample standard deviation: {:.2g}'.format(
            sample_mean, sample_std))
        p_val = scipy.stats.norm.sf(s, loc=sample_mean, scale=sample_std)
        return p_val

    else:
        log.info('Using non-parametric test')
        s = s_XAB(X, s_wAB_memo)
        total_true = 0
        total_equal = 0
        total = 0

        num_partitions = int(scipy.special.binom(2 * len(X), len(X)))
        if num_partitions > n_samples:
            # We only have as much precision as the number of samples drawn;
            # bias the p-value (hallucinate a positive observation) to
            # reflect that.
            total_true += 1
            total += 1
            log.info('Drawing {} samples (and biasing by 1)'.format(n_samples - total))
            for _ in range(n_samples - 1):
                np.random.shuffle(XY)
                Xi = XY[:size]
                assert 2 * len(Xi) == len(XY)
                si = s_XAB(Xi, s_wAB_memo)
                if si > s:
                    total_true += 1
                elif si == s:  # use conservative test
                    total_true += 1
                    total_equal += 1
                total += 1

        else:
            log.info('Using exact test ({} partitions)'.format(num_partitions))
            for Xi in it.combinations(XY, len(X)):
                Xi = np.array(Xi, dtype=np.int64)
                assert 2 * len(Xi) == len(XY)
                si = s_XAB(Xi, s_wAB_memo)
                if si > s:
                    total_true += 1
                elif si == s:  # use conservative test
                    total_true += 1
                    total_equal += 1
                total += 1

        if total_equal:
            pass
            # log.warning('Equalities contributed {}/{} to p-value'.format(total_equal, total))

        return total_true / total


def mean_s_wAB(X, A, B, cossims):
    return np.mean(s_wAB(A, B, cossims[X]))


def stdev_s_wAB(X, A, B, cossims):
    return np.std(s_wAB(A, B, cossims[X]), ddof=1)


def effect_size(X, Y, A, B, cossims):
    """
    Compute the effect size, which is defined as
        [mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)] /
            [ stddev_{w in X u Y} s(w, A, B) ]
    args:
        - X, Y, A, B : sets of target (X, Y) and attribute (A, B) indices
    """
    X = list(X)
    Y = list(Y)
    A = list(A)
    B = list(B)

    numerator = mean_s_wAB(X, A, B, cossims=cossims) - mean_s_wAB(Y, A, B, cossims=cossims)
    denominator = stdev_s_wAB(X + Y, A, B, cossims=cossims)
    return numerator / denominator


def convert_keys_to_ints(X, Y):
    return (
        dict((i, v) for (i, (k, v)) in enumerate(X.items())),
        dict((i + len(X), v) for (i, (k, v)) in enumerate(Y.items())),
    )


def run_test(encs, n_samples, parametric=False):
    ''' Run a WEAT.
    args:
        - encs (Dict[str: Dict]): dictionary mapping targ1, targ2, attr1, attr2
            to dictionaries containing the category and the encodings
        - n_samples (int): number of samples to draw to estimate p-value
            (use exact test if number of permutations is less than or
            equal to n_samples)
    '''
    X, Y = encs["targ1"]["encs"], encs["targ2"]["encs"]
    A, B = encs["attr1"]["encs"], encs["attr2"]["encs"]

    # First convert all keys to ints to facilitate array lookups
    (X, Y) = convert_keys_to_ints(X, Y)
    (A, B) = convert_keys_to_ints(A, B)

    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    log.info("Computing cosine similarities...")
    cossims = construct_cossim_lookup(XY, AB)

    log.info("Null hypothesis: no difference between %s and %s in association to attributes %s and %s",
             encs["targ1"]["category"], encs["targ2"]["category"],
             encs["attr1"]["category"], encs["attr2"]["category"])
    log.info("Computing pval...")
    pval = p_val_permutation_test(X, Y, A, B, n_samples, cossims=cossims, parametric=parametric)
    log.info("pval: %g", pval)

    log.info("computing effect size...")
    esize = effect_size(X, Y, A, B, cossims=cossims)
    log.info("esize: %g", esize)
    return esize, pval


