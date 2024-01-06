# code for general evaluation
# https://github.com/luozhouyang/python-string-similarity
from strsimpy.normalized_levenshtein import Levenshtein, NormalizedLevenshtein
from strsimpy.ngram import NGram
import evaluate
import numpy as np
from torchmetrics.text.bert import BERTScore
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("princeton-nlp/sup-simcse-roberta-large")
# https://huggingface.co/sentence-transformers
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def evaluate_classification(preds, gold):
    metric = evaluate.load("bstrai/classification_report")
    return metric.compute(predictions=preds, references=gold)

def eval_classification(y_true, y_pred, average="macro"):
    precision, recall, F1, support = precision_recall_fscore_support(y_true, y_pred, average=average)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "F1": round(F1, 3),
    }
    return metrics


def eval_binary(y_true, y_pred, pos_label=1, average="binary"):
    """pos_label: postive label is machine text here, label is 1, human text is 0"""
    precision, recall, F1, support = precision_recall_fscore_support(
        y_true, y_pred, pos_label = pos_label, average = average)
    # accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # precison
    # pre = precision_score(y_true, y_pred, pos_label = pos_label, average = average)
    # recall
    # rec = recall_score(y_true, y_pred, pos_label = pos_label, average = average)
    metrics = {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "F1": round(F1, 3),
    }
    return metrics


def word_overlap(s1, s2):
    La, Lb = [], []
    tokens1 = [token.strip() for token in s1.split() if token.strip() != ""]
    tokens2 = [token.strip() for token in s2.split() if token.strip() != ""]
    if len(tokens1) < len(tokens2):
        La, Lb = tokens2, tokens1
    else:
        La, Lb = tokens1, tokens2
    intersection = [token for token in La if token in Lb]
    return round(len(intersection)/len(La), 3)

def lexical_distance_two_text(s1, s2):
    normalized_levenshtein = NormalizedLevenshtein()
    levenshtein = Levenshtein()
    bigram = NGram(2)
    distance = {
        "edit_distance": levenshtein.distance(s1, s2),
        "norm_edit_distance": normalized_levenshtein.distance(s1, s2),
        "norm_edit_similarity": normalized_levenshtein.similarity(s1, s2),
        "bigram_distance": bigram.distance(s1, s2),
        "word_overlap": word_overlap(s1, s2)
    }
    return distance

def lexical_distance(S1, S2):
    assert(len(S1) == len(S2))
    distance = {}
    for i, (s1,s2) in enumerate(zip(S1, S2)):
        try:
            temp = lexical_distance_two_text(s1, s2)
            for k,v in temp.items():
                if distance.get(k) is None:
                    distance[k] = []
                distance[k] += [v]
        except Exception as error:
            print(f"Example {i} has {error}.")
    # due to exception, this assertation may not satisfy
    # for k, v in distance.items():
    #     assert(len(v) == len(S1))
    return distance


# -----------------------------------------------
# BERTScore
# -----------------------------------------------
# from torchmetrics.text.bert import BERTScore

def bertscore(preds, target, batch_size = 16):
    assert(len(preds) == len(target))
    bertscore = BERTScore(model_name_or_path='roberta-large', batch_size=batch_size)

    epochs = int(len(preds)/batch_size)
    if epochs * batch_size < len(preds):
        epochs += 1
    # print(epochs)

    metric = {'precision': [], 'recall': [], 'f1': []}    
    for i in range(epochs):
        s = i*batch_size
        e = (i+1)*batch_size
        if e >= len(preds):
            e = len(preds)
        sim = bertscore(preds[s:e], target[s:e])
        for k, _ in metric.items():
            metric[k] += list(sim[k].numpy())
    return metric


# -----------------------------------------------
# cosine similarity with sentence embeddings
# -----------------------------------------------
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def cosine_similarity(s1, s2):
    #Sentences are encoded by calling model.encode()
    embedding = model.encode([s1, s2])
    v1 = embedding[0]
    v2 = embedding[1]
    return np.nan_to_num(cosine(np.nan_to_num(v1), np.nan_to_num(v2)))


def semantic_similarity(preds, target):    
    sts = []
    for i, (s1, s2) in enumerate(zip(preds, target)):
        sts.append(cosine_similarity(s1,s2))
    # result = round(np.array(sts).mean(0), 3)
    # print(result)
    return sts
    

def eval_response(preds, target1, target2=None, batch_size=16):
    """target1 should be the original response, target2 should be a gold reference revision"""
    if target2 is None:
        target2 = target1
    metric = lexical_distance(preds, target1)
    metric.update(bertscore(preds, target2, batch_size = batch_size))
    metric["cosine-sentence-roberta"] = semantic_similarity(preds, target2)
    for k, v in metric.items():
        print(k, round(np.array(v).mean(0), 3))
    return metric