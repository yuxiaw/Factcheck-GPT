# baseline code for five subtasks
import os
import random
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
from utils.eval_utils import evaluate_classification, eval_response

from utils.openaiAPI import gpt
from utils.nli import nli_infer_prob, nli_labelmap
from utils.prompt import zero_shot_claim_checkworthiness, zero_shot_sentence_checkworthiness
from utils.prompt import zero_shot_edit_response, zero_shot_edit_response_given_question
from utils.prompt import zero_shot_claim_evidence_stance
from utils.prompt import zero_shot_edit_response, zero_shot_edit_response_given_question

# -------------------------------------------------------------------
# Subtask 1 and 2: checkworthiness detection of sentence and claim
# -------------------------------------------------------------------
checkworthiness_label_map = {
    'yes': 'yes',
    'not_a_claim': 'no', 
    'opinion': 'no', 
    'other': 'no',
    'no': 'no'
}

factuality_label_map = {
    "factual": 1,
    "opinion":2,
    "not_a_claim":3,
    "other":4,
    "1":1,
    "2":2,
    "3":3,
    "4":4
}

def if_checkworthy(text, granularity, model="gpt-3.5-turbo-0613"):
    if granularity == "sentence":
        user_input = zero_shot_sentence_checkworthiness.format(sentence=text)
    elif granularity == "claim":
        user_input = zero_shot_claim_checkworthiness.format(claim=text)
    # print(user_input)
    r = gpt(user_input, model = model, 
        system_role="You are a helpful factchecker assistant.", 
        num_retries=3, waiting_time = 1)
    return r


def detect_checkworthiness(texts, granularity, model="gpt-3.5-turbo-0613", savedir="st1_gpt3.5_zs.json"):
    responses = {}
    for i, text in enumerate(texts):
        r = if_checkworthy(text, granularity=granularity, model=model)
        responses[i] = r
        if i % 20 == 0:
            pd.Series(responses).to_json(savedir)

    pd.Series(responses).to_json(savedir)
    return responses


def plot_cm(gold, preds, granularity=["sentence", "claim"][0], fig_savepath="../fig/"):
    fig_savepath = os.path.join(fig_savepath, f"{granularity}_checkworthy_cm.pdf")

    cm = confusion_matrix(gold, preds)
    if granularity == "sentence":
        # sentence label space
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, 
                                                    display_labels = ["no", "yes"])
    elif granularity == "claim":
        # claim label space
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, 
                                                    display_labels = ["factual", "opinion", "not_a_claim", "other"])
    cm_display.plot()
    # plt.show()
    plt.tight_layout()
    plt.savefig(fig_savepath, format='pdf')


def read_sentences(datadir):
    data = pd.read_json(datadir, lines=True)
    labels = [checkworthiness_label_map[l] for l in list(data['checkworthy'])]
    sentences = data['sentence']
    return sentences, labels

def read_claims(datadir="../subtasks_data/subtask2_claim_checkworthiness.jsonl"):
    data = pd.read_json(datadir, lines=True)
    claims = data['claim']
    labels = [factuality_label_map[l] for l in list(data['checkworthy'])]
    return claims, labels


def eval_claim_checkworthiness(datadir="../subtasks_data/subtask1_sentence_checkworthiness.jsonl", 
                               response_savedir = "../subtasks_data/result/st2_gpt3.5_zs.json",
                               model="gpt-3.5-turbo-0613"):
     
    claims, labels = read_claims(datadir)
    if os.path.exists(response_savedir):
        print(f"Load cached predictions from {response_savedir}...")
        responses = pd.read_json(response_savedir, typ="Series")
    else:
        responses = detect_checkworthiness(claims, granularity="claim", model=model, savedir=response_savedir)

    # some predictions are unable to parse out, get corresponding pred and gold labels here
    preds, gold = [], [] 
    for i, r in responses.items():
        try:
            preds.append(factuality_label_map[r.split(".")[0].strip().lower()])
            gold.append(labels[i])
        except:
            print(i, r)

    metrics = evaluate_classification(preds=preds, gold=gold)
    return preds, gold, metrics


def eval_sentence_checkworthiness(datadir="../subtasks_data/subtask1_sentence_checkworthiness.jsonl", 
                                  response_savedir = "../subtasks_data/result/st1_gpt3.5_zs.json",
                                  model="gpt-3.5-turbo-0613"):
    sentences, labels = read_sentences(datadir)
    if os.path.exists(response_savedir):
        print(f"Load cached predictions from {response_savedir}...")
        responses = pd.read_json(response_savedir, typ="Series")
    else:
        responses = detect_checkworthiness(sentences, granularity="sentence", model=model, savedir=response_savedir)

    # some predictions are unable to parse out, get corresponding pred and gold labels here
    preds, gold = [], [] 
    for i, r in responses.items():
        try:
            preds.append(checkworthiness_label_map[r.split("\n")[0].strip().lower()])
            gold.append(labels[i])
        except:
            print(i, r)

    # metrics = eval_classification(gold, preds, average="macro")
    # print(eval_binary(gold, preds, pos_label="yes", average="binary"))
    # print(eval_binary(gold, preds, pos_label="no", average="binary"))
    label2id = {"yes":1, "no": 0}
    preds = [label2id[i] for i in preds]
    gold = [label2id[i] for i in gold]
    metrics = evaluate_classification(preds=preds, gold=gold)

    return preds, gold, metrics


def all_checkworthy(datadir="../subtasks_data/subtask1_sentence_checkworthiness.jsonl", granularity = "sentence"):
    if granularity == "sentence":
        _, labels = read_sentences(datadir)
        # all guess as checkworthy
        yes_preds = ["yes"] * len(labels)
        label2id = {"yes":1, "no": 0}
        print(evaluate_classification(preds=[label2id[i] for i in yes_preds], 
                                      gold=[label2id[i] for i in labels]))
    elif granularity == "claim":
        _, labels = read_claims(datadir)
        # all guess as factual claim, to label = 1
        print(evaluate_classification(preds=[1]*len(labels), gold=labels))


# -------------------------------------------------------------------
# Subtask 3: identify stance between (evidence, claim)
# -------------------------------------------------------------------
four_label_stance_map = {
    'irrelevant': 4,
    'refute': 3,
    'partially-support': 2,
    'completely-support': 1,
    'support': 1
}

three_label_stance_map = {
    'irrelevant': 3,
    'refute': 2,
    'partially-support': 1,
    'completely-support': 1
}

def random_guess(gold_stance):
    rg = []
    for i in range(len(gold_stance)):
        rg.append(random.sample([1,2,3,4], k=1))

    metrics = evaluate_classification(gold=gold_stance, preds=rg)
    print(metrics)


def identify_stance_gpt_zs(datadir="../subtasks_data/subtask3_claim_evidence_stance.jsonl", 
                           savedir="../subtasks_data/result/st3_gpt3.5_zs.json",
                           model="gpt-3.5-turbo-0613",
                           prompt=zero_shot_claim_evidence_stance):
    
    data = pd.read_json(datadir, lines=True)

    responses = {}
    for i, row in data.iterrows():
        user_input = prompt.format(evidence=row['evidence'], claim=row['claim'])
        # print(user_input)
        r = gpt(user_input, model = model, 
            system_role="You are a helpful factchecker assistant.", 
            num_retries=3, waiting_time = 1)
        responses[i] = r
        if i % 20 == 0:
            pd.Series(responses).to_json(savedir)

    pd.Series(responses).to_json(savedir)
    return pd.Series(responses)


def eval_subtask3(datadir="../subtasks_data/subtask3_claim_evidence_stance.jsonl", 
                  response_savedir="../subtasks_data/result/st3_gpt3.5_zs.json",
                  model="gpt-3.5-turbo-0613",
                  prompt=zero_shot_claim_evidence_stance,
                  stance_map = four_label_stance_map):
    # load response and data
    if os.path.exists(response_savedir):
        print(f"Load cached predictions from {response_savedir}...")
        responses = pd.read_json(response_savedir, typ="Series")
    else:
        responses = identify_stance_gpt_zs(datadir, response_savedir, model=model, prompt=prompt)

    # load gold labels
    data = pd.read_json(datadir, lines=True)
    gold_labels = list(data['stance'])
    gold_labels = [stance_map[s] for s in gold_labels]

    # parse predictions from responses
    preds, gold = [], []
    for k, v in responses.items():
        try:
            label = v.strip()[0]
            preds.append(int(label))
            gold.append(gold_labels[k])
        except:
            print(k, v)
    
    metrics = evaluate_classification(gold=gold, preds=preds)
    # print(metrics)
    return gold, preds, metrics


# Three-labels: support, refute and irrelevant
# RoBERTa-mnli prediction
def nli_predict_stance(datadir="../subtasks_data/subtask3_claim_evidence_stance.jsonl", 
                       savedir="../subtasks_data/result/st3_roberta_mnli.json"):
    
    stance_map = {
        'irrelevant': 3,
        'refute': 2,
        'partially-support': 1,
        'completely-support': 1
    }
    data = pd.read_json(datadir, lines=True)
    # load response and data
    if os.path.exists(savedir):
        print(f"Load cached predictions from {savedir}...")
        preds = pd.read_json(savedir, typ="Series").to_dict()
    else:
        preds = {}
        for i, row in data.iterrows():
            evidence, claim = row['evidence'], row['claim']
            pred = nli_infer_prob(evidence, claim)
            preds[i] = pred[0] # a dict like {"label":"NEUTRAL","score":0.934855938}

            # break
            if i % 10 == 0:
                # print(preds)
                pd.DataFrame(preds).to_json(savedir)
        pd.DataFrame(preds).to_json(savedir)
            
    gold = [stance_map[s] for s in data['stance']]
    predictions = [nli_labelmap[p['label']] for p in preds.values()]

    # eval for each label class
    # for i in [1,2,3]:
    #     temp = eval_classification(y_true=gold, y_pred=predictions, labels=[i], average="macro")
    #     print(temp)
    # print(eval_classification(y_true=gold, y_pred=predictions, labels=None, average="macro"))
    metrics = evaluate_classification(preds=predictions, gold=gold)
    return gold, predictions, metrics


def plot_cm_subtask3(gold, preds, fig_savedir="../fig/claim_evid_stance_cm.pdf"):
    # plot confusion matrix
    cm = confusion_matrix(gold, preds)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, 
            display_labels = ["support", "partial support", "refute", "irrelevant"])
    cm_display.plot()
    # plt.show()
    plt.tight_layout()
    plt.savefig(fig_savedir, format='pdf')


# prompt llama2 and chatgpt for prediction
def eval_subtask3_three_label_results():
    stance_map = {
        'irrelevant': 3,
        'refute': 2,
        'partially-support': 1,
        'completely-support': 1,
        'support': 1
    }

    # evaluate results by llama2 and chatgpt by three labels
    df1 = pd.read_json("../subtasks_data/result/subtask3_judgements_llama2.json")
    gold_stance = [stance_map[l] for l in list(df1['stance'])]
    llama2_labels = list(df1['llama2_label'])
    llama2_preds = [stance_map[l] for l in llama2_labels]
    print(evaluate_classification(gold=gold_stance, preds=llama2_preds))

    df2 = pd.read_json("../subtasks_data/result/subtask3_judgements_chatgpt.json")
    print(len(df2))  # 3285 < 3305
    chatgpt_labels = list(df2['chatgpt_label'])
    gold_stance = [stance_map[l] for l in list(df2['stance'])]
    chatgpt_preds = [stance_map[l] for l in chatgpt_labels]
    print(evaluate_classification(gold=gold_stance, preds=chatgpt_preds))


# -------------------------------------------------------------------
# Subtask 5: Edit the false response based on correct claims
# -------------------------------------------------------------------
def revise_response(response, claim_list, question=None,
                    prompt_mode=["with-question", "no-question"][1],
                    model=["gpt-3.5-turbo-0613", "gpt-4-0613"][1]):
    
    if prompt_mode == "with-question":
        user_input = zero_shot_edit_response_given_question.format(
        prompt=question, response=response, claims=claim_list)
    else:
        user_input = zero_shot_edit_response.format(response=response, claims=claim_list)
    
    r = gpt(user_input, model=model,
            system_role="You are good at correcting factual errors depending on correct claims.")
    return r


def subtask5(datadir="../subtasks_data/subtask5_revision.jsonl", savedir="../subtasks_data/result/"):
    df = pd.read_json(datadir, lines=True)

    data = {}
    for i, row in df.iterrows():
        if row["factual state"] == "false":
            data[i] = {
                'factual state': row['factual state'],
                'claim list': row['claim list'],
                'prompt': row['prompt'],
                'response': row['response'],
                'revised_response': row['revised_response'],
            }
            for prompt_mode in ["with-question", "no-question"]:
                for model in ["gpt-3.5-turbo-0613", "gpt-4-0613"]:
                    r = revise_response(response=row["response"], claim_list=row["claim list"], question=row["prompt"],
                                        prompt_mode=prompt_mode, model=model)
                    model_name = "-".join(model.split("-")[:2])
                    data[i].update({f"{prompt_mode}-{model_name}": r})
        else:
            continue

        df_new = pd.DataFrame.from_dict(data, orient="index")
        df_new.to_json(os.path.join(savedir, "st5.jsonl"), lines=True, orient='records')


def evaluate_revisions(datadir="../subtasks_data/result/st5.jsonl"):
    """We evaluate by intrinsic metrics and human preference evaluation.
    # edit-distance between the original response and the revision: keep original style
    # BERTScore between human revised_response and the revision: remain correct semantics"""

    df = pd.read_json(datadir, lines=True)
    print(len(df), df.columns)

    metrics = {}
    key1 = "response"
    key2 = "revised_response"
    target1 = list(df[key1])
    target2 = list(df[key2])
    for prompt_mode in ["with-question", "no-question"]:
        for model_name in ["gpt-3.5", "gpt-4"]:
            pred_key = f"{prompt_mode}-{model_name}"
            preds = list(df[pred_key]) 
            print(f"{pred_key} \n")
            metric = eval_response(preds, target1=target1, target2=target2)
            metrics[pred_key] = metric
    return metrics