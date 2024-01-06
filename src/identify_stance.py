from utils.openaiAPI import gpt
from utils.prompt import IDENTIFY_STANCE_PROMPT
from utils.nli import nli_infer

stance_map = {
    "A": "support",
    "B": "refute",
    "C": "irrelevant"
}

def parse_stance_results(r):
    try:
        return stance_map[r[0]]
    except KeyError:
        if "A" in r and "support" in r:
            return "support"
        elif "B" in r and "refute" in r:
            return "refute"
        elif "C" in r and "irrelevant" in r:
            return "irrelevant"
    except Exception as e:
        print(f"An unexpected error occurred: {r}.")
        return "irrelevant"


def identify_stance_gpt(evidence, claim, model="gpt-3.5-turbo-0613"):
    user_input = IDENTIFY_STANCE_PROMPT.format(claim=claim, evidence=evidence)
    # print(user_input)
    r = gpt(user_input, model = model, 
        system_role="You are a helpful factchecker assistant.", 
        num_retries=3, waiting_time = 1)
    return parse_stance_results(r)


def stance(evidence, claim, model="gpt-3.5-turbo-0613"):
    """input: a claim and an evidence
       output: label in [support, refute, irrelevant]"""
    if model == "nli":
        label = nli_infer(premise=evidence, hypothesis=claim)
    elif "gpt" in model:
        label = identify_stance_gpt(evidence, claim, model=model)
    else:
        print("Check the model argument, choose either gpt or nli model")
    return label