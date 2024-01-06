# give a claim, a list of evidence, determine whether the claim is factually correct or not.
# There are two strategies: 
# (1) first get stance of evidence against the claim, and then aggregate the stances; 
# (2) directly determine true/false of the claim based on all evidence

from identify_stance import stance
from utils.prompt import VERIFY_PROMPT
from utils.openaiAPI import gpt
from utils.data_util import save_to_file
from typing import List, Any, Dict
import pandas as pd


def verify_by_stance(claim: str, evidences: List[str], model:str = "gpt-3.5-turbo-0613") -> Any:
    labels = []
    for evidence in evidences:
        labels.append(stance(evidence, claim, model=model))
    
    # based on stances of evidence, determine the true/false claim by rules
    # if there is one evidence supports, we assume it is correct
    if "support" in labels:
        return True
    # if there isn't support, but refute and irrelevant, we regard as false
    elif "refute" in labels:
        return False
    else:
        # all irrelevant
        return False


def verify_claim(claim: str, evidences: List[str], 
                  model:str = "gpt-3.5-turbo-0613", num_retries=3) -> Dict[str, Any]:
    results = {}
    user_input = VERIFY_PROMPT.format(claim=claim, evidence=evidences)
    for _ in range(num_retries):
        try:
            r = gpt(user_input, model = model, 
                    system_role="You are a helpful factchecker assistant.", 
                    num_retries=3, waiting_time = 1)
            results = eval(r)
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}.")
            save_to_file(r, "verification_error.txt")

    if isinstance(results, dict):
        return results
    else:
        print(f"Error output {r}. It does not output a dict, return factual label by stance aggregation.")
        factual_label = verify_by_stance(claim, evidences, model)
        results = {
            "reasoning": "",
            "error": "",
            "correction": "",
            "factuality": factual_label
        }
        return results


def verify_document(claims: List[str], evidence: List[List[str]],
                    model: str = "gpt-3.5-turbo-0613", num_retries: int=3) -> Any: 
    results = []
    for claim, evidence_list in zip(claims, evidence):
        result = verify_claim(claim, evidence_list, model=model, num_retries=num_retries)
        result["claim"] = claim
        result["evidence"] = evidence_list
        results.append(result)
    
    # aggregate claim verification results
    df = pd.DataFrame(results)
    # to_return = {}  
    # to_return = {
    #     "reasoning": "\n".join(list(df["reasoning"])),
    #     "error": "\n".join(list(df["error"])),
    #     "correction": "\n".join(list(df["correction"])),
    #     "factuality": all(df["factuality"])
    # }
    return all(df["factuality"]), df
    
   