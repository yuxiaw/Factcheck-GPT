from utils.prompt import CHECKWORTHY_PROMPT, SPECIFY_CHECKWORTHY_CATEGORY_PROMPT
from utils.openaiAPI import gpt
from typing import List
checkworthiness_type_label_map = {
    1: "factual claim",
    2: "opinion",
    3: "not a claim or statement, but questions or imperative sentences",
    4: "others"
}

def identify_checkworthiness(texts: List[str],
                             model: str="gpt-3.5-turbo", 
                             system_role: str="You are a helpful factchecker assistant.",
                             num_retries: int=3) -> List[str]:
    """input: a list of texts to identify whether they are worth fact checking
       output: a list of ["Yes", "No", "No", ...] """
    # if gpt is unable to return correct results, we assume all texts are checkworthy
    results = ["Yes"]*len(texts)
    for _ in range(num_retries):
        try:
            user_input = CHECKWORTHY_PROMPT.format(texts = texts)
            results = eval(gpt(user_input, model=model, system_role=system_role))
            assert(len(results) == len(texts))
            break
        except AssertionError as e:
            print(f"An unexpected error occurred: {e}")
            print(f"There is {len(texts)} texts, while {len(results)} checkworthy predictions.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}") 
    return results 


def specify_checkworthiness_type(text:str,
                                 model: str="gpt-3.5-turbo", 
                                 system_role: str="You are a helpful factchecker assistant.",
                                 num_retries: int=3) -> int:
    """input: a sentence, specify it is 1. a factual claim; 2. an opinion; 
              3. not a claim (like a question or a imperative sentence); 4. other categories.
       output: select from 1,2,3,4 """
    # if gpt is unable to return correct results, we assume the input is checkworthy
    
    results = 1
    for _ in range(num_retries):
        try:
            user_input = SPECIFY_CHECKWORTHY_CATEGORY_PROMPT.format(sentence = text)
            results = eval(gpt(user_input, model=model, system_role=system_role))
            if results in [1,2,3,4]:
                break
        except Exception as e:
            print(f"An unexpected error occurred: {e}") 
    print(checkworthiness_type_label_map[results])
    return results 