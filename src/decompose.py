import nltk
# nltk.download()
# from nltk import sent_tokenize
# import spacy
# nlp = spacy.load("en_core_web_sm")
from utils.openaiAPI import gpt
from utils.data_util import save_to_file
from utils.prompt import DOC_TO_INDEPEDENT_SENTENCES_PROMPT, SENTENCES_TO_CLAIMS_PROMPT, DOC_TO_SENTENCES_PROMPT
from typing import List


def doc_to_sents(text: str, tool_name = "nltk") -> List[str]:
    if tool_name == "nltk":
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip())>=3]
    # elif tool_name == "spacy":
    #     doc = nlp(text)
    #     sentences = [str(sent).strip() for sent in doc.sents]
    return sentences


def doc2sentences(doc: str, mode: str="independent_sentences",
                  model: str="gpt-3.5-turbo", 
                  system_role: str="You are good at decomposing and decontextualizing text.",
                  num_retries: int=3) -> List[str]:
    if mode == "sentences":
        prompt = DOC_TO_SENTENCES_PROMPT
    elif mode == "independent_sentences":
        prompt = DOC_TO_INDEPEDENT_SENTENCES_PROMPT
    elif mode == "claims":
        prompt = SENTENCES_TO_CLAIMS_PROMPT

    results = None
    user_input = prompt.format(doc=doc).strip()
    for _ in range(num_retries):
        try:
            r = gpt(user_input, model=model, system_role=system_role)
            results = eval(r)
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}.")
            save_to_file(r)

    if isinstance(results, list):
        return results
    else:
        print(f"{model} output {r}. It does not output a list of sentences correctly, return NLTK split results.")
        return doc_to_sents(doc, tool_name = "nltk")