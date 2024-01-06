# given a claim, return a list of related evidence
import json
import os
from typing import List, Tuple
import time
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import CrossEncoder
import spacy
import numpy as np
from copy import deepcopy
import torch
import openai
import concurrent.futures
import backoff
from collections import Counter
import requests
import re
import itertools
from openai.error import RateLimitError
import bs4
from typing import List, Dict, Any
openai.api_key = ""  # set openai key here

QGEN_PROMPT = """I will check things you said and ask questions.

You said: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle.
To verify it,
1. I googled: Does your nose switch between nostrils?
2. I googled: How often does your nostrils switch?
3. I googled: Why does your nostril switch?
4. I googled: What is nasal cycle?

You said: The Stanford Prison Experiment was conducted in the basement of Encina Hall, Stanford’s psychology building.
To verify it,
1. I googled: Where was Stanford Prison Experiment was conducted?

You said: The Havel-Hakimi algorithm is an algorithm for converting the adjacency matrix of a graph into its adjacency list. It is named after Vaclav Havel and Samih Hakimi.
To verify it,
1. I googled: What does Havel-Hakimi algorithm do?
2. I googled: Who are Havel-Hakimi algorithm named after?

You said: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd.
To verify it,
1. I googled: Who sings the song "Time of My Life"?
2. I googled: Which film is the song "Time of My Life" from?
3. I googled: Who produced the song "Time of My Life"?

You said: Kelvin Hopins was suspended from the Labor Party due to his membership in the Conservative Party.
To verify it,
1. I googled: Why was Kelvin Hopins suspended from Labor Party?

You said: Social work is a profession that is based in the philosophical tradition of humanism. It is an intellectual discipline that has its roots in the 1800s.
To verify it,
1. I googled: What philosophical tradition is social work based on?
2. I googled: What year does social work have its root in?

You said: {claim}
To verify it,
""".strip()

def is_tag_visible(element: bs4.element) -> bool:
    """Determines if an HTML element is visible.

    Args:
        element: A BeautifulSoup element to check the visiblity of.
    returns:
        Whether the element is visible.
    """
    if element.parent.name in [
        "style",
        "script",
        "head",
        "title",
        "meta",
        "[document]",
    ] or isinstance(element, bs4.element.Comment):
        return False
    return True

def parse_api_response(api_response: str) -> List[str]:
    """Extract questions from the OpenAI API response.

    The prompt returns questions as a string with the format of an ordered list.
    This function parses this response in a list of questions.

    Args:
        api_response: Question generation response from GPT-3.
    Returns:
        questions: A list of questions.
    """
    search_string = "I googled:"
    questions = []
    for question in api_response.split("\n"):
        # Remove the search string from each question
        if search_string not in question:
            continue
        question = question.split(search_string)[1].strip()
        questions.append(question)

    return questions

@backoff.on_exception(backoff.expo, RateLimitError)
def run_question_generation(prompt, model, temperature, num_rounds, num_retries=5):
    questions = set()
    for _ in range(num_rounds):
        for _ in range(num_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages = [
                        {
                            "role": "user", "content": prompt
                        }
                    ],
                    temperature=temperature,
                    max_tokens=256,
                )
                cur_round_questions = parse_api_response(
                    response.choices[0]["message"]["content"].strip() 
                )
                    
                questions.update(cur_round_questions)
                break
            except openai.error.OpenAIError as exception:
                print(f"{exception}. Retrying...")
                time.sleep(1)

    questions = list(sorted(questions))
    return questions

def remove_duplicate_questions(model, all_questions):
    qset = [all_questions[0]]
    for question in all_questions[1:]:
        q_list = [(q, question) for q in qset]
        scores = model.predict(q_list)
        if np.max(scores) < 0.60:
            qset.append(question)
    return qset

def scrape_url(url: str, timeout: float = 3) -> Tuple[str, str]:
    """Scrapes a URL for all text information.

    Args:
        url: URL of webpage to scrape.
        timeout: Timeout of the requests call.
    Returns:
        web_text: The visible text of the scraped URL.
        url: URL input.
    """
    # Scrape the URL
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as _:
        return None, url

    # Extract out all text from the tags
    try:
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        texts = soup.findAll(text=True)
        # Filter out invisible text from the page.
        visible_text = filter(is_tag_visible, texts)
    except Exception as _:
        return None, url

    # Returns all the text concatenated as a string.
    web_text = " ".join(t.strip() for t in visible_text).strip()
    # Clean up spacing.
    web_text = " ".join(web_text.split())
    return web_text, url

def search_google(query: str, num_web_pages: int = 10, timeout : int = 6, save_url: str = '') -> List[str]:
    """Searches the query using Google. 
    Args:
        query: Search query.
        num_web_pages: the number of web pages to request.
        save_url: path to save returned urls, such as 'urls.txt'
    Returns:
        search_results: A list of the top URLs relevant to the query.
    """
    query = query.replace(" ", "+")

    # set headers: Google returns different web-pages according to agent device
    # desktop user-agent
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
    # mobile user-agent
    MOBILE_USER_AGENT = "Mozilla/5.0 (Linux; Android 7.0; SM-G930V Build/NRD90M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.125 Mobile Safari/537.36"
    headers = {'User-Agent': USER_AGENT}
    
    # set language
    # set the Google interface language, use &hl=XX
    # set the preferred language of the search results, use &lr=lang_XX
    # set language as en, otherwise it will return many translation web pages to Arabic that can't be opened correctly.
    lang = "en" 

    # scrape google results
    urls = []
    for page in range(0, num_web_pages, 10):
        # here page is google search's bottom page meaning, click 2 -> start=10
        # url = "https://www.google.com/search?q={}&start={}".format(query, page)
        url = "https://www.google.com/search?q={}&lr=lang_{}&hl={}&start={}".format(query, lang, lang, page)
        r = requests.get(url, headers=headers, timeout=timeout)
        # collect all urls by regular expression
        # how to do if I just want to have the returned top-k pages?
        urls += re.findall('href="(https?://.*?)"', r.text)

    # set to remove repeated urls
    urls = list(set(urls))

    # save all url into a txt file
    if not save_url == "":
        with open(save_url, 'w') as file:
            for url in urls:
                file.write(url + '\n')
    return urls

def chunk_text(
    text: str,
    tokenizer,
    sentences_per_passage: int = 5,
    filter_sentence_len: int = 250,
    sliding_distance: int = 2,
) -> List[str]:
    """Chunks text into passages using a sliding window.

    Args:
        text: Text to chunk into passages.
        sentences_per_passage: Number of sentences for each passage.
        filter_sentence_len: Maximum number of chars of each sentence before being filtered.
        sliding_distance: Sliding distance over the text. Allows the passages to have
            overlap. The sliding distance cannot be greater than the window size.
    Returns:
        passages: Chunked passages from the text.
    """
    if not sliding_distance or sliding_distance > sentences_per_passage:
        sliding_distance = sentences_per_passage
    assert sentences_per_passage > 0 and sliding_distance > 0

    passages = []
    try:
        doc = tokenizer(text[:500000])  # Take 500k chars to not break tokenization.
        sents = [
            s.text.replace("\n", " ")
            for s in doc.sents
            if len(s.text) <= filter_sentence_len  # Long sents are usually metadata.
        ]
        for idx in range(0, len(sents), sliding_distance):
            passages.append((" ".join(sents[idx : idx + sentences_per_passage]), idx, idx + sentences_per_passage-1))
    except UnicodeEncodeError as _:  # Sometimes run into Unicode error when tokenizing.
        print("Unicode error when using Spacy. Skipping text.")

    return passages

def get_relevant_snippets(query, tokenizer, passage_ranker, timeout=10, max_search_results_per_query=5, max_passages_per_search_result_to_return=2, sentences_per_passage=5):
    search_results = search_google(query, timeout=timeout)

    with concurrent.futures.ThreadPoolExecutor() as e:
        scraped_results = e.map(scrape_url, search_results, itertools.repeat(timeout))
    # Remove URLs if we weren't able to scrape anything or if they are a PDF.
    scraped_results = [r for r in scraped_results if r[0] and ".pdf" not in r[1]]
    # print("Num Bing Search Results: ", len(scraped_results))
    retrieved_passages = list()
    for webtext, url in scraped_results[:max_search_results_per_query]:
        passages = chunk_text(text=webtext, tokenizer=tokenizer, sentences_per_passage=sentences_per_passage)
        if not passages:
            continue

        # Score the passages by relevance to the query using a cross-encoder.
        scores = passage_ranker.predict([(query, p[0]) for p in passages]).tolist()
        passage_scores = list(zip(passages, scores))

        # Take the top passages_per_search passages for the current search result.
        passage_scores.sort(key=lambda x: x[1], reverse=True)

        relevant_items = list()
        for passage_item, score in passage_scores:
            overlap = False
            if len(relevant_items) > 0:                
                for item in relevant_items:
                    if passage_item[1] >= item[1] and passage_item[1] <= item[2]:
                        overlap = True
                        break
                    if passage_item[2] >= item[1] and passage_item[2] <= item[2]:
                        overlap = True
                        break

            # Only consider top non-overlapping relevant passages to maximise for information 
            if not overlap:
                relevant_items.append(deepcopy(passage_item))
                retrieved_passages.append(
                    {
                        "text": passage_item[0],
                        "url": url,
                        "sents_per_passage": sentences_per_passage,
                        "retrieval_score": score,  # Cross-encoder score as retr score
                    }
                )
            if len(relevant_items) >= max_passages_per_search_result_to_return:
                break
    # print("Total snippets extracted: ", len(retrieved_passages))
    return retrieved_passages

def get_web_evidences_for_claim(claim: str) -> Dict[str, Any]:
    """input: claim/sentence/document
       output: evidences is a dict with two keys: ['aggregated', 'question_wise']
       a = evidences['aggregated']
       b = evidences['question_wise']['specific question']
       a and b are both a list, len=5, each have five pieces of evidence with the keys:
       ['text', 'url', 'sents_per_passage', 'retrieval_score']"""
    evidences = dict()
    evidences["aggregated"] = list()
    question_duplicate_model = CrossEncoder('navteca/quora-roberta-base', device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),)
    tokenizer = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
    passage_ranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length=512,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    # give a claim, sometimes return the empty question list
    questions = []
    while len(questions) <= 0:
        questions = run_question_generation(
            prompt=QGEN_PROMPT.format(claim=claim),
            model = "gpt-3.5-turbo",
            temperature=0.7,
            num_rounds=2,
        )
    questions = list(set(questions))
        
    if len(questions) > 0:
        questions = remove_duplicate_questions(question_duplicate_model, questions)  
    questions = list(questions)
    print(questions)
    snippets = dict()
    for question in questions:
        snippets[question] = get_relevant_snippets(question, tokenizer, passage_ranker, max_search_results_per_query=5, max_passages_per_search_result_to_return=3)
        snippets[question] = deepcopy(sorted(snippets[question], key=lambda snippet: snippet["retrieval_score"], reverse=True)[:5])

    evidences["question_wise"] = deepcopy(snippets)
    while len(evidences["aggregated"]) < 5:
        for key in evidences["question_wise"]:
            # Take top evidences for each question
            index = int(len(evidences["aggregated"])/len(evidences["question_wise"]))
            evidences["aggregated"].append(evidences["question_wise"][key][index])
            if len(evidences["aggregated"]) >= 5:
                break 
    return evidences