import csv
import json
import numpy as np
from collections import Counter
from typing import Dict, List, Any


def save_to_file(text, filename='error_output.txt'):
    """Save a string to a file line by line."""
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(text + '\n')

def majority_vote(input_list):
    # Use Counter to count occurrences of each element
    counter = Counter(input_list)
    
    # Find the element with the maximum count (majority)
    majority_element = max(counter, key=counter.get)
    
    # Return the majority element
    return majority_element
    
def is_float(string):
    if string.replace(".", "").isnumeric():
        return True
    else:
        return False
    
def save_json(dictionary: Dict[str, Any], save_dir: str) -> None:
    # Serializing json
    json_object = json.dumps(dictionary, indent=4, ensure_ascii=False)

    # Writing to sample.json
    with open(save_dir, "w", encoding='utf-8') as outfile:
        outfile.write(json_object)


def read_json(filepath: str) -> Dict[str, Any]:
    data = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def list_to_dict(data: List[Dict[str, Any]]) -> Dict[int, Any]:
    temp = {}
    for i, d in enumerate(data):
        temp[i] = d
    return temp


def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

# def load_jsonl(input_path) -> list:
#     """
#     Read list of objects from a JSON lines file.
#     """
#     data = []
#     with open(input_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data.append(json.loads(line.rstrip('\n|\r')))
#     print('Loaded {} records from {}'.format(len(data), input_path))
#     return data

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


def cosine(u, v):
    """based on embeddings and calculate cosine similarity"""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def read_csv(input_file, quotechar=None):
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
    return lines

def save_csv(header, data, output_file):
    with open(output_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # write the header
        writer.writerow(header)
        # write multiple rows
        writer.writerows(data)


def save_array(filename, embeddings):
    # save embeddings into file
    with open(filename, 'wb') as f:
        np.save(f, embeddings)

def load_array(filename):
    with open(filename, 'rb') as f:
        a = np.load(f)
    return a

def read_txt(input_file):
    with open(input_file, "r", encoding = "utf-8") as f:
        return f.readlines()

def save_txt(data, output_file):
    with open(output_file, "w", encoding = "utf-8") as writer:
        writer.write("\n".join(data))

def clean_text(text):
    for mark in ['"', '-', '\t', ' ']:
        for i in [5, 4, 3, 2]:
            marks = mark * i
            text = text.replace(marks, '')
    return text
