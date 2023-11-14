import os
import json
from typing import Dict, Any

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

if __name__ == "__main__":
    cwd = os.getcwd()
    print("Please ensure current directory {} has input of the screen1, the input and output of screen2".format(cwd))
    files = list(os.listdir(cwd))

    input_screen1 = [file for file in files if file.endswith("screen1.json")][0]
    output_screen2 = [file for file in files if file.endswith("screen2_output.json")][0]
    id_name = input_screen1.split("_")[0].strip()
    output_file = id_name + "_screen3.json"

    d1 = read_json(os.path.join(cwd, input_screen1))
    d2 = read_json(os.path.join(cwd, output_screen2))

    sentences = d1["sentences"]
    try:
        assert(len(sentences) == len(d2))
        i = 0
        for k, v in d2.items():
            d2[k].update({"text": sentences[i]})
            i += 1
    except:
        print("The number of sentences are different!")
        input_screen2 = input_screen1[:-6] + "2.json"
        print(input_screen2)
        d2_input = read_json(os.path.join(cwd, input_screen2))
        for k, v in d2_input.items():
            # sentence index
            i = int(k.replace("sentence", "").strip()) - 1
            if k in d2.keys():
                d2[k].update({"text": sentences[i]})
            else:
                # print(k)
                d2[k] = d2_input[k]
                d2[k].update({"text": sentences[i]})
                d2[k].update({"revised_Claims": [sentences[i]]})
        
        # re-order the sentence index
        d2_sort = {}
        for k,_ in d2_input.items():
            d2_sort[k] = d2[k]
        d2 = d2_sort

    temp = {"prompt": d1['prompt'], "response": d1['response'], "sentences": d2}
    save_json(temp, os.path.join(cwd, output_file))
    print("Generate {} successfully!".format(output_file))