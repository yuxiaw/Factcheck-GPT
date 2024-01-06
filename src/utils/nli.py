from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load model directly
# Sentiment analysis pipeline
# classifier = pipeline("sentiment-analysis", model="roberta-large-mnli")

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

nli_labelmap = {
    "NEUTRAL": 3,
    "CONTRADICTION":2,
    "ENTAILMENT": 1   
}

nli2stance = {
    "NEUTRAL": "irrelevant",
    "CONTRADICTION": "refute",
    "ENTAILMENT": "support"   
}

stance_map = {
    'irrelevant': 3,
    'refute': 2,
    'partially-support': 1,
    'completely-support': 1
}


def nli_infer_prob(premise, hypothesis):
    # predict one example by nli model
    try: 
        input = "<s>{}</s></s>{}</s></s>".format(premise, hypothesis)
        pred = classifier(input)
        # print(pred)
    except:
        # token length > 514
        L = len(premise)
        premise = premise[:int(L/2)]
        input = "<s>{}</s></s>{}</s></s>".format(premise, hypothesis)
        pred = classifier(input)
        # print(pred) 
        # [{'label': 'CONTRADICTION', 'score': 0.9992701411247253}]
    return pred


def nli_infer(premise, hypothesis):
    # predict one example by nli model
    try: 
        input = "<s>{}</s></s>{}</s></s>".format(premise, hypothesis)
        pred = classifier(input)
        # print(pred)
    except:
        # token length > 514
        L = len(premise)
        premise = premise[:int(L/2)]
        input = "<s>{}</s></s>{}</s></s>".format(premise, hypothesis)
        pred = classifier(input)
        # print(pred) 
        # [{'label': 'CONTRADICTION', 'score': 0.9992701411247253}]
    return nli2stance[pred[0]['label']]