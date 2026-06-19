import random
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / 'intents.json', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = BASE_DIR / "data.pth"
data = torch.load(FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def normalize_text(text):
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def response_for_tag(tag):
    for intent in intents['intents']:
        if tag == intent["tag"]:
            return random.choice(intent['responses'])
    return None


def get_rule_based_response(msg):
    normalized_msg = normalize_text(msg)
    if not normalized_msg:
        return None

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            normalized_pattern = normalize_text(pattern)
            if normalized_msg == normalized_pattern:
                return random.choice(intent['responses'])

            similarity = SequenceMatcher(None, normalized_msg, normalized_pattern).ratio()
            if similarity >= 0.78:
                return random.choice(intent['responses'])

    words = set(normalized_msg.split())
    if "order" in words and {"how", "place", "process"} & words:
        return response_for_tag("order_help")
    if {"menu", "serve", "drinks", "drink", "order"} & words:
        return response_for_tag("menu")
    if {"recommend", "best", "strong", "choose"} & words or ("should" in words and "drink" in words):
        return response_for_tag("coffee_recommendation")
    if {"pay", "payment", "card", "cash", "paypal"} & words:
        return response_for_tag("payment")
    if {"deliver", "delivery", "shipping"} & words:
        return response_for_tag("delivery")
    if {"open", "hours", "available", "close"} & words:
        return response_for_tag("hours")
    if {"where", "address", "location", "shop"} & words:
        return response_for_tag("location")

    return None


def get_response(msg):
    if not msg or not str(msg).strip():
        return "Please type a message so I can help."

    rule_based_response = get_rule_based_response(msg)
    if rule_based_response:
        return rule_based_response

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        response = response_for_tag(tag)
        if response:
            return response
    
    return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

