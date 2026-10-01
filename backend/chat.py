import random
import re
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

CONFIDENCE_THRESHOLD = 0.75
FILL_IN = (
    "I don't have the answer for yur question right now. I will send this message to our team for you to get the answer you deserve."
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents= json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

RESPONSES = {i["tag"]: i["responses"] for i in intents["intents"]}

def extract_seats(text: str) -> int | None:
    m = re.search(r"(\d+)\s*(?:seats?|people|users|analysts|members|person|associates)", text.lower())
    if not m:
        m = re.search(r"team of (\d+)", text.lower())
    return int(m.group(1)) if m else None


def seat_quote(n: int) -> str:
    if n < 1:
        return ""
    if n <= 5:
        return f" For {n} seat{'s' if n > 1 else ''}, that's ${n * 250:,}/month ($250 each)."
    if n <= 20:
        return f" For {n} seats, that's ${n * 200:,}/month ($200 each)."
    return f" For {n} seats, you'd want to talk to the team at expertmatch.fit/contact."

def predict(msg: str) -> tuple[str, float]:
    """Return (predicted tag, probability)."""
    bag = bag_of_words(tokenize(msg), all_words)
    if bag.sum() == 0:  # no known words at all -> don't let the net guess
        return "", 0.0
    x = torch.from_numpy(bag.reshape(1, -1)).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return tags[idx], float(probs[idx])


def get_response(msg: str) -> str:
    tag, prob = predict(msg)
    if prob < CONFIDENCE_THRESHOLD:
        return FILL_IN
    answer = random.choice(RESPONSES[tag])
    if tag == "pricing":
        seats = extract_seats(msg)
        if seats:
            answer += seat_quote(seats)
    return answer

if __name__ == "__main__":
    print("Hi! I'm Effie, Let's chat! If you have no questions, type 'quit' to exit.")
    while True:
        sentence = input("You: ")
        if sentence.strip().lower() == "quit":
            break
        tag, prob = predict(sentence)
        print(f"  [intent={tag or 'none'} confidence={prob:.2f}]")
        print(f"Bot: {get_response(sentence)}")