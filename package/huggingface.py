from transformers import pipeline
import joblib
import torch
import sys

device = 0 if torch.cuda.is_available() else -1

# MODEL = "facebook/bart-large-mnli"
# MODEL = "microsoft/deberta-v3-large"
# MODEL = "microsoft/deberta-base"
MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v1"
# MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v1"
ää
CACHE_FILE = f"/tmp/model_cache_{MODEL.replace('/', '_')}.joblib"

def load_model():
    try:
        return joblib.load(CACHE_FILE)
    except FileNotFoundError:
        model = pipeline(model=MODEL, device=device)
        joblib.dump(model, CACHE_FILE)
        return model

def extract_topics(topics, text):
    model = load_model()
    result = model(text, candidate_labels=topics, multi_label=False)
    return result["scores"]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <topics> <text_file_path>")
        sys.exit(1)

    topics = sys.argv[1].split(',')
    text_file_path = sys.argv[2]

    try:
        with open(text_file_path, 'r') as text_file:
            text = text_file.read()
    except FileNotFoundError:
        print(f"Error: Text file not found: {text_file_path}")
        sys.exit(1)

    scores = extract_topics(topics, text)
    print(" ".join(map(str, scores)))
