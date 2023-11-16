from transformers import pipeline
import joblib
import torch

device = 0 if torch.cuda.is_available() else -1

#MODEL = "facebook/bart-large-mnli"
#MODEL = "microsoft/deberta-v3-large"
#MODEL = "microsoft/deberta-base"
MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v1"
#MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v1"

CACHE_FILE = f"model_cache_{MODEL.replace('/', '_')}.joblib"

def load_model():
    try:
        return joblib.load(CACHE_FILE)
    except FileNotFoundError:
        model = pipeline(model=MODEL,device=device)
        joblib.dump(model, CACHE_FILE)
        return model

def extract_topics(topics, text):
    model = load_model()
    result = model(text, candidate_labels=topics, multi_label=False)
    return result["scores"]

if __name__ == "__main__":
    import sys
    topics = sys.argv[1].split(',')
    text = sys.argv[2]
    scores = extract_topics(topics, text)
    print(" ".join(map(str, scores)))
