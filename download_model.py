import os
import urllib.request
import json

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main/"

FILES = [
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt"
]

TARGET_DIR = os.path.abspath("local_model")

def download_file(file_path):
    url = BASE_URL + file_path
    dest = os.path.join(TARGET_DIR, file_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    print(f"Downloading {url} to {dest}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"Successfully downloaded {file_path}")
    except Exception as e:
        print(f"Error downloading {file_path}: {e}")

if __name__ == "__main__":
    os.makedirs(TARGET_DIR, exist_ok=True)
    for f in FILES:
        download_file(f)
    print("Done downloading all files!")
