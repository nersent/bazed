from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

def tokenize(example):
    tokens = tokenizer(example['text'], truncation=True, max_length=512)
    return { "input_ids": tokens["input_ids"] }
  
dataset = load_dataset("JeanKaddour/minipile", split="train[:5%]")

dataset = dataset.map(tokenize, batched=True, num_proc=14)

dataset.save_to_disk("ml/dataset")
