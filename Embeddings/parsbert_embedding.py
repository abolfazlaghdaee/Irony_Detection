from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

model_name = "HooshvareLab/bert-base-parsbert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_parsbert_embeddings(texts, batch_size=32):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding sentences"):
        batch_texts = texts[i:i+batch_size]

        # Tokenize batch
        tokenized = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128 
        ).to(device)

        
        with torch.no_grad():
            outputs = model(**tokenized)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)
