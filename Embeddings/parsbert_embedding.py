from transformers import AutoTokenizer, AutoModel
import torch


model_name = "HooshvareLab/bert-base-parsbert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_parsbert_embeddings(texts):
    tokenized_train = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    tokenized_train.to(device)
    with torch.no_grad():
        embeddings = model(**tokenized_train).last_hidden_state[0, 0, :].cpu().numpy()
    return embeddings


