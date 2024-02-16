import re
import torch
from functools import lru_cache
import faiss
import numpy as np  
from transformers import BertModel, BertTokenizer
import spacy

nlp = spacy.load('en_core_web_sm')

metrics = ["incident", "alert", "resource", "config items", "virtual machines",
           "virtual networks", "storage", "disk", "resource group", "projects",
           "business service", "application"] 

metric_vectors = {i: metric for i, metric in enumerate(metrics)}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize(text):
    return tokenizer(text, return_tensors="pt", truncation=True, padding=True)

model = BertModel.from_pretrained('bert-base-uncased')
@lru_cache(maxsize=128)
def get_embeddings(encoded_text):
    with torch.no_grad():
        return model(**encoded_text).last_hidden_state.mean(dim=1).numpy()
metric_vectors_array = np.array(list(metric_vectors.keys())) 
metric_vectors_array = metric_vectors_array.reshape(len(metric_vectors), 1) 
 
index = faiss.IndexFlatIP(1)
# index = faiss.IndexFlatIP(768)
index.add(metric_vectors_array)

def extract_resource(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'ORG':
            return ent.text
    return None

def process_question(question):
    encoded_q = tokenize(question)
    embeddings = get_embeddings(encoded_q)
    
    resource = extract_resource(question)
    if resource:
        return {"resource": resource}

    distances, indices = index.search(embeddings, 1)  
    metric = metrics[indices[0][0]]
    
    return {
        "metric": metric,
        "similarity_score": distances[0][0] 
    }
