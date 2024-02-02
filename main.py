import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_resource_metric(question):
    doc = nlp(question)

    resource_candidates = [
        token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]
    ]
    metric_candidates = [
        token.text
        for token in doc
        if token.pos_ in ["NOUN", "VERB", "ADJ"] and token.dep_ in ["attr", "acomp"]
    ]

    resource = None
    metric = None
    for token in doc:
        if token.dep_ == "nsubj" and token.pos_ in ["NOUN", "PROPN"]:
            resource = token.text
            break

    for token in doc:
        if token.dep_ in ["acomp", "attr"] and token.head.text == resource:
            metric = token.text
            break

    if not resource:
        resource = max(resource_candidates, key=resource_candidates.count) if resource_candidates else None
    if not metric:
        metric = max(metric_candidates, key=metric_candidates.count) if metric_candidates else None

    return resource, metric

question = "Can you provide details on the most recent incident?"
resource, metric = extract_resource_metric(question)
print(f"Question: {question}, Resource: {resource}, Metric: {metric}")

question = "How many downtime incidents have occurred this week?"
resource, metric = extract_resource_metric(question)
print(f"Question: {question}, Resource: {resource}, Metric: {metric}")

question = "Can you provide a summary of recent incidents?"
resource, metric = extract_resource_metric(question)
print(f"Question: {question}, Resource: {resource}, Metric: {metric}")

question = "How many incidents have occurred in the past week?"
resource, metric = extract_resource_metric(question)
print(f"Question: {question}, Resource: {resource}, Metric: {metric}")
