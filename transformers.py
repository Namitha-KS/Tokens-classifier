import tensorflow as tf
from transformers import TFBertForTokenClassification, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')

resources = ["incident", "alert", "resource", "config items", "virtual machines", "virtual networks", "storage", "disk", "resource group", "projects", "business service", "application"]
metrics = ["details", "count", "summary"]

questions = [
    "Can you provide details on the most recent incident?",
    "How many downtime incidents have occurred this week?" 
]

for question in questions:

    input_ids = tokenizer.encode(question, truncation=True, padding=True)
  
    outputs = model(tf.constant([input_ids]))
    predictions = outputs[0].numpy()
  
    resources_in_question = []
    metrics_in_question = []
    for token, tag in zip(tokenizer.convert_ids_to_tokens(input_ids[0]), predictions[0]):
         if tag in resources:
             resources_in_question.append(token)
         elif tag in metrics:
             metrics_in_question.append(token)
  
    print(f"Question: {question}") 
    print(f"Resources: {resources_in_question}")
    print(f"Metrics: {metrics_in_question}")
