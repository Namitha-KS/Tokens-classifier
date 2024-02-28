import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import InputExample, InputFeatures

data = [
    {"question": "Can you provide details on the most recent incident?", "label": "Incident,Details"},
    {"question": "How many downtime incidents have occurred this week?", "label": "Incidents,Downtime"},
    {"question": "Can you provide a summary of recent incidents?", "label": "Incidents,Summary"},
    {"question": "How many incidents have occurred in the past week?", "label": "Incidents,Count"},
]

data_df = pd.DataFrame(data)

label_mapping = {label: idx for idx, label in enumerate(data_df['label'].unique())}

data_df['label_int'] = data_df['label'].map(label_mapping)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def convert_question_to_example(question, label):
    return InputExample(guid=None, text_a=question, text_b=None, label=label)

examples = [convert_question_to_example(q, l) for q, l in zip(data_df['question'], data_df['label_int'])]

features = [InputFeatures(input_ids=tokenizer.encode(example.text_a, add_special_tokens=True),
    attention_mask=tokenizer.encode(example.text_a, add_special_tokens=True),
    label=example.label) for example in examples]

max_length = max([len(f.input_ids) for f in features])

padded_features = []
for f in features:
    inputs = {"input_ids": f.input_ids, "attention_mask": f.attention_mask}
    padded_inputs = tokenizer.pad(inputs, max_length=max_length, return_tensors="pt")
    padded_input_ids = padded_inputs["input_ids"].squeeze()
    padded_attention_mask = padded_inputs["attention_mask"].squeeze()
    padded_features.append(InputFeatures(input_ids=padded_input_ids, attention_mask=padded_attention_mask, label=f.label))

all_input_ids = torch.tensor([f.input_ids for f in padded_features], dtype=torch.long)
all_attention_mask = torch.tensor([f.attention_mask for f in padded_features], dtype=torch.long)
all_labels = torch.tensor([f.label for f in padded_features], dtype=torch.long)

train_features, val_features = train_test_split(padded_features, test_size=0.2)

train_data = TensorDataset(torch.tensor([f.input_ids for f in train_features], dtype=torch.long),
    torch.tensor([f.attention_mask for f in train_features], dtype=torch.long),
    torch.tensor([f.label for f in train_features], dtype=torch.long))

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=16)

optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(3): 
    total_loss =   0
    model.train()
    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader)}")
def predict_resource_metric(question):
    model.eval()
    inputs = tokenizer(question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    _, predicted = torch.max(outputs.logits,   1)
    return predicted.item()

if __name__ == "__main__":
    question = "How many incidents have occurred in the past week?"
    predicted_label = predict_resource_metric(question)
    original_label = [label for label, idx in label_mapping.items() if idx == predicted_label][0]
    print(f"Predicted label: {original_label}")
