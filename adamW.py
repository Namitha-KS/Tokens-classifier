import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import InputExample, InputFeatures

categories = ["Incident", "Alert", "Resource", "Config Items", "Virtual Machines", "Virtual Networks", "Storage", "Disk", "Resource Group", "Projects", "Business Service", "Application"]

data = [
    {"question": "List last 5 events happened on 'onepane-vm'", "label": "Virtual Machines"},
    {"question": "What is the cost of running resources with open incidents?", "label": "Config Items"},
    {"question": "How many closed services currently has open incidents", "label": "Resource"},
    {"question": "What are the current open alerts?", "label": "Alert"},
    {"question": "How many incidents have occurred in the past week?", "label": "Incident"},
]

data_df = pd.DataFrame(data)

label_mapping = {label: idx for idx, label in enumerate(categories)}

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
    padded_inputs = tokenizer.pad(inputs, max_length=max_length, padding='max_length', return_tensors="pt")
    padded_input_ids = padded_inputs["input_ids"].squeeze()
    padded_attention_mask = padded_inputs["attention_mask"].squeeze()
    padded_features.append(InputFeatures(input_ids=padded_input_ids, attention_mask=padded_attention_mask, label=f.label))

all_input_ids = torch.stack([f.input_ids for f in padded_features], dim=0)
all_attention_mask = torch.stack([f.attention_mask for f in padded_features], dim=0)
all_labels = torch.tensor([f.label for f in padded_features], dtype=torch.long)

train_features, val_features = train_test_split(padded_features, test_size=0.2)

train_data = TensorDataset(torch.stack([f.input_ids for f in train_features], dim=0),
    torch.stack([f.attention_mask for f in train_features], dim=0),
    torch.tensor([f.label for f in train_features], dtype=torch.long))

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=16)

optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(5): 
    total_loss = 0
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
    _, predicted = torch.max(outputs.logits, 1)
    return categories[predicted.item()]

if __name__ == "__main__":
    questions = [
        "Are there any new alerts generated in the last 24 hours?",
        "What's the average response time for resolving alerts?",
        "How many incidents have occurred in the past week?",
        "Provide details on the last five incidents",
        "Which resource has the higher number of alerts ?",
        "List resources in 'Prod' environment",
        "Show cost usage across resource in 'Prod' environment",
        "List last 5 events happened on 'onepane-vm'",
        "List number of resources in each service",
        "which application has the highest resource consumption",
        "What are the current open alerts?",
        "What is the total cost of running the 'onepane' application",
        "For the past week, what's the distribution of alerts based on severity",
        "Which resources have not generated any incidents in the last month",
        "Which resource had the most degradation during last week",
        "Which resources have had changes happened to them in the last 24 hours",
        "List resources that have not experienced events in the last two weeks but have triggered incidents.",
        "What is the frequency of events triggered in each resource?",
        "How many resources are currently deployed in the 'Dev' and 'Staging' environments?",
        "What is the current health of our cloud resources?",
        "How many alerts were triggered in the last 24 hours?",
        "Can you identify the top three resource-intensive applications based on their monthly cloud costs?",
        "Provide a breakdown of the monthly cloud costs for each resource type in the 'Production' environment.",
        "Provide a summary of monthly cloud costs for all resources in the 'demo' environment.",
        "Can you compare the total monthly costs of 'onepane' and 'onepanesaas' in the 'prod' environment?",
        "Which resource has the highest monthly cost in the 'Production' environment?",
        "What is the overall health status of our cloud infrastructure today?",
        "Which services have exceeded their allotted budget",
        "what is the average mean time to recover among current open services",
        "What is the count of open incidents for resources owned by 'John Doe'?",
        "which resources are matched to an external platform",
        "What is the cost of running resources with open incidents?",
        "What are the incidents triggered in resources with criticality level of 'high'",
        "List number of resources in each open services",
        "What is the total cost of resources that were used by closed services?",
        "Which platforms have the highest distribution of incidents",
        "What is the average response time of incidents triggered in resources with 'high' criticality level?",
        "Which open services have a low priority",
        "Which platform has the most resources running in them.",
        "Compare the cost of running each service in the last month to the month before",
        "what is the priority of the worst-performing open services?",
        "How many closed services currently has open incidents",
        "Compare the average cost of running each resource to that of last month",
        "What are the changes that occurred today.",
    ]

    for question in questions:
        predicted_category = predict_resource_metric(question)
        print(f"Question: {question}\nPredicted category: {predicted_category}\n")
