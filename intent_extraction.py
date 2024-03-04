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

def extract_intent(question):
    doc = nlp(question)

    intent_keywords = {
        "alerts_new": ["new alerts", "generated", "last 24 hours"],
        "response_time": ["average response time", "resolving alerts"],
        "incidents_count": ["how many incidents", "occurred", "past week"],
        "incidents_details": ["provide details", "last five incidents"],
        "resource_alerts_count": ["higher number of alerts", "resource"],
        "resources_list_env": ["list resources", "'prod' environment"],
        "cost_usage": ["show cost usage", "'prod' environment"],
        "events_last_5": ["list last 5 events", "'onepane-vm'"],
        "resources_count_service": ["number of resources", "each service"],
        "app_highest_consumption": ["application", "highest resource consumption"],
        "open_alerts": ["current open alerts"],
        "total_cost": ["total cost", "running", "'onepane' application"],
        "alerts_distribution": ["past week", "distribution of alerts", "based on severity"]
    }

    for intent, keywords in intent_keywords.items():
        for keyword in keywords:
            if keyword in question:
                return intent

    return "other"

# List of questions
questions = [
    "Are there any new alerts generated in the last 24 hours?",
    "What's the average response time for resolving alerts?",
    "How many incidents have occurred in the past week?",
    "Provide details on the last five incidents",
    "Which resource has the higher number of alerts?",
    "List resources in 'Prod' environment",
    "Show cost usage across resource in 'Prod' environment",
    "List last 5 events happened on 'onepane-vm'",
    "List number of resources in each service",
    "Which application has the highest resource consumption?",
    "What are the current open alerts?",
    "What is the total cost of running the 'onepane' application",
    "For the past week, what's the distribution of alerts based on severity"
]

# Process each question
for question in questions:
    intent = extract_intent(question)
    resource, metric = extract_resource_metric(question)
    print(f"Question: {question}\nIntent: {intent}\nResource: {resource}\nMetric: {metric}\n")
