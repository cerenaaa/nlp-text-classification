"""Synthetic multi-label text classification corpus."""
import random
import numpy as np

LABELS = ["billing", "technical_support", "account_management", "feature_request",
          "bug_report", "cancellation", "upgrade", "security"]

TEMPLATES = {
    "billing": ["My invoice shows an incorrect charge of {amount}",
                "I was double charged for {product} this month",
                "Please explain the {fee} fee on my account"],
    "technical_support": ["The app crashes when I try to {action}",
                           "I cannot connect to {service} since the update",
                           "Getting error code {code} on {platform}"],
    "account_management": ["I need to update my email address to {email}",
                            "Please change the admin user for account {id}",
                            "How do I add a new user to my team?"],
    "feature_request": ["It would be great if the dashboard showed {metric}",
                         "Can you add support for {integration} integration?",
                         "Please add bulk export functionality"],
    "bug_report": ["Found a bug where {feature} does not save changes",
                    "The {page} page returns a 500 error intermittently",
                    "Data is disappearing after {action}"],
    "cancellation": ["I want to cancel my subscription effective {date}",
                      "Please close my account and delete my data",
                      "I am switching to a competitor, please cancel"],
}

FILLERS = dict(amount=["$29.99","$150","$500"], product=["Pro Plan","Add-on","API"],
               fee=["processing","platform","overage"], action=["export","login","save"],
               service=["API","dashboard","email"], code=["E401","500","NaN"],
               platform=["iOS","Chrome","Windows"], email=["new@co.com","admin@firm.io"],
               id=["ACC-1234","ORG-9988"], metric=["ARR","churn","NPS"],
               integration=["Salesforce","Slack","Zapier"], feature=["draft","filter","tag"],
               page=["billing","settings","reports"], date=["end of month","immediately"])


def generate_corpus(n: int = 2000, multi_label_rate: float = 0.2, seed: int = 42) -> list[dict]:
    random.seed(seed)
    samples = []
    label_list = list(TEMPLATES.keys())
    for _ in range(n):
        primary = random.choice(label_list)
        labels = [primary]
        if random.random() < multi_label_rate:
            labels.append(random.choice([l for l in label_list if l != primary]))
        template = random.choice(TEMPLATES[primary])
        for k, v in FILLERS.items():
            template = template.replace("{" + k + "}", random.choice(v))
        samples.append({"text": template, "labels": labels})
    print(f"Generated {n} samples | {sum(len(s['labels'])>1 for s in samples)} multi-label")
    return samples


def train_test_split(samples: list[dict], test_size: float = 0.2, seed: int = 42):
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    n_test = int(len(shuffled) * test_size)
    return shuffled[n_test:], shuffled[:n_test]
