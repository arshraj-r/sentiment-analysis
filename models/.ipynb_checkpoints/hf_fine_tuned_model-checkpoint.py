import torch
from transformers import pipeline
print(torch.cuda.is_available())  
device = 0 if torch.cuda.is_available() else -1

model_name="cardiffnlp/twitter-roberta-base-sentiment" #approx 500mb size
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, device=device)

text = [
    "I love this movie!",
    "This was a terrible experience.",
    "It's okay, not great but not bad either."
]

# Run prediction
results = sentiment_pipeline(text)

