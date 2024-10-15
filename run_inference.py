from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_path = "./fine-tuned-bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Put the model in evaluation mode
model.eval()

# Sample text for prediction (you can change this to any review text)
sample_text = "This movie was average. The acting and the plot was solid."

# Tokenize the text and prepare it for the model
inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Run the model to get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted class (0 = negative, 1 = positive)
predicted_class = torch.argmax(logits, dim=1).item()

# Output the result
if predicted_class == 1:
    print("Positive review")
else:
    print("Negative review")