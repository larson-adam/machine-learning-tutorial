from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)

# Load fine-tuned model and tokenizer
model_path = "./fine-tuned-bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data['text']

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Return the predicted label as a response
    return jsonify({
        "text": text,
        "prediction": "positive" if predicted_class == 1 else "negative"
    })

if __name__ == "__main__":
    app.run(debug=True)