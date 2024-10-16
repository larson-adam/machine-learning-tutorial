from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)

# Load fine-tuned model and tokenizer
model_path = "./fine-tuned-bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data['text']

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get model predictions (logits)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = F.softmax(logits, dim=1)

    # Get the predicted class (0 = negative, 1 = positive)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Get the confidence score for the predicted class
    confidence = probabilities[0][predicted_class].item()

    # Return the predicted class and confidence score
    return jsonify({
        "text": text,
        "prediction": "positive" if predicted_class == 1 else "negative",
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')