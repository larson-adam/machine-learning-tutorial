# End-to-End Machine Learning Pipeline with Hugging Face and Web App

This project demonstrates how to fine-tune a pre-trained **BERT** model for sentiment analysis using the Hugging Face library and deploy it as a web app. The tutorial covers key machine learning concepts and also includes instructions on creating a **Vue.js** frontend that interacts with the Flask API.

## Project Overview

We are fine-tuning a **BERT** model, pre-trained by Google, to classify movie reviews from the **IMDb dataset** as positive or negative. After fine-tuning, we will create a simple API using Flask and a user interface with Vue.js. Additionally, we will return a **confidence score** for each prediction, indicating how certain the model is about the prediction.

## Installation and Setup

### 1. Clone the Repository
```bash
git clone <repository_url>
```

### 2. Install Python Dependencies

Make sure you have the required libraries for training and running the API. You can install them using pip:

```bash
pip install -r requirements.txt
```

### 3. Set Up Python Virtual Environment (Optional)

It’s recommended to use a virtual environment for Python dependencies:

**MacOS**
```bash
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
```

**Windows**
```bash
python3 -m venv venv
venv\Scripts\activate
```

### 4. Run the Fine-Tuning Script

Execute the fine_tune_bert.py script to fine-tune the BERT model on the IMDb dataset:

```bash
python3 fine_tune_bert.py
```
> This will generate files like pytorch_model.bin, config.json, and tokenizer files in the `./fine-tuned-bert/` directory.

## Using the Fine-Tuned Model for Inference

Once the model is fine-tuned, we can use it to make predictions on new movie reviews.

### 1. Create a New Python File (run_inference.py)

Here’s how you can load the fine-tuned model and make predictions:

```python
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
```

### 2. Run the Inference Script

```bash
python3 run_inference.py
```
> This will output whether the movie review is classified as positive or negative.

## Create Flask API

Next, we will create a Flask API that accepts movie reviews via HTTP requests and returns predictions.

### Create a Flask App (app.py)

```python
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
```

### Dockerize Flask API

To make it easier to deploy the app, you can package it into a Docker container.

1. Create a Dockerfile

```Dockerfile
# Use the official Python image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the app files to the container
COPY . /app

# Install required Python packages
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
```

> The API will run on http://127.0.0.1:5000. You can test the API by sending a POST request with some text to the /predict endpoint.

2. Build the Dockerimage:

```bash
docker build -t bert-sentiment-app .
```

3. Run the Docker Container with a Safe Port

```bash
docker run -p 8000:5000 bert-sentiment-app
```
> You can try using `5000:5000`, but I had to map port 8000 on my local machine to port 5000 inside the Docker container to bypass an error: `ERR_UNSAFE_PORT`

### Creating a Vue.js Frontend

To make the project more interactive, we’ll build a Vue.js frontend to interface with the Flask API.

1. Set Up a Vue.js Project

If you don’t have Vue installed, you can set up a new project using Vue CLI:

```bash
npm install -g @vue/cli
vue create sentiment-app
```

Navigate into your project folder:

```bash
cd sentiment-app
```

2. Modify the Vue Component

Open the `src/components/HelloWorld.vue` file and replace its contents with the following code to create the sentiment analysis form. This will now display the confidence score returned by the Flask API.

```html
<template>
  <div class="sentiment-app">
    <h1>Movie Review Sentiment Analysis</h1>
    <textarea v-model="reviewText" placeholder="Enter your movie review here"></textarea>
    <button @click="analyzeSentiment">Analyze Sentiment</button>
    <div v-if="prediction">
      <h2>Prediction: {{ prediction }}</h2>
      <p>Confidence: {{ confidence }}</p>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      reviewText: '',
      prediction: null,
      confidence: null,
    };
  },
  methods: {
    async analyzeSentiment() {
      try {
        const response = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: this.reviewText }),
        });

        const data = await response.json();
        this.prediction = data.prediction;
        this.confidence = data.confidence;
      } catch (error) {
        console.error('Error fetching prediction:', error);
      }
    },
  },
};
</script>

<style scoped>
.sentiment-app {
  max-width: 600px;
  margin: 0 auto;
  padding: 20px;
}
textarea {
  width: 100%;
  height: 150px;
  margin-bottom: 10px;
}
button {
  padding: 10px 20px;
}
</style>
```

3. Run the Vue.js App

Now, run the Vue app:

```bash
npm run serve
```

You can access the app in your browser at http://localhost:8080. The Vue.js frontend will now display the sentiment prediction and the confidence score.

## Conclusion

In this project, we’ve fine-tuned a pre-trained BERT model using Hugging Face, built a Flask API to serve predictions, and created a Vue.js frontend for user interaction. The Flask API now includes a confidence score for each prediction, which is displayed in the Vue.js frontend. This setup provides a complete end-to-end machine learning pipeline, from training to deployment.

You can further improve this project by adding more styling, deploying it to a cloud platform, or optimizing the inference process.