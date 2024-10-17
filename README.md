# End-to-End Machine Learning Pipeline with Hugging Face and Web App

This project demonstrates how to fine-tune a pre-trained **BERT** model for sentiment analysis using the Hugging Face library and deploy it as a web app. The tutorial covers key machine learning concepts and also includes instructions on creating a **Vue.js** frontend that interacts with the Flask API.

## Project Overview

We are fine-tuning a **BERT** model, pre-trained by Google, to classify movie reviews from the **IMDb dataset** as positive or negative. After fine-tuning, we will create a simple API using Flask and a user interface with Vue.js. Additionally, we will return a **confidence score** for each prediction, indicating how certain the model is about the prediction.

## Installation and Setup

### 1. Clone the Repository

**SSH**
```bash
git clone git@github.com:larson-adam/machine-learning-tutorial.git
```

**HTTPS**
```bash
git clone https://github.com/larson-adam/machine-learning-tutorial.git
```

### 2. Set Up Python Virtual Environment

It’s recommended to use a virtual environment for Python dependencies:

**MacOS**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows**
```bash
python3.12 -m venv .venv
.venv\Scripts\activate
```

### 3. Install Python Dependencies

Make sure you have the required libraries for training and running the API. You can install them using pip:

```bash
pip install -r requirements.txt
```

### 4. Run the Fine-Tuning Script

Execute the fine_tune_bert.py script to fine-tune the BERT model on the IMDb dataset:

```bash
python3 fine_tune_bert.py
```

While this runs, you can understand more about the [input parameters here](https://github.com/larson-adam/machine-learning-tutorial/blob/main/dictionary.md#trainingargument-parameters-explained) and the [output values here](https://github.com/larson-adam/machine-learning-tutorial/blob/main/dictionary.md#trainingargument-parameters-explained)

> This will generate files like pytorch_model.bin, config.json, and tokenizer files in the `./fine-tuned-bert/` directory.

## Using the Fine-Tuned Model for Inference

Once the model is fine-tuned, we can use it to make predictions on new movie reviews.

### 1. Understanding the Fine-Tuning Script

You can use [`run_inference.py`](https://github.com/larson-adam/machine-learning-tutorial/blob/main/run_inference.py) to load the fine-tuned model and make predictions.

> I added in-line comments to help understand what is happening in this file. 

### 2. Run the Inference Script

```bash
python3 run_inference.py
```
> This will output whether the movie review is classified as positive or negative.

## Creating a Dockerized Flask API

Next, we will create a Flask API that accepts movie reviews via HTTP requests and returns predictions.

### The Flask App

The Flask App is already created in [`app.py`](https://github.com/larson-adam/machine-learning-tutorial/blob/main/app.py). 
This will serve up an API endpoint that we can hit. You can serve it locally, but we will Dockerize the Flask App in the next steps.

> Again, there are in-line comments to help understand what is happening.

### Dockerizing the Flask API

To make it easier to deploy the app, you can package it into a Docker container.

#### 1. Understanding Dockerfile

I have alrady the [Dockerfile](https://github.com/larson-adam/machine-learning-tutorial/blob/main/Dockerfile)

#### 2. Build the Dockerimage:

```bash
docker build -t bert-sentiment-app .
```

#### 3. Run the Docker Container with a Safe Port

```bash
docker run -p 8000:5000 bert-sentiment-app
```
> You can try using `5000:5000`, but I had to map port 8000 on my local machine to port 5000 inside the Docker container to bypass an error: `ERR_UNSAFE_PORT`

#### 4. Testing the Flask API

Before connecting the Vue.js frontend, lets make sure the Flask API endpoint is working in the docker container by sending a POST request to the `/predict` endpoint with `curl`. Try this command:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I really enjoyed this movie!"}'
```

You should get a response that looks like this:

```JSON
{
  "confidence": 0.9992263317108154,
  "prediction": "positive",
  "text": "I really enjoyed this movie!"
}
```

## Creating a Vue.js Frontend

To make the project more interactive, we’ll build a Vue.js frontend to interface with the Flask API.

### 1. Set Up a Vue.js Project

If you don’t have Vue installed, you can set up a new project using Vue CLI:

```bash
npm install -g @vue/cli
vue create ml-sentiment-app
```

Navigate into your project folder:

```bash
cd ml-sentiment-app
```

### 2. Modify the Vue Component

Open the `src/components/HelloWorld.vue` file and replace its contents with the following code to create the sentiment analysis form. This will display the resonse returned by the Flask API.

https://github.com/larson-adam/machine-learning-tutorial/blob/7b2b5b8f1cf4d58997b843583b4748c0488f2ac1/sentiment-app/src/components/HelloWorld.vue#L1-L58

### 3. Run the Vue.js App

Now, run the Vue app:

```bash
npm run serve
```

> You can access the app in your browser at http://localhost:8080. The Vue.js frontend will now display the sentiment prediction and the confidence score.

The final result should look something like this:

![alt text](https://github.com/larson-adam/machine-learning-tutorial/blob/main/Movie%20Review%20Screenshot.png)

## Conclusion

In this project, we’ve fine-tuned a pre-trained BERT model using Hugging Face, built a Flask API to serve predictions, and created a Vue.js frontend for user interaction. The Flask API now includes a confidence score for each prediction, which is displayed in the Vue.js frontend. This setup provides a complete end-to-end machine learning pipeline, from training to deployment.

You can further improve this project by adding more styling, deploying it to a cloud platform, or optimizing the inference process.

## Dictionary and Context

I've compiled a list of helpful reference to explain some of the terminology you'll come across in this tutorial.

[dictionary.md](https://github.com/larson-adam/machine-learning-tutorial/blob/main/dictionary.md)

## Fine-Tuning In-Depth

For a more indepth explanation of the fine-tunning process checkout the explanationl:

[fine-tuning.md](https://github.com/larson-adam/machine-learning-tutorial/blob/main/fine-tuning.md) 