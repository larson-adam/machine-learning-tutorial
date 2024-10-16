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