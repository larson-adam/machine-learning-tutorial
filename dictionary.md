## Dictionary

### General

- **Model**: A machine learning program that makes predictions based on input data. BERT is a powerful model for understanding text, and it has been pre-trained on vast amounts of data.
- **Tokenizer**: A tool that breaks down text into smaller units (tokens) so the model can understand and process it. BERT works with tokens, not raw text.
- **Tokenization**: The process of converting raw text into tokens (words, subwords) that the model can process.
- **Dataset**: A collection of data used for training or testing a machine learning model. In this case, the dataset consists of movie reviews and their associated sentiment labels (positive/negative).
- **Padding and Truncation**: Ensures that all inputs to the model are of the same length by either adding extra tokens (padding) or cutting off long texts (truncation).
- **Trainer**: A class provided by Hugging Face that abstracts much of the training process, including loss calculation, backpropagation, and evaluation.
- **Accelerate**: A Hugging Face library that optimizes model training by distributing the workload across multiple hardware devices.
- **PyTorch**: A deep learning framework that provides dynamic computation, allowing for more flexibility during training.

### TrainingArgument Parameters Explained:

- **output_dir**: Where model checkpoints will be saved.
- **evaluation_strategy**: When to run evaluation (in this case, after every epoch).
- **per_device_train_batch_size**: The number of examples the model processes before updating the weights.
- **per_device_eval_batch_size**: Same as above, but for evaluation.
- **num_train_epochs**: How many times the model will pass through the entire dataset during training.
- **weight_decay**: A regularization technique that helps prevent overfitting by penalizing large model weights.
- **logging_dir**: Directory to save logs for tracking progress.
- **logging_steps**: The frequency of logging information about the training process (e.g., every 500 steps).

### Training Process Output Explained:

#### Examample Output:
```json
{'loss': 0.4277, 'grad_norm': 4.493, 'learning_rate': 4.73e-05, 'epoch': 0.16}
```

- **Loss**: A measure of how far the model’s predictions are from the actual labels. The lower the loss, the better the model is performing.
- **Grad Norm**: Represents the magnitude of gradient updates. Larger gradients suggest more significant changes to the model’s weights, while smaller gradients indicate finer adjustments.
- **Learning Rate**: Controls how much to change the model weights with each update. Smaller learning rates mean slower, more gradual learning.
- **Epoch**: Shows the progress through the dataset. An epoch is one complete pass through the training data.

### Some Fine-Tune Generated Files:

When you fine-tuned the model and saved it, several important files were created in the `./fine-tuned-bert/` directory:

- **config.json**: Contains the configuration of the model (e.g., model architecture, hyperparameters). This is needed when loading the model for inference or further fine-tuning.
- **pytorch_model.bin**: This file contains the actual learned weights of the fine-tuned model. It’s the file that allows the model to make predictions based on what it learned during training.
- **tokenizer_config.json**: Configuration for the tokenizer, which ensures that the input text is processed in the same way it was during training.
- **vocab.txt** (or similar): This is the vocabulary file used by the tokenizer. It maps words to their corresponding token IDs.