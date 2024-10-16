from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

# Load tokenizer and pre-trained BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Preprocess the dataset (tokenization)
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create the Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine-tuned-bert")
tokenizer.save_pretrained("./fine-tuned-bert")