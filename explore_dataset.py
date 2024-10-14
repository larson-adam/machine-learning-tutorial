from datasets import load_dataset

# Load the IMDb dataset
dataset = load_dataset("imdb")

print(dataset['train'][0])