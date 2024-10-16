from datasets import load_dataset

# Load the IMDb dataset
dataset = load_dataset('imdb')

# Function to find the first positive and negative review
def find_first_positive_negative(dataset):
    positive_review = None
    negative_review = None
    
    for example in dataset['train']:
        if example['label'] == 1 and positive_review is None:
            positive_review = example['text']
            print(f"First Positive Review: \n{positive_review}\n")
        
        if example['label'] == 0 and negative_review is None:
            negative_review = example['text']
            print(f"First Negative Review: \n{negative_review}\n")
        
        # If both positive and negative reviews are found, break the loop
        if positive_review and negative_review:
            break

# Find the first positive and negative reviews
find_first_positive_negative(dataset)