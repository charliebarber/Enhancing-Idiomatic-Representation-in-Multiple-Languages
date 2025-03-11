# First cell - Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# Second cell - Load the model
model_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # Update with your path
model = SentenceTransformer(model_path)

# Third cell - Define example sentences
idiom_examples = [
    # Format: (sentence with idiom, correct paraphrase, literal interpretation)
    ("She kicked the bucket last year.", "She died last year.", "She kicked a bucket last year."),
    ("It's raining cats and dogs.", "It's raining heavily.", "Cats and dogs are falling from the sky."),
    # Add more examples here
]

# Fourth cell - Calculate and visualize similarities
for idiom, paraphrase, literal in idiom_examples:
    sentences = [idiom, paraphrase, literal]
    embeddings = model.encode(sentences)
    similarities = cosine_similarity(embeddings)
    
    print(f"Idiom: {idiom}")
    print(f"Correct paraphrase: {paraphrase}")
    print(f"Literal interpretation: {literal}")
    print("\nSimilarity matrix:")
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            print(f"{i+1} and {j+1}: {similarities[i][j]:.4f}")
    print("-" * 80)
    
    # Visualize with a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarities, annot=True, cmap="YlGnBu", 
                xticklabels=["Idiom", "Paraphrase", "Literal"], 
                yticklabels=["Idiom", "Paraphrase", "Literal"])
    plt.title(f"Similarity Matrix for: '{idiom}'")
    plt.savefig(f"{idiom}.png")