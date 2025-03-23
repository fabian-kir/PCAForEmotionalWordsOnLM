import torch
from math import dist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"  # or "mistralai/Mistral-7B-Instruct-v0.2" print("grabbing tokenizer")
cache_dir = "./model_cache"
snapshot_download(model_name, cache_dir=cache_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name)
print("grabbing model")
model = AutoModel.from_pretrained(cache_dir, device_map="auto", torch_dtype=torch.float16)
print("evaluating model")
model.eval()  # Disable dropout for consistency

# Extract embeddings (first "layer" - the input token embeddings)
print("grabbing embeddings")
# The correct way to access the embeddings in newer versions
embeddings = model.get_input_embeddings().weight.data.cpu().numpy()
print(embeddings)


def get_token_embedding(token_text):
    # Tokenize the input text (this returns a list of token IDs)
    token_ids = tokenizer.encode(token_text, add_special_tokens=False)

    # If the token gets split into multiple sub-tokens
    if len(token_ids) > 1:
        print(f"Note: '{token_text}' is tokenized into {len(token_ids)} sub-tokens:")
        for i, token_id in enumerate(token_ids):
            sub_token = tokenizer.decode([token_id])
            print(f"  Sub-token {i + 1}: '{sub_token}' (ID: {token_id})")
            print(f"  Embedding shape: {embeddings[token_id].shape}")
            print(f"  First few values: {embeddings[token_id][:5]}")

        # Return a list of embeddings for all sub-tokens
        return [embeddings[token_id] for token_id in token_ids]
    else:
        # If it's a single token, return its embedding
        token_id = token_ids[0]
        print(f"Token: '{token_text}' (ID: {token_id})")
        print(f"Embedding shape: {embeddings[token_id].shape}")
        print(f"First few values: {embeddings[token_id][:5]}")
        return embeddings[token_id]

def get_tokenizer_vocab():
    res = []
    for token, token_id in tokenizer.get_vocab().items():
        res.append((token, token_id))

    return res


def find_nearest_tokens(embedding_point, top_k=5):
    """
    Find the nearest tokens to a given point in the embedding space.

    Args:
        embedding_point: A numpy array of shape (768,) representing a point in the embedding space
        top_k: Number of nearest tokens to return (default: 5)

    Returns:
        List of tuples (token, distance) sorted by distance (closest first)
    """
    # Calculate distances between the point and all token embeddings
    distances = []
    vocab = tokenizer.get_vocab()

    for token, token_id in vocab.items():
        token_embedding = embeddings[token_id]
        # Calculate Euclidean distance
        distance = float(dist(embedding_point, token_embedding))
        distances.append((token, distance))

    # Sort by distance (closest first)
    distances.sort(key=lambda x: x[1])

    # Return top_k closest tokens
    return distances[:top_k]


def create_pca_graph(embeddings_list, words=None, title="PCA of Word Embeddings",
                     mode='2d', width=1000, height=800):
    """
    Create an interactive PCA visualization of word embeddings using Plotly.

    Args:
        embeddings_list: List of embedding vectors to project
        words: Optional list of words corresponding to the embeddings (for labeling points)
        title: Title for the plot
        mode: '2d' or '3d' to select visualization type
        width: Width of the figure in pixels
        height: Height of the figure in pixels

    Returns:
        Plotly figure object
    """
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    # Convert list of embeddings to a 2D numpy array
    embeddings_array = np.array(embeddings_list)

    # Apply PCA - use 3 components if in 3d mode, otherwise 2
    n_components = 3 if mode == '3d' else 2
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings_array)

    # Create a DataFrame for easier plotting with Plotly
    df = pd.DataFrame(reduced_embeddings, columns=[f'PC{i + 1}' for i in range(n_components)])

    # Add words to the DataFrame if provided
    if words:
        df['word'] = words
        hover_data = ['word']
    else:
        hover_data = None

    # Get explained variance information
    explained_var = pca.explained_variance_ratio_

    # Create the plot - either 2D or 3D
    if mode == '3d':
        fig = px.scatter_3d(
            df, x='PC1', y='PC2', z='PC3',
            hover_name='word' if words else None,
            labels={
                'PC1': f"PC1 ({explained_var[0]:.2%} variance)",
                'PC2': f"PC2 ({explained_var[1]:.2%} variance)",
                'PC3': f"PC3 ({explained_var[2]:.2%} variance)"
            },
            title=title,
            width=width, height=height,
        )

        # Add text labels in 3D space
        if words:
            fig.add_trace(
                go.Scatter3d(
                    x=df['PC1'], y=df['PC2'], z=df['PC3'],
                    mode='text',
                    text=words,
                    textposition='top center',
                    textfont=dict(color='black', size=10),
                    showlegend=False
                )
            )
    else:
        fig = px.scatter(
            df, x='PC1', y='PC2',
            hover_name='word' if words else None,
            labels={
                'PC1': f"PC1 ({explained_var[0]:.2%} variance)",
                'PC2': f"PC2 ({explained_var[1]:.2%} variance)"
            },
            title=title,
            width=width, height=height,
        )

        # Add text labels in 2D space
        if words:
            fig.add_trace(
                go.Scatter(
                    x=df['PC1'], y=df['PC2'],
                    mode='text',
                    text=words,
                    textposition='top center',
                    textfont=dict(color='black', size=10),
                    showlegend=False
                )
            )

    # Update layout for better interactivity
    fig.update_layout(
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def main():
    # Import necessary packages for Plotly
    import plotly.io as pio

    # Set default renderer for Plotly (use 'browser' to open in a web browser)
    pio.renderers.default = 'browser'  # or 'notebook' if using Jupyter

    with open("words_about_feelings.txt", "r") as file:
        feelings = [" " + line.lower().replace("\n", "") for line in file.readlines()]

    # Store the words and their embeddings
    words = []
    feelings_embedded = []

    for word in feelings:
        embedding = get_token_embedding(word)

        # Handle multi-token words (we'll use the average of sub-token embeddings)
        if isinstance(embedding, list):
            # For multi-token words, take the average of all token embeddings
            avg_embedding = np.mean(embedding, axis=0)
            feelings_embedded.append(avg_embedding)
            words.append(word.strip())
        else:
            feelings_embedded.append(embedding)
            words.append(word.strip())

    print(f"Created embeddings for {len(feelings_embedded)} words")

    # Create and display the interactive PCA visualization (2D)
    fig_2d = create_pca_graph(
        feelings_embedded,
        words,
        title="2D PCA of Emotional Words",
        mode='2d'
    )

    # Save as HTML for easy sharing
    fig_2d.write_html("emotional_words_pca_2d.html")

    # Display the visualization (this will open in a browser or notebook)
    fig_2d.show()

    # Create and display 3D visualization
    fig_3d = create_pca_graph(
        feelings_embedded,
        words,
        title="3D PCA of Emotional Words",
        mode='3d'
    )

    # Save as HTML for easy sharing
    fig_3d.write_html("emotional_words_pca_3d.html")

    # Display the 3D visualization
    fig_3d.show()

    # Optional: still print some information about similar words
    print("\nSome examples of similar emotional words:")
    for i, word in enumerate(words[:3]):  # Just show a few examples
        print(f"\nWords most similar to '{word}':")
        nearest = find_nearest_tokens(feelings_embedded[i], top_k=5)
        for nearby_word, distance in nearest:
            print(f"  {nearby_word}: {distance:.4f}")



if __name__ == "__main__":
    main()
