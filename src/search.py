import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess_text

class LyricSearchEngine:
    def __init__(self, df, embeddings):
        self.df = df
        self.embeddings = embeddings

    def search(self, lyric_snippet: str, top_k: int = 5):
        # 1. Preprocess user input
        clean_snippet = preprocess_text(lyric_snippet)

        # 2. Use the model (PyTorch based) to encode the snippet
        # This assumes you pass the model to the search function or have it available 
        from model import LyricEmbeddingModel
        model_wrapper = LyricEmbeddingModel() 
        snippet_embedding = model_wrapper.model.encode([clean_snippet])

        # 3. Calculate Cosine Similarity
        # This is the "Similarity Model" part of your task
        similarities = cosine_similarity(snippet_embedding, self.embeddings)[0]

        # 4. Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        candidates = self.df.iloc[top_indices].copy()
        candidates["score"] = similarities[top_indices]

        return candidates[["track_name", "artist_name", "score"]]
    