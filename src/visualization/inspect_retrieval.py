import argparse
import sys
import os
import pickle
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch.nn as nn
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore

# Define a dummy model for demonstration purposes
class DummyDBME(nn.Module):
    def __init__(self):
        super().__init__()
        self.episodic_store = EpisodicStore(key_dim=128, slot_dim=128, capacity=100)
        self.k_store = KStore(key_dim=128, value_dim=128, capacity=20)
        # A dummy encoder to simulate turning text into a vector
        self.encoder = nn.Linear(1, 128) # Simplistic placeholder

    def encode_query(self, query_text):
        # In a real model, this would involve tokenization and a transformer encoder.
        # Here, we simulate it by hashing the text to create a pseudo-random but deterministic vector.
        query_hash = hash(query_text)
        seed = torch.tensor([query_hash % 1e9], dtype=torch.float32)
        with torch.no_grad():
            vector = self.encoder(seed.unsqueeze(0))
        return torch.nn.functional.normalize(vector, p=2, dim=-1)

    def forward(self, x):
        return x

def inspect_retrieval(model_path, query_text, top_k=5):
    """
    Loads a model, encodes a query, and inspects the retrieval results.
    
    Args:
        model_path (str): Path to the model checkpoint.
        query_text (str): The text of the query.
        top_k (int): The number of results to retrieve.
    """
    print(f"Loading model from {model_path}...")
    model = DummyDBME()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.eval()

    print(f"Encoding query: '{query_text}'")
    query_vector = model.encode_query(query_text)

    # Retrieve from Episodic Store
    es_results = model.episodic_store.retrieve(query_vector, k=top_k)
    print("\n--- Episodic Store Retrieval Results ---")
    if es_results and es_results.get('ids'):
        for i, (score, meta) in enumerate(zip(es_results['scores'][0], es_results['meta'][0])):
            print(f"  Rank {i+1}: Score: {score:.4f}, Metadata: {meta}")
    else:
        print("  No results found.")

    # Retrieve from K-Store
    ks_results = model.k_store.retrieve(query_vector, k=top_k)
    print("\n--- K-Store Retrieval Results ---")
    if ks_results and ks_results.get('ids'):
        for i, (score, meta) in enumerate(zip(ks_results['scores'][0], ks_results['meta'][0])):
            print(f"  Rank {i+1}: Score: {score:.4f}, Metadata: {meta}")
    else:
        print("  No results found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect memory retrieval for a given query.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--query", type=str, required=True, help="The query to inspect.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to retrieve.")
    
    args = parser.parse_args()
    
    inspect_retrieval(args.model_path, args.query, args.top_k)