import argparse
import argparse
import sys
import os
import torch
import numpy as np
import pickle

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
        self.encoder = nn.Linear(1, 128) # Matching inspect_retrieval.py

    def forward(self, x):
        return x

def extract_and_save_memory_data(model_path, output_path):
    """
    Loads a trained model, extracts data from its memory stores, and saves it to a file.
    
    Args:
        model_path (str): Path to the saved model checkpoint.
        output_path (str): Path to save the extracted memory data.
    """
    print(f"Loading model from {model_path}...")
    model = DummyDBME()
    # In a real scenario, you would load a state dict. For this demo, we'll use a freshly initialized model.
    # If a model file exists, load it. Otherwise, the dummy model will be used as is.
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.eval()
    
    episodic_store = model.episodic_store
    k_store = model.k_store
    
    print("Extracting data from memory stores...")
    
    # EpisodicStore.store is a property that returns a list of dicts. We need to convert it.
    es_items = episodic_store.store
    es_data = {
        'slots': np.array([item['slot'] for item in es_items]) if es_items else np.array([]),
        'keys': np.array([item['key'] for item in es_items]) if es_items else np.array([]),
        'metadata': [item['meta'] for item in es_items] if es_items else [],
    }
    
    ks_data = k_store.export_all_data()
    # Ensure keys in ks_data are consistent, e.g., 'slots' instead of 'values'
    if 'values' in ks_data:
        ks_data['slots'] = ks_data.pop('values')

    memory_data = {
        'episodic_store': es_data,
        'k_store': ks_data,
    }
    
    print(f"Saving extracted memory data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(memory_data, f)
        
    print("Data extraction complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract memory data from a trained DBME model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the extracted data.")
    
    args = parser.parse_args()
    
    extract_and_save_memory_data(args.model_path, args.output_path)