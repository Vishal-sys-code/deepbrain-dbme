import os
import pickle
import sys
import numpy as np
import subprocess

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from src.visualization.extract_memory_data import DummyDBME

def generate_synthetic_model_and_data(model_path, data_path):
    """Generates a dummy model with synthetic data and saves them."""
    print("Generating synthetic model and data...")
    
    model = DummyDBME()
    
    # --- Populate Episodic Store ---
    fact_center = torch.randn(1, 128)
    paris_facts_data = [
        {'id': 101, 'timestamp': 10, 'text': 'Paris is in France'},
        {'id': 102, 'timestamp': 12, 'text': 'Eiffel Tower is in Paris'},
        {'id': 103, 'timestamp': 15, 'text': 'Louvre is in Paris'},
    ]
    paris_slots = torch.cat([fact_center + torch.randn(1, 128) * 0.1 for _ in paris_facts_data])
    for i, meta in enumerate(paris_facts_data):
        model.episodic_store.add(paris_slots[i], paris_slots[i], meta=meta)

    stale_fact_slot = torch.randn(1, 128) + 1.5
    model.episodic_store.add(stale_fact_slot, stale_fact_slot, meta={'id': 201, 'timestamp': 2, 'text': 'Stale fact'})

    for i in range(20):
        other_slot = torch.randn(1, 128)
        model.episodic_store.add(other_slot, other_slot, meta={'id': 300+i, 'timestamp': 20+i})

    # --- Populate K-Store ---
    paris_prototype = torch.mean(paris_slots, dim=0, keepdim=True)
    model.k_store.add(paris_prototype, paris_prototype, meta={
        'consolidation_timestamp': 25, 'source_ids': [101, 102, 103]
    })
    
    other_prototype_src_indices = [300, 301, 302, 303, 304]
    other_prototype_slots = [s['slot'] for s in model.episodic_store.store if s['meta'].get('id') in other_prototype_src_indices]
    if other_prototype_slots:
        # Convert numpy arrays to tensors before stacking
        other_prototype_tensors = [torch.from_numpy(slot) for slot in other_prototype_slots]
        other_prototype = torch.mean(torch.stack(other_prototype_tensors), dim=0, keepdim=True)
        model.k_store.add(other_prototype, other_prototype, meta={
            'consolidation_timestamp': 30, 'source_ids': other_prototype_src_indices
        })

    # Save the model state
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Synthetic model saved to {model_path}")
    
    # Now, extract the data from the model to a pickle file for the visualization scripts
    # This reuses the logic we just made functional
    subprocess.run([
        sys.executable, "src/visualization/extract_memory_data.py",
        "--model_path", model_path,
        "--output_path", data_path
    ], check=True)
    print(f"Extracted data saved to {data_path}")


def run_visualizations(model_path, data_path, output_dir):
    """Runs all the visualization scripts on the generated data."""
    print("\nRunning visualization scripts...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot slot clusters
    subprocess.run([
        sys.executable, "src/visualization/plot_slot_clusters.py",
        "--data_path", data_path,
        "--output_path", os.path.join(output_dir, "slot_clusters.png")
    ], check=True)
    
    # 2. Plot memory timeline
    subprocess.run([
        sys.executable, "src/visualization/plot_memory_timeline.py",
        "--data_path", data_path,
        "--output_path", os.path.join(output_dir, "memory_timeline.png")
    ], check=True)

    # 3. Inspect retrieval for "Paris"
    print("\n--- Running Retrieval Inspection ---")
    subprocess.run([
        sys.executable, "src/visualization/inspect_retrieval.py",
        "--model_path", model_path,
        "--query", "What is in Paris?",
    ], check=True)
    print("--- End Retrieval Inspection ---\n")

    # 4. Trace consolidation for the "Paris" prototype (ID 0)
    subprocess.run([
        sys.executable, "src/visualization/trace_consolidation.py",
        "--data_path", data_path,
        "--output_path", os.path.join(output_dir, "consolidation_trace_paris.png"),
        "--prototype_id", "0"
    ], check=True)
    
    print("All visualizations generated successfully.")


if __name__ == "__main__":
    MODEL_PATH = "demos/models/synthetic_model.pt"
    DATA_PATH = "demos/data/synthetic_memory.pkl"
    OUTPUT_DIR = "demos/outputs"
    
    generate_synthetic_model_and_data(MODEL_PATH, DATA_PATH)
    run_visualizations(MODEL_PATH, DATA_PATH, OUTPUT_DIR)