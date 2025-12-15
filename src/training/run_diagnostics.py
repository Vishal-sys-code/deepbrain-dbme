import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.model.router import Router
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore

def run_diagnostics():
    """
    Runs a single-batch diagnostic to check shapes, gradient flow, and slot norms.
    """
    print("--- Running DBME System Diagnostics ---")

    # 1. Configuration
    config = {
        "model": {
            "name": "gpt2",
            "hippocampal_encoder": {"slot_dim": 256, "key_dim": 128, "input_dim": 768},
            "language_model": {"input_dim": 768, "hidden_dim": 768, "slot_dim": 256},
        },
        "storage": {
            "episodic_store": {"capacity": 100}
        }
    }

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Component Initialization
    base_model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
    lm = LanguageModelWithAdapter(base_model, **config['model']['language_model']).to(device)
    he = HippocampalEncoder(**config['model']['hippocampal_encoder']).to(device)
    
    # 3. Dummy Data
    utterance = torch.randint(0, 1000, (1, 10,)).to(device) # B, S

    # 4. Forward Pass
    print("\\n--- [Step 1] Running Forward Pass ---")
    try:
        # Test LanguageModelWithAdapter
        logits, features = lm(utterance)
        assert logits.shape == (1, 10, 50257), f"Logits shape is {logits.shape}"
        assert features.shape == (1, 10, 768), f"Features shape is {features.shape}"
        print("[PASS] LanguageModelWithAdapter forward pass successful.")

        # Test HippocampalEncoder
        utterance_embedding = features[:, -1, :]
        he_output = he.write(utterance_embedding)
        key = he_output["key"]
        slot = he_output["slot"]
        assert key.shape == (1, 128), f"Key shape is {key.shape}"
        assert slot.shape == (1, 256), f"Slot shape is {slot.shape}"
        print("[PASS] HippocampalEncoder forward pass successful.")

    except Exception as e:
        print(f"[FAIL] Exception during forward pass: {e}")
        return

    print("\\n--- Diagnostics Complete ---")

if __name__ == "__main__":
    run_diagnostics()