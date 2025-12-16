import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        'lm_adapter_lr': 1e-4,
        'he_lr': 3e-4,
        "model": {
            "name": "gpt2",
            "consolidation": {"mode": "prototype"},
            "router": {"mode": "learned"},
            "hippocampal_encoder": {"slot_dim": 256, "key_dim": 128, "input_dim": 768},
            "language_model": {"fusion_mode": "adapter", "input_dim": 768, "hidden_dim": 768, "slot_dim": 256},
            "retrieval_k": 5,
            "insertion_mode": "per-utterance"
        },
        "storage": {
            "episodic_store": {"eviction_policy": "fifo", "capacity": 100}
        }
    }

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Component Initialization
    base_model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
    lm = LanguageModelWithAdapter(base_model, **config['model']['language_model']).to(device)
    he = HippocampalEncoder(**config['model']['hippocampal_encoder']).to(device)
    router = Router(input_dim=config['model']['language_model']['input_dim']).to(device)
    es = EpisodicStore(slot_dim=config['model']['hippocampal_encoder']['slot_dim'], key_dim=config['model']['hippocampal_encoder']['key_dim'], **config['storage']['episodic_store']).to(device)
    kstore = KStore(key_dim=config['model']['hippocampal_encoder']['key_dim'], value_dim=config['model']['hippocampal_encoder']['slot_dim']).to(device)
    he_decoder = nn.Sequential(
        nn.Linear(he.slot_dim, he.input_dim),
        nn.ReLU()
    ).to(device)

    # Optimizers
    params = list(lm.parameters()) + list(he.parameters()) + list(router.parameters()) + list(he_decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    # Losses
    criterion_lm = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    criterion_router = nn.CrossEntropyLoss()

    # 3. Dummy Data
    utterance = torch.randint(0, 1000, (1, 10,)).to(device) # B, S

    # 4. Forward Pass
    print("\\n--- [Step 1] Running Forward Pass ---")
    optimizer.zero_grad()

    try:
        logits_pre, ctx_emb = lm(utterance)
        utterance_embedding = ctx_emb[:, -1, :]
        key, slot, _ = he.write(utterance_embedding)

        # Check for NaNs immediately after HE
        if torch.isnan(slot).any():
            raise RuntimeError("[FAIL] NaN detected in HE slot_vector!")
        print("[PASS] No NaNs in HE output.")

        es.add(key.unsqueeze(0), slot.unsqueeze(0))

        recon_emb = he_decoder(slot)
        pred_recon = recon_emb.unsqueeze(0)
        target_recon = utterance_embedding.detach()
        assert pred_recon.shape == target_recon.shape, f"Shape Mismatch: HE Recon {pred_recon.shape} vs {target_recon.shape}"
        loss_he_recon = criterion_mse(pred_recon, target_recon)
        print("[PASS] HE reconstruction loss calculated without shape errors.")

        # Simulate retrieval and fusion
        route_choice, route_probs = router.route(utterance_embedding)
        es_results = es.retrieve(key.unsqueeze(0), k=1)
        k_results = kstore.retrieve(key.unsqueeze(0), k=1)
        k_vals = k_results["slots"] if k_results["slots"] is not None else torch.zeros_like(es_results["slots"])
        
        p_es = route_probs[:, 0].view(-1, 1, 1)
        p_k = route_probs[:, 1].view(-1, 1, 1)
        memory_context_soft = p_es * es_results["slots"] + p_k * k_vals

        logits_fused, _ = lm(utterance, memory_context=memory_context_soft)
        
        shift_logits = logits_fused[..., :-1, :].contiguous()
        shift_labels = utterance[..., 1:].contiguous()
        loss_lm = criterion_lm(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        print("[PASS] LM loss calculated.")

        # Temporary diagnostic losses
        projection_target = nn.Linear(he.input_dim, he.slot_dim).to(device)
        optimizer.add_param_group({'params': projection_target.parameters()})
        he_pred = slot.unsqueeze(0)
        he_target = projection_target(utterance_embedding.detach())
        loss_he_direct = criterion_mse(he_pred, he_target)
        print("[PASS] HE direct diagnostic loss calculated.")

        # Oracle Label Generation
        with torch.no_grad():
            logits_es, _ = lm(utterance, memory_context=es_results["slots"])
            shift_logits_es = logits_es[..., :-1, :].contiguous()
            loss_es = criterion_lm(shift_logits_es.view(-1, shift_logits_es.size(-1)), shift_labels.view(-1))

            logits_ks, _ = lm(utterance, memory_context=k_vals)
            shift_logits_ks = logits_ks[..., :-1, :].contiguous()
            loss_ks = criterion_lm(shift_logits_ks.view(-1, shift_logits_ks.size(-1)), shift_labels.view(-1))
        
        oracle_label = torch.tensor([0 if loss_es < loss_ks else 1], device=device)
        loss_router = criterion_router(route_probs, oracle_label)
        lambda_router = 0.1

        total_loss = loss_lm + loss_he_recon + loss_he_direct + (lambda_router * loss_router)
        print("Forward pass successful.")
    except Exception as e:
        print(f"[FAIL] Exception during forward pass: {e}")
        return

    # 5. Backward Pass
    print("\\n--- [Step 2] Running Backward Pass ---")
    try:
        total_loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print(f"[FAIL] Exception during backward pass: {e}")
        return

    # 6. Gradient Checks
    print("\\n--- [Step 3] Checking Gradient Flow ---")
    he_grads = False
    for n, p in he.named_parameters():
        if p.grad is not None and torch.sum(torch.abs(p.grad)) > 0:
            he_grads = True
            break
    if he_grads:
        print("[PASS] HE has gradients: True")
    else:
        print("[FAIL] HE has gradients: False")

    router_grads = False
    for n, p in router.named_parameters():
        if p.grad is not None and torch.sum(torch.abs(p.grad)) > 0:
            router_grads = True
            break
    if router_grads:
        print("[PASS] Router has gradients: True")
    else:
        print("[FAIL] Router has gradients: False")

    # 7. Slot Norms Check
    print("\\n--- [Step 4] Checking Slot Norms ---")
    slots_in_es = es.slots_buffer[:es.size]
    if slots_in_es.numel() > 0:
        norms = torch.norm(slots_in_es, p=2, dim=-1)
        if len(slots_in_es) > 1:
            print(f"Slot Norms Mean: {norms.mean().item():.4f}, Std: {norms.std().item():.4f}")
        else:
            print(f"Slot Norms Mean: {norms.mean().item():.4f}, Std: n/a (single slot)")
        print("[PASS] Slot norms calculated.")
    else:
        print("[INFO] No slots in ES to check.")

    print("\\n--- Diagnostics Complete ---")


if __name__ == "__main__":
    run_diagnostics()