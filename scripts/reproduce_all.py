import os
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    ret = os.system(command)
    if ret != 0:
        print(f"Error executing: {command}")
        sys.exit(ret)

def main():
    print("Optimization: Starting DBME Claim-Driven Reproduction Suite")
    print("===========================================================")
    
    # 1. Data Generation
    print("\n[Step 1] Generating Synthetic Data...")
    run_command("python data/gen_retention_sessions.py")
    run_command("python data/gen_retention_queries.py")
    
    # 2. Baselines (C1)
    print("\n[Step 2] Running Baselines (C1)...")
    # KV Cache
    # We use a smaller subset or limit if needed for speed, but full run for paper
    run_command("python src/baselines/runner.py --baseline kv_cache --output results/baseline_kv.json")
    
    # Retrieval Baseline
    run_command("python src/baselines/runner.py --baseline retrieval --output results/baseline_retrieval.json")
    
    # Compressive (using Retrieval as proxy/same as baseline for now if code shared)
    run_command("python src/baselines/runner.py --baseline compressive --output results/baseline_compressive.json")

    # 3. DBME Retention (C1)
    print("\n[Step 3] Running DBME Retention (C1)...")
    run_command("python scripts/run_retention_dbme.py")
    
    # 4. C1 Evaluation
    print("\n[Step 4] Evaluating C1 (Retention)...")
    run_command("python eval/eval_retention.py")
    
    # 5. C2 Consolidation
    print("\n[Step 5] Running C2 Consolidation Test...")
    run_command("python scripts/run_consolidation_eval.py")
    
    # 6. C3 Forgetting
    print("\n[Step 6] Running C3 Forgetting Test...")
    run_command("python scripts/run_forgetting_eval.py")
    
    # 7. Final Summary
    print("\n[Step 7] Generating Final Summary...")
    run_command("python eval/eval_summary.py")
    
    print("\n===========================================================")
    print("Reproduction Complete. Check 'results/' and 'evaluation_results/'.")

if __name__ == "__main__":
    # Ensure we are in project root
    if not os.path.exists("src"):
        print("Please run this script from the project root directory.")
        sys.exit(1)
        
    os.makedirs("results", exist_ok=True)
    main()
