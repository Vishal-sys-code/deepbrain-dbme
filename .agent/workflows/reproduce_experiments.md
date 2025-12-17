---
description: Verify DBME Experiments
---

To verify the DBME claims (C1-C3), run the following command:

```bash
# Runs the full reproduction suite
# 1. Generates data
# 2. Runs Baselines
# 3. Runs DBME
# 4. Generates Evaluations and Summaries
python scripts/reproduce_all.py
```

If you need to run specific steps individually:

```bash
# 1. Data Generation
python data/gen_retention_sessions.py
python data/gen_retention_queries.py

# 2. Baselines
python src/baselines/runner.py --baseline kv_cache --output results/baseline_kv.json
python src/baselines/runner.py --baseline retrieval --output results/baseline_retrieval.json

# 3. DBME Run
python scripts/run_retention_dbme.py

# 4. Evaluation
python eval/eval_retention.py
python scripts/run_consolidation_eval.py
python scripts/run_forgetting_eval.py
python eval/eval_summary.py
```

The plots will be saved to `evaluation_results/` and JSON metrics to `results/`.
