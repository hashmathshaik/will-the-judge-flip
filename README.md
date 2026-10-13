# Beyond Stated Confidence: Predicting Order Instability in LLM Judges

## Abstract

Large language model (LLM) judges can reverse confident verdicts when the same responses exchange positions. We use the first judgment's internal state to predict these reversals and recheck only comparisons with the highest predicted risk. Across eight judges, probes achieved higher held-out areas under the receiver operating characteristic curve (AUROCs) than all ten baseline point estimates. Probe AUROCs ranged from 0.642–0.917 on JudgeBench and 0.721–0.847 on LLMBar. For Qwen3-8B, checking half of JudgeBench pairs detected 90.0% of preference reversals and saved 25% of judgment calls. The predictor transferred to MT-Bench without retraining, achieving 0.853 AUROC versus 0.797 for the strongest transferred output baseline. The same 50% checking budget detected 92.0% of reversals. These results support selective checking for order instability, without establishing judgment correctness.

## Run experiments

Use Python 3.11 and NVIDIA GPUs. Each checkpoint runs on one GPU; choose devices with sufficient memory. The Llama checkpoint requires authorized Hugging Face access.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash jobs/run_all.sh 0 1 2 3
```

Run stages separately:

```bash
bash jobs/prepare_data.sh
python -m position_bias.run_suite --benchmark judgebench --gpus 0 1 2 3
python -m position_bias.run_suite --benchmark llmbar --gpus 0 1 2 3
python -m position_bias.run_suite --benchmark mtbench --gpus 0 1 2 3
bash jobs/run_analysis.sh judgebench
bash jobs/run_analysis.sh llmbar
bash jobs/run_transfer.sh
bash jobs/run_controls.sh
```

Check a small cohort:

```bash
bash jobs/run_pilot.sh runs/pilot cuda qwen3_1_7b
```

Checkpoint and analysis settings are in `configs/`. Outputs are written to `runs/`.
