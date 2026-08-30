# Will the Judge Flip?

Code for **Will the Judge Flip? Predicting Position-Sensitive LLM Judgments from Residual Stream Activations**.

This anonymous research release studies whether a residual stream activation from one LLM-judge evaluation predicts whether the judge will reverse its preferred response after the two candidates exchange positions. The code evaluates Qwen3-1.7B, Qwen3-4B, Qwen3-8B, and Llama-3.1-8B-Instruct on JudgeBench and MT-Bench.

On the 534 JudgeBench pairs shared by all four judges, the linear probes achieve .621--.850 AUROC. Linear-probe AUROC exceeds the combined baseline, which uses confidence, verdict-label logits, response lengths, and the judge's `AB` choice, by .062--.113. Frozen JudgeBench-trained probes achieve .685--.853 AUROC on 1,802 shared MT-Bench comparisons without MT-Bench fitting or recalibration.

## Repository layout

| Path | Contents |
| --- | --- |
| `position_bias/` | Data preparation, judgment generation, feature extraction, probes, baselines, metrics, and controls |
| `jobs/` | Reproduction commands for each experiment stage |
| `configs/` | Pinned dataset and model revisions |
| `tests/` | Unit tests that do not load model weights |

Datasets, generated judgments, activation arrays, analysis outputs, downloaded model weights, and paper files are excluded from Git.

## Environment

The experiments use Python 3.10 and the exact package versions in `requirements.txt`. Create an isolated environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

Install `requirements-dev.txt` to run the release lint check:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m ruff check position_bias tests
python3 -m ruff format --check position_bias tests
```

The paper used deterministic bfloat16 inference on four 48-GB NVIDIA RTX A6000 GPUs. The four judges are independent and may instead be run sequentially on one suitable GPU. Model downloads and generated run directories require substantial storage.

Llama-3.1-8B-Instruct is gated on Hugging Face. Accept the model license and authenticate with Hugging Face before running the context audit or Llama inference.

## Judgment and feature protocol

Each comparison contains two fixed response identities, A and B. The prompt displays them as slots X and Y. The `AB` evaluation places A in X and B in Y; the `BA` evaluation reverses that assignment. The parser maps the X/Y verdict back to the fixed response identity before defining a position flip.

The judge must produce exactly three fields:

```text
REASONING: Response X answers the request more accurately.
CONFIDENCE: 82
VERDICT: X
```

Generation is greedy with a 256-token allowance, and Qwen3 thinking mode is disabled. Feature extraction replays the saved generation through the same token-by-token KV-cache path. It records the residual stream activation immediately before the verdict at four predeclared depths: 25%, 50%, 75%, and the penultimate layer. Only `AB` features enter the primary predictors; `BA` supplies the position-flip label.

## Reproduce JudgeBench results

Run every command from the repository root.

Download and normalize JudgeBench:

```bash
python3 -m position_bias.prepare_judgebench
```

Audit context length and confirm that X and Y are single verdict tokens for both tokenizer families:

```bash
bash jobs/check_context.sh
```

Run the four judges. The commands pin the exact checkpoint revisions used in the paper:

```bash
CUDA_VISIBLE_DEVICES=0 bash jobs/run_full_judge.sh judgebench Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e qwen3_1_7b
CUDA_VISIBLE_DEVICES=1 bash jobs/run_full_judge.sh judgebench Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c qwen3_4b
CUDA_VISIBLE_DEVICES=2 bash jobs/run_full_judge.sh judgebench Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218 qwen3_8b
CUDA_VISIBLE_DEVICES=3 bash jobs/run_full_judge.sh judgebench meta-llama/Meta-Llama-3.1-8B-Instruct 0e9e39f249a16976918f6564b8830bc894c89659 llama3_1_8b
```

Fit the nested linear probes and matched baselines, then compute the descriptive reference-agreement result:

```bash
bash jobs/run_analysis.sh
bash jobs/reference_agreement.sh
```

The analysis forms one common 534-pair set across judges. It uses five grouped outer folds and three grouped inner folds. Layer and L2 regularization selection use inner-fold Brier score. Feature standardization, fitting, and selection occur inside the relevant training folds.

## Reproduce MT-Bench transfer

Download and deduplicate the public annotations into 2,400 unique comparisons, then run the context audit:

```bash
bash jobs/prepare_mtbench.sh
```

Run the same four checkpoints on MT-Bench:

```bash
CUDA_VISIBLE_DEVICES=0 bash jobs/run_full_judge.sh mtbench Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e qwen3_1_7b
CUDA_VISIBLE_DEVICES=1 bash jobs/run_full_judge.sh mtbench Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c qwen3_4b
CUDA_VISIBLE_DEVICES=2 bash jobs/run_full_judge.sh mtbench Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218 qwen3_8b
CUDA_VISIBLE_DEVICES=3 bash jobs/run_full_judge.sh mtbench meta-llama/Meta-Llama-3.1-8B-Instruct 0e9e39f249a16976918f6564b8830bc894c89659 llama3_1_8b
```

Fit one final probe per judge on all JudgeBench examples and evaluate the frozen probes on the 1,802 MT-Bench comparisons shared by all four judges:

```bash
bash jobs/run_transfer.sh
```

MT-Bench data do not affect feature standardization, layer selection, regularization selection, fitting, thresholds, or calibration.

## Reproduce the repeated-presentation control

The control repeats the `AB` evaluation on a fixed 100-pair subset and compares the repeated verdict with the saved `AB` and `BA` verdicts:

```bash
bash jobs/prepare_same_order_control.sh
CUDA_VISIBLE_DEVICES=0 bash jobs/run_same_order_control.sh qwen3_1_7b
CUDA_VISIBLE_DEVICES=1 bash jobs/run_same_order_control.sh qwen3_4b
CUDA_VISIBLE_DEVICES=2 bash jobs/run_same_order_control.sh qwen3_8b
CUDA_VISIBLE_DEVICES=3 bash jobs/run_same_order_control.sh llama3_1_8b
bash jobs/summarize_same_order_control.sh
```

## Reproducibility safeguards

- Dataset and checkpoint revisions are pinned in `configs/`.
- The parser rejects malformed completions rather than guessing a verdict.
- Candidate identities are canonicalized only after parsing the displayed X/Y slot.
- Saved token sequences are replayed through the generation-time cache path before feature extraction.
- All judges and predictors use identical shared examples and outer folds on JudgeBench.
- Related variants of one question remain in the same fold.
- Feature standardization and model selection use training data only.
- Group-bootstrap resampling uses the underlying question as the unit.
- The transfer pipeline never fits or recalibrates on MT-Bench.

## Upstream data and model terms

- [JudgeBench](https://huggingface.co/datasets/ScalerLab/JudgeBench) is distributed under the MIT License.
- [MT-Bench human judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) are distributed under CC BY 4.0.
- [Qwen3](https://huggingface.co/Qwen/Qwen3-8B) is distributed under Apache 2.0.
- [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) is governed by the Llama 3.1 Community License and Acceptable Use Policy.

This repository does not redistribute model weights. Users are responsible for complying with the terms of every upstream dataset and checkpoint.
