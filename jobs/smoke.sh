#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${TMPDIR:-/tmp}/judge-smoke-matplotlib"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/judge-smoke-cache"
python3 - <<'PY'
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from position_bias.artifacts import load_model
from position_bias.audit import check_budget, rank_pairs
from position_bias.features import text_hash
from position_bias.io_utils import write_json, write_jsonl
from position_bias.modeling import Runtime, generate_judgment, teacher_forced_extract
from position_bias.protocol import PROTOCOL_VERSION, SYSTEM_PROMPT, USER_TEMPLATE, canonicalize, candidate_layers
from position_bias.run_analysis import analyze
from position_bias.run_transfer import transfer

torch.manual_seed(42)
tiny = LlamaForCausalLM(LlamaConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
             num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
             max_position_embeddings=64, bos_token_id=0, eos_token_id=1, pad_token_id=0)).eval()
with torch.inference_mode():
    states = tiny.model(input_ids=torch.tensor([[5,4]])).last_hidden_state[0]
    tiny.lm_head.weight.zero_()
    tiny.lm_head.weight[4] = torch.linalg.lstsq(states, torch.tensor([10.,-10.])).solution
    tiny.lm_head.weight[2] = torch.linalg.lstsq(states, torch.tensor([-10.,10.])).solution
class Tokenizer:
    pad_token_id, eos_token_id = 0, 1
    def apply_chat_template(self, *args, **kwargs): return [5]
    def encode(self, text, **kwargs): return [2 if text == ' A' else 3]
    def decode(self, ids, **kwargs):
        pieces = {4:'A has a clearer answer.\nConfidence: 80\nVerdict:',2:' A',3:' B',5:'prompt'}
        return ''.join(pieces.get(int(i),'') for i in ids)
runtime = Runtime(tiny, Tokenizer(), torch.device('cpu'), torch.float32, candidate_layers(4), {'A':2,'B':3})
pair = {'pair_id':'tiny','question':'Q','response_a':'A','response_b':'B'}
judgment = generate_judgment(runtime,pair,'AB',2,64,capture_features=True)
assert judgment.parsed.format_ok and judgment.parsed.slot == 'A'
direct = teacher_forced_extract(runtime,judgment)
judgment.extracted_features = None
replayed = teacher_forced_extract(runtime,judgment)
np.testing.assert_allclose(direct.hidden_states,replayed.hidden_states,atol=1e-5)
assert canonicalize('A','AB') != canonicalize('A','BA')
assert canonicalize('A','AB') == canonicalize('B','BA')
assert check_budget(np.array([1,0]),np.array([.9,.1]),['a','b'],.5)['call_savings_fraction'] == .25
assert [ ['a','b'][i] for i in rank_pairs(np.array([.5,.5]),['a','b']) ] == [ ['b','a'][i] for i in rank_pairs(np.array([.5,.5]),['b','a']) ]

with tempfile.TemporaryDirectory(prefix='judge-smoke-') as temp:
    root = Path(temp)
    rng = np.random.default_rng(42)
    def materialize(name, count):
        directory = root/name
        records, metadata, activations = [], [], []
        for i in range(count):
            y = i%2
            p = {'format_ok':True,'confidence_ok':True,'verdict_reached':True,'slot':'A',
                 'confidence':70+i%25,'reasoning':f'Candidate compares request group {i//2} with careful explanation'}
            records.append({'pair_id':f'{name}-{i}','group_id':f'{name}-q{i//2}','source':name,'coarse_domain':'smoke',
                'input_text':f'Question {i//2}: compare answer alpha and answer beta {i%3}',
                'ab':{'parse':p,'canonical_verdict':'A'},
                'ba':{'parse':{**p,'slot':'A' if y else 'B'},'canonical_verdict':'B' if y else 'A'},'y_pos':y,
                'samples':[{'parse':{**p,'slot':'A' if j%2 else 'B'}} for j in range(5)]})
            metadata.append({'pair_id':f'{name}-{i}','confidence':70+i%25,'signed_logit_margin':float(rng.normal()),
                  'response_a_tokens':10+i%3,'response_b_tokens':12+i%4,'signed_length_difference':-2+i%2,'rationale_tokens':12,
                  'verdict_probability':.8,'verdict_entropy':float(rng.uniform(.1,1)), 'label_entropy':.5,
                  'rationale_mean_log_probability':-.5,'rationale_mean_entropy':1.})
            activations.append(rng.normal(size=(4,6))+(2*y-1)*.8)
        hidden = np.asarray(activations,dtype=np.float32)
        write_jsonl(directory/'judgments.jsonl',records)
        write_jsonl(directory/'feature_metadata.jsonl',metadata)
        manifest = {'protocol_version':PROTOCOL_VERSION,'model':'synthetic-smoke','requested_revision':'0'*40,
                    'system_prompt':SYSTEM_PROMPT,'user_template':USER_TEMPLATE,'candidate_layers':[1,2,3,4],
                    'dtype':'torch.float32','max_new_tokens':256,'do_sample':False,'qwen_thinking':False,'context_limit':32768,
                    'synthetic':True}
        write_json(directory/'run.json',manifest)
        np.savez_compressed(directory/'features.npz',pair_ids=np.asarray([r['pair_id'] for r in records]),
                             layer_numbers=np.array([1,2,3,4]),hidden_states=hidden,input_hidden_states=hidden*.7)
        embedding_file = directory/'embeddings.npz'
        np.savez_compressed(embedding_file,pair_ids=np.asarray([r['pair_id'] for r in records]),
            text_hashes=np.asarray([text_hash(r) for r in records]),input_embeddings=rng.normal(size=(count,4)),
            rationale_embeddings=rng.normal(size=(count,4)))
        return directory, embedding_file
    source, embeddings = materialize('source',40)
    target, _ = materialize('target',24)
    result = analyze('smoke',source,root/'analysis/smoke',embeddings,bootstrap_reps=8)
    assert result['full_ten_baselines'] and len(result['predictors']) == 12
    frozen_weights = (root/'analysis/smoke/probe.npz').read_bytes()
    target_result = transfer('smoke',target,root/'analysis/smoke',root/'transfer/smoke',bootstrap_reps=8)
    assert (root/'analysis/smoke/probe.npz').read_bytes() == frozen_weights
    assert target_result['target_fitting_or_calibration'] is False
    model = load_model(root/'analysis/smoke/probe.npz')
    assert np.isfinite(model.weights).all()
    write_json(root/'analysis/results.json',{'judges':{'smoke':result}})
    subprocess.run([sys.executable,'-m','position_bias.controls','--run',str(source),'--analysis',str(root/'analysis/smoke'),
                    '--embeddings',str(embeddings),'--output-dir',str(root/'controls')],check=True)
    subprocess.run([sys.executable,'-m','position_bias.report','--analysis',str(root/'analysis'),'--output-dir',str(root/'report')],check=True)
    report = {'status':'passed','inputs':'synthetic smoke only; not experimental evidence','source_pairs':40,'target_pairs':24,
              'checked':['tiny transformer generation/capture/replay','canonical reversal identity','deterministic tie ranking',
                         'five outer / three inner grouped folds','ten baselines','source-only model export',
                         'frozen transfer without changing source weights','question bootstrap and checking costs',
                         'ridge activation controls','CSV/figure export'],
              'full_model_experiments_run':False}
    write_json(Path('tmp/smoke-report.json'),report)
    print(json.dumps(report,indent=2))
PY
