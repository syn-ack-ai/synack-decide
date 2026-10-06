# Decision Index run code

The exact engine that produced the SynACK Decide v7 Decision Index 0.2.1 run (59.54, complete).
`v2/engine_v2.py` and `v2/common.py` are byte-identical to the files the run used. The layout (`v2/` next to
`bench/labels255.json`) is the one `common.py` expects.

```bash
pip install "torch>=2.8" "transformers>=5.18" accelerate \
  "https://github.com/apolinario/decision-index/archive/87d4650b42b377c0291a89c1f1a879f9b31082bf.zip"
cd leaderboard
PYTHONPATH=v2 python -m decision_index pipeline --engine engine_v2:V2Engine \
  --option model=SkyPanther/synack-decide-26b-a4b --out run-v7
```

- **Model:** the merged bf16 weights, about 52 GB of VRAM. The run used one RTX PRO 6000 with PyTorch 2.14.1+cu130
  and transformers 5.18.0.
- **Readout:** each question is rendered exactly as in training (`common.py`): the system turn, then compact JSON
  with the state, the question and the labelled options. One forward pass reads the logits of the option labels
  (A–Z, AA, AB, …, up to 255 single tokens), and a softmax over them gives every option a probability. Nothing is
  generated. Yes/no questions use two fixed options and report p(true).
- **Questions in one request:** they are batched in one padded forward pass under a 32,768-token budget, and their
  shared prompt prefix is computed once (`prefix_reuse`, default on).
- **No truncation, no option filtering:** questions with more than 255 options, or prompts over 32,768 tokens,
  raise `Unsupported`. In the run, only rows outside the 0.2.1 edition hit this: 167 of the 442 excluded rows, all ToolRet
  retrieval prompts of 34,000 to 38,444 tokens (never scored). Raising the limit with `--option max_tokens=40960` should
  cover them (not run).
- **Latency:** per-request device-synchronized wall time. The median was 132 ms in the run.
- **Other paths:** `engine_cuda.py` and `serve.py` at the repo root use the same prompt and readout.
  `serve.py --backend cuda` serves `POST /v1/systemone` for the kit's `http` engine.
