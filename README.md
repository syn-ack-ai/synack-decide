# SynACK Decide

Inference code for **SynACK Decide 26B-A4B**, an open "System One" decision model. Give it a `state` and typed
questions (multiple choice with 2–255 options, or yes/no). It returns an answer with a probability for every
option, from one forward pass per question.

- Weights: [SkyPanther/synack-decide-26b-a4b](https://huggingface.co/SkyPanther/synack-decide-26b-a4b) (bf16),
  [-mlx-8bit](https://huggingface.co/SkyPanther/synack-decide-26b-a4b-mlx-8bit) (Apple Silicon),
  [-GGUF](https://huggingface.co/SkyPanther/synack-decide-26b-a4b-GGUF) (llama.cpp)
- Results, training details and limitations: see the model card.
- Write-up: https://syn-ack.ai/posts/teaching-a-small-model-to-decide

## Wire format

Same as the Jev Decision Index harness and llama.cpp `/v1/systemone`:

```bash
curl -s localhost:8765/v1/systemone -H 'Content-Type: application/json' -d @examples/request.json
```
```json
{"model": "...", "answers": {
   "team":   {"type": "choice", "choice": "billing", "confidence": 0.997, "probabilities": {"billing": 0.997, "shipping": 0.001, "technical": 0.001}},
   "urgent": {"type": "noul", "noul": 0.95}},
 "usage": {"input_tokens": 257}, "latency_ms": 450}
```

## Run it

**Apple Silicon (MLX):**
```bash
pip install -r requirements-mlx.txt
python serve.py --state "I was charged twice." --question "Which team?" --options billing,shipping,technical --yesno "Is it urgent?"
python serve.py --serve 8765            # POST /v1/systemone (add --host 0.0.0.0 for your LAN; there is no auth)
```

**CUDA (transformers, bf16, about 52 GB of VRAM):**
```bash
pip install -r requirements-cuda.txt
python serve.py --backend cuda --serve 8765
```

**llama.cpp** (Oct 2026 or newer, which has `/v1/systemone`): download a GGUF from the `-GGUF` repo, then
```bash
llama-server -m synack-decide-26b-a4b-Q8_0.gguf --port 8080     # POST /v1/systemone
```

**Decision Index harness** ([apolinario/decision-index](https://github.com/apolinario/decision-index)):
```bash
PYTHONPATH=. python -m decision_index run --engine engine_cuda:CudaEngine --model SkyPanther/synack-decide-26b-a4b ...
PYTHONPATH=. python -m decision_index run --engine engine_mlx:MLXEngine --model SkyPanther/synack-decide-26b-a4b-mlx-8bit ...
```

## Files

| File | What |
|---|---|
| `common.py` | the training prompt: system turn, compact JSON user turn, 255 single-token labels (`labels255.json`) |
| `engine_mlx.py` | MLX engine (`load_mlx`, `label_probs`, `MLXEngine`) |
| `engine_cuda.py` | transformers engine; batches multi-question requests (`CudaEngine`) |
| `serve.py` | `/v1/systemone` server and one-off CLI |
| `gguf/systemone.jinja` | llama.cpp decision template (type `nimble`) that reproduces `common.py` byte for byte |
| `gguf/add_metadata.py` | adds the decision type and template to a converted GGUF |

Building your own GGUF:
```bash
python llama.cpp/convert_hf_to_gguf.py <merged-hf-dir> --outtype bf16 --outfile raw.gguf
python gguf/add_metadata.py raw.gguf synack-decide-bf16.gguf --llama-cpp llama.cpp
llama-quantize synack-decide-bf16.gguf synack-decide-Q8_0.gguf Q8_0
```

Tips:
- Ask the judgement question and let code do the mechanical steps (sorting, counting). The model answers in
  one pass and slips on two-step compositions such as "third largest".
- Keep option descriptions short. Use meaningful keys; generic keys like `option_1` are hidden from the model.

## License

Code: Apache-2.0. Weights: Apache-2.0, following the base model; see the model card for training-data terms.
