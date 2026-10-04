"""SynACK Decide on Apple Silicon (MLX): one forward pass per question, softmax over the option labels.

  from engine_mlx import load_mlx, label_probs
  tok, model, ids = load_mlx("SkyPanther/synack-decide-26b-a4b-mlx-8bit")
  probs, n_tokens = label_probs(tok, model, ids, state, {"type": "choice", "instructions": "...", "criteria": {...}})

Also a Decision Index harness engine (github.com/apolinario/decision-index):
  python -m decision_index run --engine engine_mlx:MLXEngine --model SkyPanther/synack-decide-26b-a4b-mlx-8bit ...
"""
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, SYSTEM, decision_user, question_criteria  # noqa: E402

try:
    from decision_index.engines.base import Engine, Unsupported
except ImportError:  # the harness is optional
    Engine = object

    class Unsupported(ValueError):
        pass


def load_mlx(model, adapter=None):
    m, tok = load(model, adapter_path=adapter)
    ids = []
    for l in LABELS:
        t = tok.encode(l, add_special_tokens=False)
        assert len(t) == 1, (l, t)
        ids.append(t[0])
    return tok, m, ids


def label_probs(tok, model, label_ids, state, q, limit=32768, temperature=1.0):
    crit = question_criteria(q)
    keys = list(crit)
    if not 2 <= len(keys) <= len(LABELS):
        raise Unsupported(f"supports 2-{len(LABELS)} options per question; question has {len(keys)}")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": decision_user(state, q, keys, crit)}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    if len(ids) + 1 > limit:
        raise Unsupported(f"prompt of {len(ids)} tokens exceeds the {limit}-token context window")
    logits = model(mx.array(ids)[None])[0, -1]
    p = mx.softmax(logits[mx.array(label_ids[:len(keys)])].astype(mx.float32) / temperature)
    mx.eval(p)
    return dict(zip(keys, p.tolist())), len(ids)


class MLXEngine(Engine):
    name = "synack-decide-mlx"
    latency = "In-process MLX wall time per request (one forward pass per question, evaluated); excludes loading."

    def __init__(self, model, adapter=None, max_tokens=32768, temperature=1.0, **options):
        if Engine is not object:
            super().__init__(**options)
        self.tok, self.model, self.label_ids = load_mlx(model, adapter)
        self.limit, self.temperature = int(max_tokens), float(temperature)
        self.provenance = {"kind": "synack-decide-mlx", "model": model, "adapter": adapter, "temperature": self.temperature}

    def __call__(self, state, questions):
        answers, n_in = {}, 0
        for qkey, q in questions.items():
            p, n = label_probs(self.tok, self.model, self.label_ids, state, q, self.limit, self.temperature)
            n_in += n
            answers[qkey] = ({"type": "noul", "noul": p["true"]} if q["type"] == "noul"
                             else {"type": "choice", "choice": max(p, key=p.get), "probabilities": p})
        return {"model": self.provenance["model"], "answers": answers, "usage": {"input_tokens": n_in}}, None
