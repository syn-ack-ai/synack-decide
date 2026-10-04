"""SynACK Decide on CUDA (transformers, bf16): the engine behind the published numbers.

Each question is rendered exactly like training (common.py): state + that question + labelled
options; one forward pass; softmax over the option labels. noul questions use true/false options
and report p(true). No truncation, no option filtering; > 255 options raises Unsupported.
Multi-question requests are batched (right padding, logits at each row's last real token).

  from engine_cuda import CudaEngine
  eng = CudaEngine("SkyPanther/synack-decide-26b-a4b")
  response, _ = eng(state, questions)

Decision Index harness: python -m decision_index run --engine engine_cuda:CudaEngine --model SkyPanther/synack-decide-26b-a4b ...
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from decision_index.engines.base import Engine, Unsupported
except ImportError:  # the harness is optional
    Engine = object

    class Unsupported(ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, SYSTEM, decision_user, question_criteria  # noqa: E402


def truthy(x):
    return str(x).lower() in ("1", "true", "yes")


def load_cuda(model, adapter=None, four_bit=False, merge=True):
    tok = AutoTokenizer.from_pretrained(model)
    if four_bit:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               llm_int8_skip_modules=["vision_tower", "audio_tower", "embed_vision", "embed_audio", "lm_head"])
        m = AutoModelForCausalLM.from_pretrained(model, quantization_config=q, dtype=torch.bfloat16, device_map={"": 0})
    else:
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map={"": 0})
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
        if merge and not four_bit:
            # Fold LoRA into the base weights once. Unmerged LoRA on the fused MoE experts
            # (PEFT target_parameters) recomputes the expert deltas on every forward pass.
            m = m.merge_and_unload()
    m.eval()
    label_ids = []
    for l in LABELS:
        ids = tok.encode(l, add_special_tokens=False)
        assert len(ids) == 1, (l, ids)
        label_ids.append(ids[0])
    return tok, m, label_ids


class CudaEngine(Engine):
    name = "synack-decide"
    latency = "Device-synchronized in-process wall time per request (one forward pass per question); excludes loading."

    def __init__(self, model="SkyPanther/synack-decide-26b-a4b", adapter=None, four_bit=False, max_tokens=32768,
                 temperature=1.0, merge=True, token_budget=32768, **options):
        if Engine is not object:
            super().__init__(**options)
        self.tok, self.model, self.label_ids = load_cuda(model, adapter, truthy(four_bit), truthy(merge))
        self.limit = int(max_tokens)
        self.temperature = float(temperature)
        self.token_budget = int(token_budget)  # padded tokens per batched forward (multi-question requests)
        self.provenance = {"kind": "synack-decide", "model": model, "adapter": adapter, "four_bit": truthy(four_bit),
                           "merged": truthy(merge) and bool(adapter) and not truthy(four_bit),
                           "temperature": self.temperature,
                           "policy": "SynACK Decide prompt, labels A..IV (single tokens), softmax over option labels; "
                                     "no truncation; >255 options raises Unsupported."}

    def synchronize(self):
        torch.cuda.synchronize()

    @torch.inference_mode()
    def probs(self, state, q):
        crit = question_criteria(q)
        keys = list(crit)
        if not 2 <= len(keys) <= len(LABELS):
            raise Unsupported(f"supports 2-{len(LABELS)} options per question; question has {len(keys)}")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": decision_user(state, q, keys, crit)}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                           tokenize=True, return_dict=False)
        if len(ids) + 1 > self.limit:
            raise Unsupported(f"prompt of {len(ids)} tokens exceeds the {self.limit}-token context window")
        logits = self.model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1).logits[0, -1]
        p = torch.softmax(logits[self.label_ids[:len(keys)]].float() / self.temperature, -1).tolist()
        return dict(zip(keys, p)), len(ids)

    def _prompt(self, state, q):
        crit = question_criteria(q)
        keys = list(crit)
        if not 2 <= len(keys) <= len(LABELS):
            raise Unsupported(f"supports 2-{len(LABELS)} options per question; question has {len(keys)}")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": decision_user(state, q, keys, crit)}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                           tokenize=True, return_dict=False)
        if len(ids) + 1 > self.limit:
            raise Unsupported(f"prompt of {len(ids)} tokens exceeds the {self.limit}-token context window")
        return keys, ids

    def _parts(self):
        """(backbone, lm_head, final softcap) of the underlying causal LM, PEFT-wrapped or merged."""
        top = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        cfg = getattr(top.config, "text_config", top.config)
        return top.model, top.lm_head, getattr(cfg, "final_logit_softcapping", None)

    @torch.inference_mode()
    def _batch_probs(self, prompts):
        """One right-padded forward for several prompts. Causal attention means real tokens never see the
        trailing padding (exact even with sliding-window layers); logits are taken at each row's last real token."""
        L = max(len(ids) for _, ids in prompts)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((len(prompts), L), pad, dtype=torch.long)
        mask = torch.zeros((len(prompts), L), dtype=torch.long)
        for r, (_, p) in enumerate(prompts):
            ids[r, :len(p)] = torch.tensor(p)
            mask[r, :len(p)] = 1
        backbone, lm_head, softcap = self._parts()
        h = backbone(input_ids=ids.cuda(), attention_mask=mask.cuda()).last_hidden_state
        last = torch.tensor([len(p) - 1 for _, p in prompts], device=h.device)
        logits = lm_head(h[torch.arange(len(prompts), device=h.device), last]).float()
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
        out = []
        for r, (keys, _) in enumerate(prompts):
            p = torch.softmax(logits[r, self.label_ids[:len(keys)]].float() / self.temperature, -1).tolist()
            out.append(dict(zip(keys, p)))
        return out

    def __call__(self, state, questions):
        """All questions of a request in as few forward passes as the token budget allows."""
        items = [(qkey, q, *self._prompt(state, q)) for qkey, q in questions.items()]
        answers, n_in = {}, sum(len(ids) for *_, ids in items)
        order = sorted(range(len(items)), key=lambda i: len(items[i][3]))
        chunk, chunk_max = [], 0
        def flush(chunk):
            for i, p in zip(chunk, self._batch_probs([(items[i][2], items[i][3]) for i in chunk])):
                qkey, q = items[i][0], items[i][1]
                answers[qkey] = ({"type": "noul", "noul": p["true"]} if q["type"] == "noul"
                                 else {"type": "choice", "choice": max(p, key=p.get), "probabilities": p})
        for i in order:
            L = len(items[i][3])
            if chunk and max(chunk_max, L) * (len(chunk) + 1) > self.token_budget:
                flush(chunk)
                chunk, chunk_max = [], 0
            chunk.append(i)
            chunk_max = max(chunk_max, L)
        if chunk:
            flush(chunk)
        return {"model": self.provenance["model"], "answers": answers, "usage": {"input_tokens": n_in}}, None
