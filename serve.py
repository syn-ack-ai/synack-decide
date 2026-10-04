"""SynACK Decide: /v1/systemone server and one-off CLI (MLX on Apple Silicon, or CUDA).

Server (Decision Index / llama.cpp wire format):
  python serve.py --serve 8765 [--backend mlx|cuda] [--model ...]
  POST /v1/systemone {"state": ..., "questions": {"id": {"type": "choice", "instructions": "...",
                      "criteria": {"opt": "description" | null, ...}} | {"type": "noul", "instructions": "..."}}}
  -> {"model", "answers": {"id": {"type": "choice", "choice", "probabilities", "confidence"} | {"type": "noul", "noul"}},
      "usage": {"input_tokens"}, "latency_ms"}

One-off CLI:
  python serve.py --state "Customer: I was charged twice." --question "Which team?" --options billing,shipping,technical
  python serve.py --state "..." --yesno "Is the customer angry?"
"""
import argparse
import json
import platform
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS  # noqa: E402

MODELS = {"mlx": "SkyPanther/synack-decide-26b-a4b-mlx-8bit", "cuda": "SkyPanther/synack-decide-26b-a4b"}


class Decider:
    def __init__(self, backend, model):
        t = time.time()
        self.name, self.backend = model, backend
        if backend == "mlx":
            from engine_mlx import label_probs, load_mlx
            tok, m, ids = load_mlx(model)
            self._probs = lambda state, q: label_probs(tok, m, ids, state, q)
        else:
            from engine_cuda import CudaEngine
            eng = CudaEngine(model)
            self._probs = eng.probs
        self.lock = threading.Lock()
        print(f"loaded {model} ({backend}) in {time.time() - t:.1f}s", file=sys.stderr, flush=True)

    def answer(self, state, questions):
        if not isinstance(questions, dict) or not questions:
            raise ValueError('"questions" must be a non-empty object')
        out, n_tok, t0 = {}, 0, time.time()
        with self.lock:
            for qid, q in questions.items():
                if not isinstance(q, dict) or q.get("type") not in ("choice", "noul"):
                    raise ValueError(f'question "{qid}": type must be "choice" or "noul"')
                if q["type"] == "choice" and not (2 <= len(q.get("criteria") or {}) <= len(LABELS)):
                    raise ValueError(f'question "{qid}": choice needs 2-{len(LABELS)} criteria')
                probs, n = self._probs(state, q)
                n_tok += n
                if q["type"] == "noul":
                    out[qid] = {"type": "noul", "noul": round(probs["true"], 6)}
                else:
                    best = max(probs, key=probs.get)
                    out[qid] = {"type": "choice", "choice": best, "confidence": round(probs[best], 6),
                                "probabilities": {k: round(v, 6) for k, v in probs.items()}}
        return {"model": self.name, "answers": out, "usage": {"input_tokens": n_tok},
                "latency_ms": round((time.time() - t0) * 1000, 1)}


def serve(decider, host, port):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path not in ("/v1/systemone", "/systemone"):
                return self._send(404, {"error": "not found; POST /v1/systemone"})
            try:
                b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, decider.answer(b.get("state", ""), b.get("questions")))
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": str(e)})

        def log_message(self, *args):
            pass  # request bodies are never logged

    print(f"System One endpoint: http://{host}:{port}/v1/systemone", file=sys.stderr, flush=True)
    # Single-threaded on purpose: MLX GPU streams belong to the thread that loaded the model.
    HTTPServer((host, port), H).serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", choices=["mlx", "cuda"],
                   default="mlx" if platform.system() == "Darwin" and platform.machine() == "arm64" else "cuda")
    p.add_argument("--model", help="model path or Hugging Face id (default depends on --backend)")
    p.add_argument("--serve", type=int, help="port for the /v1/systemone server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--state", default="")
    p.add_argument("--question", help="choice question (use with --options)")
    p.add_argument("--options", help="comma-separated options, or a JSON object {option: description}")
    p.add_argument("--yesno", help="yes/no question")
    a = p.parse_args()
    if not (a.serve or a.question or a.yesno):
        p.error("give --question/--options and/or --yesno, or --serve PORT")
    if a.question and not a.options:
        p.error("--question needs --options")
    d = Decider(a.backend, a.model or MODELS[a.backend])
    if a.serve:
        return serve(d, a.host, a.serve)
    qs = {}
    if a.question:
        crit = json.loads(a.options) if a.options.strip().startswith("{") else {o.strip(): None for o in a.options.split(",")}
        qs["q"] = {"type": "choice", "instructions": a.question, "criteria": crit}
    if a.yesno:
        qs["yesno"] = {"type": "noul", "instructions": a.yesno}
    print(json.dumps(d.answer(a.state, qs), indent=2))


if __name__ == "__main__":
    main()
