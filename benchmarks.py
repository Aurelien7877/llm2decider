"""Score a decider on ToxicChat, a public prompt-safety benchmark (downloaded from the Hugging Face Hub).

    python benchmarks.py --model models/decider [--n 200]

ToxicChat (lmsys/toxic-chat, toxicchat0124 test): real user prompts, label = toxicity.
Reports AUPRC (threshold-free, the metric of the Llama Guard paper), plus F1 / recall / FPR at --threshold (default 0.5).
"""
import argparse, json, os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
try:
    import truststore; truststore.inject_into_ssl()
except ImportError:
    pass
import numpy as np
from datasets import load_dataset
from unsloth import FastDecisionModel
from train import QUESTIONS, QID

BENCHMARKS = {
    "ToxicChat": lambda: [(r["user_input"], r["toxicity"] == 1) for r in load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")],
}


def auprc(p, y):  # average precision: mean of the precision at the rank of each positive
    order = np.argsort(-p); y = y[order]
    return float((np.cumsum(y) / np.arange(1, len(y) + 1))[y].mean())


def at(p, y, t):
    b = p >= t; tp = (b & y).sum()
    prec, rec = tp / max(b.sum(), 1), tp / max(y.sum(), 1)
    return {"f1": float(2 * prec * rec / max(prec + rec, 1e-9)), "recall": float(rec), "fpr": float(b[~y].mean()), "precision": float(prec)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=0, help="first n examples of each benchmark only")
    ap.add_argument("--out", default="benchmarks.json")
    a = ap.parse_args()

    model, tok = FastDecisionModel.from_pretrained(model_name=a.model, max_seq_length=512, load_in_4bit=False)
    FastDecisionModel.for_inference(model)
    FastDecisionModel.predict(model, tok, "warm-up", QUESTIONS)
    report = {}
    for name, get in BENCHMARKS.items():
        data = get()[: a.n or None]
        y = np.array([lab for _, lab in data])
        t0 = time.time()
        p = np.array([FastDecisionModel.predict(model, tok, x, QUESTIONS)[QID]["probabilities"]["true"] for x, _ in data])
        report[name] = {"n": len(data), "positives": int(y.sum()), "auprc": auprc(p, y),
                        "threshold 0.5": at(p, y, 0.5), f"threshold {a.threshold:g}": at(p, y, a.threshold),
                        "ms_per_message": (time.time() - t0) * 1e3 / len(data)}
        print(name, json.dumps(report[name]), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print("→", a.out)
