#!/usr/bin/env python3
"""
measure_ram.py — Generic RAM Measurement Tool for GGUF Models
=============================================================
Measures real peak RSS RAM usage via psutil for one or more GGUF model
variants, then optionally updates reports/model_comparison.json with the
measured peak_ram_gb values.

Usage examples
--------------
# Measure a single model, print results only (no JSON update):
python measure_ram.py \
    --model-key qwen36_27b_q4km \
    --model-label "Qwen3.6 27B Q4_K_M" \
    --model-path /root/slm_eval_harness/models/qwen_3.6_27b/Qwen3.6-27B-Q4_K_M.gguf

# Measure multiple models defined in a config file and update model_comparison.json:
python measure_ram.py --config configs/ram_measure_qwen36_27b.json --update-json

# Measure the built-in Qwen3.6 27B preset and update model_comparison.json:
python measure_ram.py --preset qwen36_27b --update-json

# Override inference settings:
python measure_ram.py --preset qwen36_27b --n-ctx 2048 --max-tokens 128 --passes 3 --update-json

Config file format (JSON)
-------------------------
[
  {
    "key":   "my_model_q4",
    "label": "My Model Q4_K_M",
    "path":  "/path/to/model-Q4_K_M.gguf"
  },
  {
    "key":   "my_model_bf16",
    "label": "My Model BF16",
    "path":  "/path/to/model-BF16-00001-of-00002.gguf"
  }
]
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import psutil
from llama_cpp import Llama

# ── Defaults ───────────────────────────────────────────────────────────────────
BASE_DIR            = Path("/root/slm_eval_harness")
REPORTS_DIR         = BASE_DIR / "reports"
MODEL_COMPARISON    = REPORTS_DIR / "model_comparison.json"
DEFAULT_PROMPT      = "Explain the difference between supervised and unsupervised learning."
DEFAULT_N_CTX       = 512
DEFAULT_MAX_TOKENS  = 64
DEFAULT_PASSES      = 2

# ── Built-in presets ───────────────────────────────────────────────────────────
PRESETS = {
    "qwen36_27b": [
        {
            "key":   "qwen36_27b_q4km",
            "label": "Qwen3.6 27B Q4_K_M",
            "path":  str(BASE_DIR / "models/qwen_3.6_27b/Qwen3.6-27B-Q4_K_M.gguf"),
        },
        {
            "key":   "qwen36_27b_q80",
            "label": "Qwen3.6 27B Q8_0",
            "path":  str(BASE_DIR / "models/qwen_3.6_27b/Qwen3.6-27B-Q8_0.gguf"),
        },
        {
            "key":   "qwen36_27b_bf16",
            "label": "Qwen3.6 27B BF16",
            "path":  str(BASE_DIR / "models/qwen_3.6_27b/BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf"),
        },
    ],
    "gemma4_26b": [
        {
            "key":   "gemma4-26b-a4b-q4",
            "label": "Gemma4 26B Q4_K_M",
            "path":  str(BASE_DIR / "models/gemma4_26b/gemma-4-27b-it-Q4_K_M.gguf"),
        },
        {
            "key":   "gemma4-26b-a4b-q8",
            "label": "Gemma4 26B Q8_0",
            "path":  str(BASE_DIR / "models/gemma4_26b/gemma-4-27b-it-Q8_0.gguf"),
        },
        {
            "key":   "gemma4-26b-a4b-bf16",
            "label": "Gemma4 26B BF16",
            "path":  str(BASE_DIR / "models/gemma4_26b/BF16/gemma-4-27b-it-BF16-00001-of-00002.gguf"),
        },
    ],
}


# ── Core helpers ───────────────────────────────────────────────────────────────

def get_rss_gb() -> float:
    """Return current process RSS memory in GB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def measure_model(model_info: dict, n_ctx: int, max_tokens: int,
                  passes: int, prompt: str) -> dict:
    """
    Load a GGUF model, run `passes` inference passes, return RAM measurements.

    Parameters
    ----------
    model_info : dict  — keys: key, label, path
    n_ctx      : int   — context window size passed to llama-cpp
    max_tokens : int   — tokens to generate per inference pass
    passes     : int   — number of inference passes to run
    prompt     : str   — test prompt used for inference
    """
    key   = model_info["key"]
    label = model_info["label"]
    path  = model_info["path"]

    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"  n_ctx={n_ctx}  max_tokens={max_tokens}  passes={passes}")
    print(f"{'='*62}")

    if not Path(path).exists():
        print(f"  ERROR: model file not found — skipping")
        return {"key": key, "label": label, "error": "model file not found"}

    # 1. Baseline
    gc.collect()
    time.sleep(2)
    initial_ram = get_rss_gb()
    print(f"  Initial RAM (before load): {initial_ram:.2f} GB")

    # 2. Load
    print(f"  Loading model...")
    t0 = time.time()
    model = Llama(
        model_path=path,
        n_ctx=n_ctx,
        n_gpu_layers=0,   # CPU only
        verbose=False,
    )
    load_time = time.time() - t0
    post_load_ram = get_rss_gb()
    print(f"  Post-load RAM:             {post_load_ram:.2f} GB  (load took {load_time:.1f}s)")

    # 3. Inference passes
    peak_ram = post_load_ram
    for i in range(passes):
        print(f"  Inference pass {i+1}/{passes} ...")
        _ = model(prompt, max_tokens=max_tokens, temperature=0.0, echo=False)
        current_ram = get_rss_gb()
        peak_ram = max(peak_ram, current_ram)
        print(f"    RAM after pass {i+1}: {current_ram:.2f} GB")

    model_ram_usage = peak_ram - initial_ram

    print(f"\n  ── Results ──────────────────────────────────────────")
    print(f"  initial_ram_gb:      {initial_ram:.2f}")
    print(f"  post_load_ram_gb:    {post_load_ram:.2f}")
    print(f"  peak_ram_gb:         {peak_ram:.2f}")
    print(f"  model_ram_usage_gb:  {model_ram_usage:.2f}")

    # 4. Unload
    print(f"\n  Unloading model and freeing RAM...")
    del model
    gc.collect()
    time.sleep(5)
    print(f"  RAM after unload: {get_rss_gb():.2f} GB")

    return {
        "key":               key,
        "label":             label,
        "path":              path,
        "initial_ram_gb":    round(initial_ram, 2),
        "post_load_ram_gb":  round(post_load_ram, 2),
        "peak_ram_gb":       round(peak_ram, 2),
        "model_ram_usage_gb": round(model_ram_usage, 2),
        "n_ctx":             n_ctx,
        "max_tokens":        max_tokens,
        "passes":            passes,
    }


def update_model_comparison(results: list, json_path: Path = MODEL_COMPARISON):
    """Write measured peak_ram_gb + model_ram_usage_gb into model_comparison.json."""
    if not json_path.exists():
        print(f"\nERROR: {json_path} not found — cannot update.")
        return

    with open(json_path) as f:
        data = json.load(f)

    models = data.get("models", {})
    updated, skipped = [], []

    for r in results:
        if "error" in r:
            skipped.append((r["key"], r["error"]))
            continue
        key = r["key"]
        if key not in models:
            skipped.append((key, "key not found in model_comparison.json"))
            continue
        old = models[key].get("inference_metrics", {}).get("peak_ram_gb", "N/A")
        models[key].setdefault("inference_metrics", {}).update({
            "peak_ram_gb":        r["peak_ram_gb"],
            "model_ram_usage_gb": r["model_ram_usage_gb"],
        })
        updated.append((key, old, r["peak_ram_gb"]))

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\n{'='*62}")
    print(f"  {json_path} updated")
    if updated:
        print(f"\n  {'Model Key':<32} {'Old (GB)':>10} {'New (GB)':>10}")
        print(f"  {'-'*54}")
        for key, old, new in updated:
            print(f"  {key:<32} {str(old):>10} {new:>10.2f}")
    if skipped:
        print(f"\n  Skipped:")
        for key, reason in skipped:
            print(f"    {key}: {reason}")
    print(f"{'='*62}")


def print_summary(results: list):
    print(f"\n{'='*62}")
    print("  FINAL SUMMARY")
    print(f"  {'Model':<32} {'peak_ram_gb':>12} {'model_ram_gb':>13}")
    print(f"  {'-'*59}")
    for r in results:
        if "error" in r:
            print(f"  {r.get('label', r['key']):<32} {'ERROR':>12}")
        else:
            print(f"  {r['label']:<32} {r['peak_ram_gb']:>12.2f} {r['model_ram_usage_gb']:>13.2f}")
    print(f"{'='*62}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure peak RSS RAM for GGUF models via llama-cpp-python.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model source (mutually exclusive)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--preset",     choices=list(PRESETS.keys()),
                     help="Use a built-in model preset.")
    src.add_argument("--config",     type=Path,
                     help="Path to a JSON config file listing models to measure.")
    src.add_argument("--model-path", type=str,
                     help="Path to a single GGUF file (use with --model-key / --model-label).")

    # Single-model metadata (used with --model-path)
    p.add_argument("--model-key",   default="custom_model",
                   help="Key to use in model_comparison.json (default: custom_model).")
    p.add_argument("--model-label", default=None,
                   help="Human-readable label (default: derived from filename).")

    # Inference settings
    p.add_argument("--n-ctx",      type=int, default=DEFAULT_N_CTX,
                   help=f"Context window size (default: {DEFAULT_N_CTX}).")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help=f"Tokens to generate per inference pass (default: {DEFAULT_MAX_TOKENS}).")
    p.add_argument("--passes",     type=int, default=DEFAULT_PASSES,
                   help=f"Number of inference passes per model (default: {DEFAULT_PASSES}).")
    p.add_argument("--prompt",     type=str, default=DEFAULT_PROMPT,
                   help="Test prompt for inference passes.")

    # Output
    p.add_argument("--update-json", action="store_true",
                   help="Write measured values into reports/model_comparison.json.")
    p.add_argument("--json-path",   type=Path, default=MODEL_COMPARISON,
                   help=f"Path to model_comparison.json (default: {MODEL_COMPARISON}).")
    p.add_argument("--output",      type=Path, default=None,
                   help="Save raw results to this JSON file (optional).")

    return p


def resolve_models(args) -> list:
    """Return a list of model dicts from whichever source was specified."""
    if args.preset:
        return PRESETS[args.preset]

    if args.config:
        if not args.config.exists():
            print(f"ERROR: config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        with open(args.config) as f:
            models = json.load(f)
        if not isinstance(models, list):
            print("ERROR: config file must be a JSON array of model objects.", file=sys.stderr)
            sys.exit(1)
        return models

    if args.model_path:
        label = args.model_label or Path(args.model_path).stem
        return [{"key": args.model_key, "label": label, "path": args.model_path}]

    # Default: run the qwen36_27b preset if nothing specified
    print("No model source specified — running built-in 'qwen36_27b' preset.")
    print("Use --help to see all options.\n")
    return PRESETS["qwen36_27b"]


def main():
    parser = build_parser()
    args   = parser.parse_args()
    models = resolve_models(args)

    print("=" * 62)
    print("  measure_ram.py — GGUF RAM Measurement")
    print(f"  Models to measure : {len(models)}")
    print(f"  n_ctx             : {args.n_ctx}")
    print(f"  max_tokens        : {args.max_tokens}")
    print(f"  passes            : {args.passes}")
    print(f"  System RAM avail  : {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print("=" * 62)

    results = []
    for m in models:
        result = measure_model(m, args.n_ctx, args.max_tokens, args.passes, args.prompt)
        results.append(result)

    print_summary(results)

    if args.update_json:
        update_model_comparison(results, args.json_path)

    # Save raw output
    out_path = args.output
    if out_path is None and (args.preset or args.config):
        tag = args.preset or args.config.stem
        out_path = REPORTS_DIR / f"inference_metrics_{tag}_ram.json"

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Raw results saved to: {out_path}")


if __name__ == "__main__":
    main()
