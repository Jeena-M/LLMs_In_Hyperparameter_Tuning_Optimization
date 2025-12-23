#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Model Comparison for Hyperparameter Suggestion (Equal-Budget)
- Compares ChatGPT variants (OpenAI GPT-4o family) vs grid & random
- Same prompt, K candidates/model, function-aware LR clamps
- Equal iteration budget for all methods/models
- Logs to CSV and draws simple bar charts

Requires:
  export OPENAI_API_KEY="sk-..."
  pip install "openai>=1.0.0" pandas matplotlib
"""

import os, json, time, itertools, random, sys
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Optional provider (OpenAI) --------
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    pass

# ======= Config =======

# provider ∈ {"openai","mock"}
LLM_MODELS = [
    {"provider": "openai", "model": "gpt-4o"},        # OpenAI flagship
    {"provider": "openai", "model": "gpt-4o-mini"},   # OpenAI fast/budget
    {"provider": "mock",   "model": "mock-stable"},   # Fallback deterministic baseline
]

K_CANDIDATES = 3          # candidates proposed by each LLM model
MAX_ITER_BUDGET = 2000    # equal budget for every method/model
SEEDS = [0, 1, 2]         # replicate runs (increase later)
DIMS  = [2]               # try [2,5,10] to scale up

# LR bounds per function (used for clamping + in prompts)
LR_BOUNDS = {
    "Rastrigin":  (1e-4, 1e-2),
    "Rosenbrock": (1e-5, 1e-3),
}

# ======= Benchmark functions & gradients =======

def rastrigin(x: np.ndarray) -> float:
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2*np.pi*x))

def rastrigin_grad(x: np.ndarray) -> np.ndarray:
    return 2*x + 20*np.pi*np.sin(2*np.pi*x)

def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    g = np.zeros_like(x)
    g[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    g[1:-1] = 200*(x[1:] - x[:-1]**2) - 400*x[1:-1]*(x[2:] - x[1:-1]**2) - 2*(1 - x[1:-1])
    g[-1] = 200*(x[-1] - x[-2]**2)
    return g

FUNCTIONS = [
    ("Rastrigin",  rastrigin,  rastrigin_grad),
    ("Rosenbrock", rosenbrock, rosenbrock_grad),
]

# ======= Gradient Descent (with safety) =======

def gradient_descent(f, grad_f, x0, lr, max_iter=1000, tol=1e-6):
    x = x0.copy()
    prev_f = None
    for i in range(max_iter):
        g = grad_f(x)
        # safety: clip to avoid overflow
        g = np.clip(g, -1e3, 1e3)
        x_new = x - lr * g
        fx = f(x)
        if not np.isfinite(fx) or not np.all(np.isfinite(x_new)):
            return x, np.inf, i+1  # failure
        x = x_new
        if np.linalg.norm(g) < tol:
            return x, f(x), i+1
        # plateau check every 50 steps
        if prev_f is not None and (i % 50 == 0) and abs(prev_f - fx) < 1e-8:
            return x, f(x), i+1
        prev_f = fx
    return x, f(x), max_iter

# ======= Baselines: grid & random =======

def grid_search(f, grad_f, x0, func_name):
    lo, hi = LR_BOUNDS[func_name]
    # 3-point log grid
    learning_rates = [lo, (lo*hi)**0.5, hi]
    results = []
    for lr in learning_rates:
        x, val, iters = gradient_descent(f, grad_f, x0, lr, max_iter=MAX_ITER_BUDGET)
        results.append({"method":"grid","lr":lr,"max_iter":MAX_ITER_BUDGET,"final_val":val,"iters":iters})
    return sorted(results, key=lambda r: r["final_val"])

def random_search(f, grad_f, x0, func_name, n_samples=5):
    lo, hi = LR_BOUNDS[func_name]
    results = []
    for _ in range(n_samples):
        lr = 10 ** np.random.uniform(np.log10(lo), np.log10(hi))
        x, val, iters = gradient_descent(f, grad_f, x0, lr, max_iter=MAX_ITER_BUDGET)
        results.append({"method":"random","lr":lr,"max_iter":MAX_ITER_BUDGET,"final_val":val,"iters":iters})
    return sorted(results, key=lambda r: r["final_val"])

# ======= LLM helpers =======

def clamp_lr(func_name: str, lr: float) -> float:
    lo, hi = LR_BOUNDS[func_name]
    return float(max(lo, min(hi, lr)))

def build_prompt(func_name: str, dim: int, k: int) -> str:
    lo, hi = LR_BOUNDS[func_name]
    return (
        "You are an optimization expert.\n"
        f"We minimize the {func_name} function in R^{dim} using vanilla gradient descent (no momentum).\n"
        f"Return {k} candidate hyperparameter sets as a JSON list of objects with keys:\n"
        'learning_rate (float), max_iter (int).\n'
        f"Constraints: learning_rate in [{lo}, {hi}], max_iter = {MAX_ITER_BUDGET}.\n"
        "Prefer conservative, stable learning rates. Respond with ONLY the JSON list."
    )

def parse_json_list(s: str, k: int, fallback_lr: float) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(s.strip())
        if isinstance(obj, list):
            cands = obj
        elif isinstance(obj, dict) and "candidates" in obj:
            cands = obj["candidates"]
        else:
            cands = []
    except Exception:
        cands = []
    if not cands:
        cands = [{"learning_rate": fallback_lr, "max_iter": MAX_ITER_BUDGET} for _ in range(k)]
    out = []
    for c in cands[:k]:
        try:
            lr = float(c.get("learning_rate", fallback_lr))
        except Exception:
            lr = fallback_lr
        try:
            mi = int(c.get("max_iter", MAX_ITER_BUDGET))
        except Exception:
            mi = MAX_ITER_BUDGET
        out.append({"learning_rate": lr, "max_iter": mi})
    return out

# ---- Provider adapters ----

def call_openai(model: str, prompt: str, k: int, fallback_lr: float) -> List[Dict[str, Any]]:
    if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI not available or OPENAI_API_KEY missing.")
    client = OpenAI()
    # Use JSON mode for stricter formatting
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},  # allows {"candidates":[...]} or list
    )
    content = completion.choices[0].message.content
    return parse_json_list(content, k, fallback_lr)

def call_mock(model: str, prompt: str, k: int, fallback_lr: float) -> List[Dict[str, Any]]:
    # Deterministic, conservative candidates near fallback
    return [{"learning_rate": fallback_lr * (0.8 + 0.2*i), "max_iter": MAX_ITER_BUDGET} for i in range(1, k+1)]

PROVIDER_FN = {
    "openai": call_openai,
    "mock":   call_mock,
}

def llm_best_for_model(entry: Dict[str, str], func_name: str, f, grad_f, x0: np.ndarray, dim: int):
    provider = entry["provider"]
    model    = entry["model"]
    lo, hi = LR_BOUNDS[func_name]
    prompt = build_prompt(func_name, dim, K_CANDIDATES)
    fallback_lr = float((lo*hi)**0.5)

    try:
        candidates = PROVIDER_FN[provider](model, prompt, K_CANDIDATES, fallback_lr)
    except Exception as e:
        # Graceful skip: return a "skipped" record
        return {
            "method":"llm","provider":provider,"model":model,"function":func_name,
            "lr": np.nan, "max_iter": MAX_ITER_BUDGET, "final_val": np.inf, "iters": 0,
            "skipped_reason": str(e)
        }

    best = None
    for c in candidates:
        lr = clamp_lr(func_name, float(c["learning_rate"]))
        # Force equal budget
        mi = MAX_ITER_BUDGET
        _, val, iters = gradient_descent(f, grad_f, x0, lr, max_iter=mi)
        rec = {
            "method": "llm", "provider": provider, "model": model, "function": func_name,
            "lr": lr, "max_iter": mi, "final_val": val, "iters": iters,
            "skipped_reason": ""
        }
        if (best is None) or (val < best["final_val"]):
            best = rec
    return best

# ======= Runner =======

def run_once(dim: int, seed: int) -> pd.DataFrame:
    np.random.seed(seed); random.seed(seed)
    rows = []
    for name, f, grad in FUNCTIONS:
        print(f"\n🔹 {name} (R^{dim}), seed={seed}, budget={MAX_ITER_BUDGET}")
        # single shared init per (function, dim, seed) for fairness
        x0 = np.random.uniform(-2, 2, size=dim)

        # Baselines
        best_g = grid_search(f, grad, x0, name)[0]; best_g["function"] = name
        best_r = random_search(f, grad, x0, name)[0]; best_r["function"] = name
        print(f"GRID      → lr={best_g['lr']:.4e}, val={best_g['final_val']:.6g}, iters={best_g['iters']}")
        print(f"RANDOM    → lr={best_r['lr']:.4e}, val={best_r['final_val']:.6g}, iters={best_r['iters']}")
        rows.extend([best_g, best_r])

        # LLM models
        for entry in LLM_MODELS:
            rec = llm_best_for_model(entry, name, f, grad, x0, dim)
            rows.append(rec)
            tag = f"{rec.get('provider','?')}/{rec.get('model','?')}"
            if np.isfinite(rec["final_val"]):
                print(f"LLM[{tag}] → lr={rec['lr']:.4e}, val={rec['final_val']:.6g}, iters={rec['iters']}")
            else:
                print(f"LLM[{tag}] → skipped ({rec['skipped_reason']})")
    df = pd.DataFrame(rows)
    df["dim"]  = dim
    df["seed"] = seed
    return df

def run_all() -> pd.DataFrame:
    all_df = []
    t0 = time.time()
    for dim in DIMS:
        for seed in SEEDS:
            all_df.append(run_once(dim, seed))
    full = pd.concat(all_df, ignore_index=True)
    elapsed = time.time() - t0
    out_csv = "llm_model_compare_results.csv"
    full.to_csv(out_csv, index=False)
    print(f"\nSaved results → {out_csv} (elapsed {elapsed:.1f}s)")
    return full

# ======= Visualization =======

def visualize(df: pd.DataFrame):
    # One bar chart per function (median across seeds)
    for func in df["function"].unique():
        sub = df[(df["function"] == func) & (df["max_iter"] == MAX_ITER_BUDGET)].copy()

        # Aggregate median final value over seeds for each (method, provider, model)
        sub["label"] = sub.apply(
            lambda r: r["method"] if r["method"]!="llm" else f"llm-{r['provider'].split('-')[0]}:{r['model']}",
            axis=1,
        )
        agg = sub.groupby("label", as_index=False)["final_val"].median()
        plt.figure()
        plt.bar(agg["label"], agg["final_val"])
        plt.title(f"{func} — Median Best Final Value by Method/Model (lower=better)")
        plt.ylabel("Final function value (median across seeds)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.show()

# ======= Entry point =======

if __name__ == "__main__":
    df = run_all()

    # Quick textual summary
    with pd.option_context('display.max_rows', None, 'display.max_colwidth', 60):
        cols = ["function","dim","seed","method","provider","model","lr","max_iter","final_val","iters","skipped_reason"]
        print("\n--- Summary (per seed) ---")
        print(df[cols].fillna(""))

    # Aggregate table for quick paste into report
    def success_eps(func):
        return 1e-2 if func=="Rastrigin" else 1e-4

    rows = []
    for (func, dim, method, provider, model), sub in df.groupby(["function","dim","method","provider","model"]):
        # Ignore non-finite values (skipped runs, etc.)
        valid = sub[np.isfinite(sub["final_val"])].copy()
        if valid.empty:
            med = np.inf
            iqr = np.nan
            succ = 0.0
            med_it = np.nan
        else:
            med = valid["final_val"].median()
            iqr = valid["final_val"].quantile(0.75) - valid["final_val"].quantile(0.25)
            succ_mask = valid["final_val"] <= success_eps(func)
            succ = succ_mask.mean()
            if succ_mask.any():
                med_it = valid.loc[succ_mask, "iters"].median()
            else:
                med_it = np.nan

        rows.append([func, dim, method, provider or "", model or "", med, iqr, succ, med_it])

    summary = pd.DataFrame(
        rows,
        columns=["function","dim","method","provider","model","median_final","IQR","success_rate","median_iters_success"]
    )
    summary = summary.sort_values(["function","dim","method","provider","model"])
    print("\n--- Aggregated (median/IQR & success rate) ---")
    print(summary)

    # Basic plots
    visualize(df)
