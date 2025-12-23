import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
import json


# 1. Benchmark Functions


def rastrigin(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def rastrigin_grad(x):
    return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def rosenbrock_grad(x):
    grad = np.zeros_like(x)
    grad[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    grad[1:-1] = 200 * (x[1:] - x[:-1] ** 2) - 400 * x[1:-1] * (x[2:] - x[1:-1] ** 2) - 2 * (1 - x[1:-1])
    grad[-1] = 200 * (x[-1] - x[-2] ** 2)
    return grad


# 2. Gradient Descent Function


def gradient_descent(f, grad_f, x0, lr, max_iter=1000, tol=1e-6):
    x = x0.copy()
    for i in range(max_iter):
        grad = grad_f(x)
        x -= lr * grad
        if np.linalg.norm(grad) < tol:
            break
    return x, f(x), i + 1  # final x, final function value, iterations



# 3. Grid Search


def grid_search(f, grad_f, x0):
    learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1]
    max_iters = [500, 1000, 2000]
    results = []

    for lr, mi in itertools.product(learning_rates, max_iters):
        x, val, iters = gradient_descent(f, grad_f, x0, lr, max_iter=mi)
        results.append({"method": "grid", "lr": lr, "max_iter": mi, "final_val": val, "iters": iters})
    return sorted(results, key=lambda x: x["final_val"])



# 4. Random Search


def random_search(f, grad_f, x0, n_samples=10):
    results = []
    for _ in range(n_samples):
        lr = 10 ** np.random.uniform(-4, 0)  # log-uniform between 1e-4 and 1
        mi = random.choice([500, 1000, 2000])
        x, val, iters = gradient_descent(f, grad_f, x0, lr, max_iter=mi)
        results.append({"method": "random", "lr": lr, "max_iter": mi, "final_val": val, "iters": iters})
    return sorted(results, key=lambda x: x["final_val"])



# 5. LLM-Guided Search (Manual or via API)


def llm_suggest(prompt_text=None):
    """
    Placeholder for LLM suggestions.
    If you have OpenAI API access, you can uncomment and use the real call below.
    For now, it loads a mock suggestion.
    """
    # --- Example using OpenAI API ---
    # import openai
    # response = openai.ChatCompletion.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": prompt_text}],
    # )
    # text = response["choices"][0]["message"]["content"]
    # try:
    #     params = json.loads(text)
    # except:
    #     params = {"learning_rate": 0.01, "max_iter": 1000}

    # --- Mock version (replace with actual GPT-4 call later) ---
    params = {"learning_rate": 0.02, "max_iter": 1200}
    return params


def llm_search(f, grad_f, x0, prompt_text):
    params = llm_suggest(prompt_text)
    x, val, iters = gradient_descent(f, grad_f, x0, lr=params["learning_rate"], max_iter=params["max_iter"])
    return {"method": "llm", "lr": params["learning_rate"], "max_iter": params["max_iter"], "final_val": val, "iters": iters}



# 6. Experiment Runner


def run_experiment():
    functions = [
        ("Rastrigin", rastrigin, rastrigin_grad),
        ("Rosenbrock", rosenbrock, rosenbrock_grad),
    ]
    all_results = []

    for name, f, grad_f in functions:
        print(f"\n🔹 Running on {name} function")

        # Random initialization in R^2
        x0 = np.random.uniform(-2, 2, size=2)

        grid_results = grid_search(f, grad_f, x0)
        random_results = random_search(f, grad_f, x0)
        llm_prompt = f"""
        You are optimizing the {name} function using gradient descent.
        Suggest learning_rate and max_iter values that minimize it efficiently.
        Output JSON: {{"learning_rate": ..., "max_iter": ...}}
        """
        llm_result = llm_search(f, grad_f, x0, llm_prompt)

        best_grid = grid_results[0]
        best_random = random_results[0]
        best_llm = llm_result

        # Combine best results
        combined = [best_grid, best_random, best_llm]
        for c in combined:
            c["function"] = name
            all_results.append(c)

        # Print quick summary
        for c in combined:
            print(f"{c['method'].upper()} → lr={c['lr']:.4f}, val={c['final_val']:.6f}, iters={c['iters']}")

    return all_results

# Visualization

def visualize_results(results):
    import pandas as pd
    df = pd.DataFrame(results)

    for func in df["function"].unique():
        sub = df[df["function"] == func]
        plt.figure()
        plt.bar(sub["method"], sub["final_val"])
        plt.title(f"{func} Function - Best Final Value by Method")
        plt.ylabel("Final Function Value (Lower = Better)")
        plt.show()

    print("\n--- Summary Table ---")
    print(df[["function", "method", "lr", "max_iter", "final_val", "iters"]])

if __name__ == "__main__":
    results = run_experiment()
    visualize_results(results)
