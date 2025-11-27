import json
import os
import matplotlib.pyplot as plt

RESULTS_PATH = os.path.join("results", "benchmark_results.json")

def load_results(path=RESULTS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}. "
                                f"Run performance_analysis.py first.")
    with open(path, "r") as f:
        return json.load(f)

def plot_speedup(results):
    # Group by data_size
    by_size = {}
    for r in results:
        by_size.setdefault(r["data_size"], []).append(r)

    plt.figure(figsize=(8, 5))
    for data_size, entries in by_size.items():
        entries = sorted(entries, key=lambda x: x["processes"])
        procs = [e["processes"] for e in entries]
        speedup = [e["speedup"] for e in entries]
        plt.plot(procs, speedup, marker="o", label=f"N={data_size}")

    plt.xlabel("Number of MPI processes")
    plt.ylabel("Speedup (T_serial / T_parallel)")
    plt.title("Parallel KD-Tree Speedup vs. Process Count")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results", "speedup_plot.png"))
    plt.show()

def plot_efficiency(results):
    by_size = {}
    for r in results:
        by_size.setdefault(r["data_size"], []).append(r)

    plt.figure(figsize=(8, 5))
    for data_size, entries in by_size.items():
        entries = sorted(entries, key=lambda x: x["processes"])
        procs = [e["processes"] for e in entries]
        eff = [e["efficiency"] for e in entries]
        plt.plot(procs, eff, marker="o", label=f"N={data_size}")

    plt.xlabel("Number of MPI processes")
    plt.ylabel("Parallel efficiency (%)")
    plt.title("Parallel KD-Tree Efficiency vs. Process Count")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results", "efficiency_plot.png"))
    plt.show()

if __name__ == "__main__":
    results = load_results()
    plot_speedup(results)
    plot_efficiency(results)

