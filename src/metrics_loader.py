import pandas as pd
import numpy as np

def load_metrics(filepath='data/metrics.csv'):
    """Load datacenter metrics from CSV."""
    df = pd.read_csv(filepath)
    return df.values

def generate_synthetic_metrics(n_datacenters=1000, seed=42):
    """Generate synthetic metrics for benchmarking."""
    np.random.seed(seed)
    return np.random.rand(n_datacenters, 4)
