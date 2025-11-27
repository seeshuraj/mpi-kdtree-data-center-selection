import numpy as np
from sklearn.neighbors import KDTree
import time

class SerialKDTree:
    def __init__(self, data):
        """Build KD-tree on a single process."""
        self.data = data
        start = time.time()
        self.tree = KDTree(self.data, leaf_size=30)
        self.build_time = time.time() - start

    def query(self, query_points, k=1):
        """Return k nearest neighbours for each query point."""
        distances, indices = self.tree.query(query_points, k=k)
        return distances, indices

if __name__ == "__main__":
    np.random.seed(42)
    data = np.random.rand(1000, 4)      # 1000 points, 4D
    queries = np.random.rand(10, 4)     # 10 query points

    print("=== Serial KD-Tree Baseline ===")
    print(f"Data shape: {data.shape}")
    print(f"Query shape: {queries.shape}")

    tree = SerialKDTree(data)
    print(f"Build time: {tree.build_time:.4f} s")

    dists, inds = tree.query(queries, k=5)
    print("First query neighbours indices:", inds[0])
    print("First query neighbours distances:", dists[0])

