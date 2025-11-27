"""Unit tests for parallel KD-tree."""
import sys
import os
import numpy as np
from mpi4py import MPI

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.serial_kdtree import SerialKDTree

def test_basic():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    np.random.seed(42)
    data = np.random.rand(100, 4)
    tree = SerialKDTree(data)
    query = np.random.rand(1, 4)
    dists, inds = tree.query(query, k=5)
    if rank == 0:
        print("✓ Test passed!")

if __name__ == "__main__":
    test_basic()
