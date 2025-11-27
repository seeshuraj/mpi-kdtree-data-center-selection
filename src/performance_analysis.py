from mpi4py import MPI
import numpy as np
import time
from parallel_kdtree_mpi import ParallelKDTree
from serial_kdtree import SerialKDTree
import json

def benchmark():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Test parameters
    data_sizes = [500, 1000, 5000]
    dimensions = 4
    results = []
    
    for data_size in data_sizes:
        if rank == 0:
            data = np.random.rand(data_size, dimensions)
            queries = np.random.rand(10, dimensions)
            
            # Serial baseline
            serial_start = time.time()
            serial_tree = SerialKDTree(data)
            serial_distances, serial_indices = serial_tree.query(queries, k=5)
            serial_time = time.time() - serial_start
        else:
            data = None
            queries = None
            serial_time = None
        
        # Broadcast to all
        data = comm.bcast(data if rank == 0 else None, root=0)
        queries = comm.bcast(queries if rank == 0 else None, root=0)
        
        # Parallel execution
        parallel_start = time.time()
        tree = ParallelKDTree(data)
        local_data = tree.distribute_data()
        tree.build_local_tree(local_data)
        parallel_distances, parallel_indices = tree.parallel_query(queries, k=5)
        parallel_time = time.time() - parallel_start
        
        if rank == 0:
            speedup = serial_time / parallel_time
            efficiency = speedup / size * 100
            profile = tree.profile_communication()
            
            result = {
                'data_size': data_size,
                'processes': size,
                'serial_time': serial_time,
                'parallel_time': parallel_time,
                'speedup': speedup,
                'efficiency': efficiency,
                'comm_overhead_percent': profile['comm_overhead']
            }
            results.append(result)
            
            print(f"\n=== Data Size: {data_size} ===")
            print(f"Serial time: {serial_time:.4f}s")
            print(f"Parallel time ({size} processes): {parallel_time:.4f}s")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Efficiency: {efficiency:.2f}%")
            print(f"Communication overhead: {profile['comm_overhead']:.2f}%")
    
    if rank == 0:
        with open('results/benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n✓ Results saved to results/benchmark_results.json")

if __name__ == "__main__":
    benchmark()

