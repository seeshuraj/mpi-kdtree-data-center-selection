from mpi4py import MPI
import numpy as np
from sklearn.neighbors import KDTree
import time

class ParallelKDTree:
    def __init__(self, data, metric_names=None):
        """
        Initialize parallel KD-Tree.
        
        Args:
            data: (N, D) array of points
            metric_names: Names of metrics/dimensions
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.data = data
        self.metric_names = metric_names or [f"dim_{i}" for i in range(data.shape[1])]
        self.local_tree = None
        self.global_tree = None
        self.comm_time = 0
        self.compute_time = 0
        
    def distribute_data(self):
        """Distribute data evenly across MPI processes."""
        if self.rank == 0:
            # Split data into chunks
            split_indices = np.array_split(np.arange(len(self.data)), self.size)
            local_data = self.data[split_indices[0]]
        else:
            local_data = None
        
        # Scatter data to all processes
        start_time = time.time()
        local_data = self.comm.scatter(
            [self.data[split_indices[i]] if self.rank == 0 else None 
             for i in range(self.size)], 
            root=0
        )
        self.comm_time += time.time() - start_time
        
        return local_data
    
    def build_local_tree(self, local_data):
        """Build KD-Tree locally on each process."""
        start_time = time.time()
        self.local_tree = KDTree(local_data, leaf_size=30)
        self.compute_time += time.time() - start_time
        
        if self.rank == 0:
            print(f"Process {self.rank}: Built local tree with {len(local_data)} points")
    
    def parallel_query(self, query_points, k=1):
        """
        Perform parallel nearest-neighbor search.
        
        Args:
            query_points: (Q, D) array of query points
            k: Number of nearest neighbors
            
        Returns:
            distances: (Q, k) array of distances
            indices: (Q, k) array of global indices
        """
        # Broadcast query points to all processes
        start_time = time.time()
        query_points = self.comm.bcast(query_points, root=0)
        self.comm_time += time.time() - start_time
        
        # Local search
        start_time = time.time()
        local_distances, local_indices = self.local_tree.query(query_points, k=k)
        self.compute_time += time.time() - start_time
        
        # Gather results to root process
        start_time = time.time()
        all_distances = self.comm.gather(local_distances, root=0)
        all_indices = self.comm.gather(local_indices, root=0)
        self.comm_time += time.time() - start_time
        
        if self.rank == 0:
            # Find global best k neighbors
            global_distances = []
            global_indices = []
            
            for q in range(len(query_points)):
                # Collect all (distance, index, rank) tuples
                candidates = []
                for rank, (dists, inds) in enumerate(zip(all_distances, all_indices)):
                    for dist, idx in zip(dists[q], inds[q]):
                        candidates.append((dist, idx + rank * len(self.data) // self.size))
                
                # Sort and take top k
                candidates.sort()
                top_k = candidates[:k]
                global_distances.append([d[0] for d in top_k])
                global_indices.append([d[1] for d in top_k])
            
            return np.array(global_distances), np.array(global_indices)
        
        return None, None
    
    def profile_communication(self):
        """Return communication profiling information."""
        return {
            'rank': self.rank,
            'communication_time': self.comm_time,
            'computation_time': self.compute_time,
            'total_time': self.comm_time + self.compute_time,
            'comm_overhead': self.comm_time / (self.comm_time + self.compute_time) * 100 if (self.comm_time + self.compute_time) > 0 else 0
        }


# Example usage
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Generate sample datacenter metrics
    if rank == 0:
        np.random.seed(42)
        data = np.random.rand(1000, 4)  # 1000 datacenters, 4 metrics
        metrics = ['latency', 'storage', 'throughput', 'geo_distance']
        query = np.random.rand(1, 4)
        print(f"Starting parallel KD-Tree with {size} processes...")
        print(f"Data shape: {data.shape}, Metrics: {metrics}")
    else:
        data = None
        metrics = None
        query = None
    
    # Broadcast to all processes
    data = comm.bcast(data if rank == 0 else None, root=0)
    metrics = comm.bcast(metrics if rank == 0 else None, root=0)
    query = comm.bcast(query if rank == 0 else None, root=0)
    
    # Build parallel tree
    tree = ParallelKDTree(data, metrics)
    local_data = tree.distribute_data()
    tree.build_local_tree(local_data)
    
    # Query
    distances, indices = tree.parallel_query(query, k=5)
    
    # Profile
    profile = tree.profile_communication()
    comm.gather(profile, root=0)
    
    if rank == 0:
        print(f"\nNearest neighbors: indices={indices}, distances={distances}")
        print(f"Communication overhead: {profile['comm_overhead']:.2f}%")

