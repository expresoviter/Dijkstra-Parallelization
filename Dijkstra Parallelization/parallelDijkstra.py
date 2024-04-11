import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

# CUDA kernel
kernel = """
    __global__ void dijkstra(float *d, int *p, int *visited, int n, int start)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;

        if(i < n)
        {
            if(visited[i] == 0)
            {
                int j;
                for(j = 0; j < n; j++)
                {
                    if(visited[j] == 1)
                    {
                        int weight = p[i*n + j];
                        if(weight != 0 && d[j] + weight < d[i])
                        {
                            d[i] = d[j] + weight;
                            visited[i] = 1;
                        }
                        else
                        {
                            visited[i] = 1;
                        }
                    }
                }
            }
        }
    }
"""


def parallel_dijkstra_gpu(graph, start, n_threads=8):
    n = graph.shape[0]
    graph = graph.flatten()
    d = np.full(n, np.inf)
    d[start] = 0
    visited = np.zeros(n)
    visited[start] = 1

    d_gpu = gpuarray.to_gpu(d.astype(np.float32))
    p_gpu = gpuarray.to_gpu(graph.astype(np.int32))
    visited_gpu = gpuarray.to_gpu(visited.astype(np.int32))

    mod = SourceModule(kernel)
    func = mod.get_function("dijkstra")

    block_size = (n_threads, 1, 1)
    grid_size = (int(np.ceil(n / block_size[0])), 1)
    for i in range(n):
        func(d_gpu, p_gpu, visited_gpu, np.int32(n), np.int32(start), block=block_size, grid=grid_size)

    d = d_gpu.get()

    return d


def test_parallel_dijkstra_gpu(graph, n_threads):
    start = 0
    distance = parallel_dijkstra_gpu(graph, start, n_threads)
    return distance


def execution_time(graph, n_threads):
    time1 = time.time()
    dist = test_parallel_dijkstra_gpu(graph, n_threads)
    time1 = time.time() - time1
    return dist, time1
