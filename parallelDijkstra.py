import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time

MOD = SourceModule("""
    #include <stdio.h>
    __global__ void Find_Vertex(int *graph, int *visited, int *distances, int *updlen) {
        int u = threadIdx.x;
        int n = blockDim.x;
        if (visited[u] == 0) {
            visited[u] = 1;
            int v;
            for (v = 0; v < n; v++) {
                int w = graph[u*n + v];
                if (w < 1000000) {
                    if (updlen[v] > distances[u] + w) {
                        updlen[v] = distances[u] + w;
                    }
                }
            }
        }
    }

    __global__ void Update_Paths(int *visited, int *distances, int *updlen)
	{
         int u = threadIdx.x;
         if(distances[u] > updlen[u])
           {
                distances[u] = updlen[u];
                visited[u] = 0;
            }

         updlen[u] = distances[u];
        }

    """)


def cuda_execution(graph_matrix, vertices, visited, distances, updation, numThreads=8):
    graph_matrix_gpu = cuda.mem_alloc(graph_matrix.size * graph_matrix.dtype.itemsize)
    visited_gpu = cuda.mem_alloc(visited.size * visited.dtype.itemsize)
    distances_gpu = cuda.mem_alloc(distances.size * distances.dtype.itemsize)
    updlen_gpu = cuda.mem_alloc(updation.size * updation.dtype.itemsize)

    cuda.memcpy_htod(graph_matrix_gpu, graph_matrix)
    cuda.memcpy_htod(visited_gpu, visited)
    cuda.memcpy_htod(distances_gpu, distances)
    cuda.memcpy_htod(updlen_gpu, updation)

    _find_vertex = MOD.get_function("Find_Vertex")
    _update_paths = MOD.get_function("Update_Paths")

    for i in range(vertices.size):
        _find_vertex(
            graph_matrix_gpu,
            visited_gpu,
            distances_gpu,
            updlen_gpu,
            block=(numThreads, 1, 1))
        for k in range(vertices.size):
            _update_paths(visited_gpu, distances_gpu, updlen_gpu, block=(numThreads, 1, 1))

    cuda.memcpy_dtoh(visited, visited_gpu)
    cuda.memcpy_dtoh(distances, distances_gpu)
    cuda.memcpy_dtoh(updation, updlen_gpu)


def test_parallel_dijkstra_gpu(n, graph, n_threads):
    start = 0
    distance = parallel_dijkstra(n, graph, start, n_threads)
    return distance


def execution_time(n, graph, n_threads):
    time1 = time.time()
    dist = test_parallel_dijkstra_gpu(n, graph, n_threads)
    time1 = time.time() - time1
    return dist, time1


def parallel_dijkstra(n, adj_list, source, n_threads=8):
    graph_matrix = np.zeros((n, n), dtype=np.int32)
    for vertex, neighbors in adj_list.items():
        for neighbor, weight in neighbors:
            graph_matrix[vertex][neighbor] = weight


    vertices = np.array(list(range(n))).astype(np.int32)
    visited = np.zeros_like(vertices).astype(np.int32)
    distances = np.zeros_like(vertices).astype(np.int32)
    updation = np.zeros_like(vertices).astype(np.int32)
    wmf = graph_matrix.flatten().astype(np.int32)

    for i in range(vertices.size):
        if i == source:
            distances[i] = 0
        else:
            distances[i] = 1000000
            updation[i] = distances[i]

    cuda_execution(wmf, vertices, visited, distances, updation, n_threads)

    return distances
