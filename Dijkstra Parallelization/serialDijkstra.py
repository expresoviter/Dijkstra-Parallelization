import numpy as np
import time


def dijkstra(graph, start):
    n = graph.shape[0]
    distances = np.full(n, np.inf)
    distances[start] = 0
    paths = [[] for _ in range(n)]
    paths[start] = [start]
    visited = np.zeros(n)
    visited[start] = 1
    for i in range(n):
        if visited[i]:
            for j in range(n):
                if graph[i][j] and distances[j] > distances[i] + graph[i][j]:
                    distances[j] = distances[i] + graph[i][j]
                    paths[j] = paths[i] + [j]
                    visited[j] = 1
    return distances, paths


def test_serial_dijkstra(graph):
    start = 0
    distance, path = dijkstra(graph, start)
    print("Шляхи для послідовного алгоритму:", path)
    return distance


def execution_time(graph):
    time1 = time.time()
    dist = test_serial_dijkstra(graph)
    time1 = time.time() - time1
    return dist, time1
