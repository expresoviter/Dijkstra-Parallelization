import heapq
import time
import numpy as np


def test_serial_dijkstra(n, graph):
    start = 0
    distance, path = dijkstra(n, graph, start)
    # print("Шляхи для послідовного алгоритму:", path)
    return distance


def execution_time(n, graph):
    time1 = time.time()
    dist = test_serial_dijkstra(n, graph)
    time1 = time.time() - time1
    return dist, time1


def dijkstra(n, adj_list, src):
    distances = np.full(n, np.inf)
    distances[src] = 0

    minHeap = []
    heapq.heapify(minHeap)
    heapq.heappush(minHeap, (0, src))
    paths = [[] for _ in range(n)]
    paths[src].append(src)

    while minHeap:
        cur_distance, node = heapq.heappop(minHeap)
        neighbors = adj_list[node]
        for k in neighbors:
            neighbor_node, weight = k[0], k[1]
            possible_update = weight + cur_distance
            if possible_update < distances[neighbor_node]:
                distances[neighbor_node] = possible_update
                heapq.heappush(minHeap, (possible_update, neighbor_node))
                paths[neighbor_node] = paths[node] + [neighbor_node]
    return distances, paths
