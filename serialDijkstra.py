import heapq
import math
import random
import time
import numpy as np


def generate_complex_graph(n, min_weight=1, max_weight=50, completeness=1.0):
    graph = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < completeness:
                weight = random.randint(min_weight, max_weight)
                graph[i].append((j, weight))
                graph[j].append((i, weight))

    print("Кількість вершин:", n)
    print("Згенерований список суміжності:")
    for vertex, neighbours in graph.items():
        if (vertex == 5 or vertex == n - 5) and n > 10:
            print("***************")
        if vertex < 5 or vertex > n - 5:
            print(f"Вершина {vertex}: {neighbours}")
    return graph


def generate_test_graph(n, small_distance_nodes_ratio):
    graph = {i: [] for i in range(n)}
    num_small_distance_nodes = int(n * small_distance_nodes_ratio)
    if num_small_distance_nodes == 0:
        num_small_distance_nodes = 1
    small_distance_nodes = random.sample(range(n), num_small_distance_nodes)

    for i in range(n):
        for j in range(i, n):
            if i != j:
                if i in small_distance_nodes or j in small_distance_nodes:
                    weight = random.randint(1, 10)
                else:
                    weight = random.randint(500, 1000)
                graph[i].append((j, weight))
                graph[j].append((i, weight))

    print("Згенерований список суміжности:")
    for vertex, neighbours in graph.items():
        if (vertex == 5 or vertex == n - 5 ) and n > 10:
            print("***************")
        if vertex < 5 or vertex > n - 5 or vertex in small_distance_nodes:
            print(f"Вершина {vertex}: {neighbours}")
    return graph


def test_serial_dijkstra(n, graph):
    start = 0
    distance, path = dijkstra(n, graph, start)
    #print("Шляхи для послідовного алгоритму:", path)
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


def run_sequential_test():
    nodeNumber = int(input("Введіть кількість вершин: "))
    graph = generate_complex_graph(nodeNumber)
    distS, serialExecutionTime = execution_time(nodeNumber, graph)
    print("Час виконання алгоритму: ", serialExecutionTime)
    #print("Відстані:\n", distS)


if __name__ == "__main__":
    run_sequential_test()
