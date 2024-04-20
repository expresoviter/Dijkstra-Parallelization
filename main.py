import numpy as np
import random
import serialDijkstra
import parallelDijkstra


def generate_complex_graph(n, min_weight=1, max_weight=50, completeness=1.0):
    graph = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < completeness:
                weight = random.randint(min_weight, max_weight)
                graph[i].append((j, weight))
                graph[j].append((i, weight))

    print("Кількість вершин:", n)
    print("Згенерований список суміжности:")
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
        if (vertex == 5 or vertex == n - 5) and n > 10:
            print("***************")
        if vertex < 5 or vertex > n - 5 or vertex in small_distance_nodes:
            print(f"Вершина {vertex}: {neighbours}")
    return graph


def run_sequential_test():
    nodeNumber = int(input("Введіть кількість вершин: "))
    graph = generate_test_graph(nodeNumber, 0.1)
    distS, serialExecutionTime = serialDijkstra.execution_time(nodeNumber, graph)
    print("Час виконання алгоритму: ", serialExecutionTime)
    print("Відстані:\n", distS)


def run_comparison_test(warmup=20):
    n_threads = 32
    nodeNumber = int(input("Введіть кількість вершин: "))
    graph = generate_complex_graph(nodeNumber)
    for i in range(warmup):
        distS, serialExecutionTime = serialDijkstra.execution_time(nodeNumber, graph)
        distP, cudaExecutionTime = parallelDijkstra.execution_time(nodeNumber, graph, n_threads)
    print("Час виконання послідовного алгоритму: ", serialExecutionTime)
    #print("Відстані послідовно:", distS, "\nВідстані паралельно:", distP)
    print(f"Час виконання паралельного алгоритму з {n_threads} потоків: ", cudaExecutionTime)
    print("Чи результати послідовного та паралельного алгоритмів збігаються:", np.array_equal(distS, distP))


if __name__ == "__main__":
    #run_sequential_test()
    run_comparison_test(1)
