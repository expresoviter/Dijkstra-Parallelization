from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLineEdit, QLabel, QWidget
import parallelDijkstra, serialDijkstra
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import random
import time


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Execution time")
        self.inputLabel = QLabel("Enter the number of nodes:", self)
        self.inputField = QLineEdit(self)
        self.okButton = QPushButton("OK", self)
        self.executionTimeLabel = QLabel(self)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.okButton.clicked.connect(self.run_algorithm)

        layout = QVBoxLayout()
        layout.addWidget(self.inputLabel)
        layout.addWidget(self.inputField)
        layout.addWidget(self.okButton)
        layout.addWidget(self.executionTimeLabel)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def run_algorithm(self):
        nodeNumber = int(self.inputField.text())
        graph = generate_complex_graph(nodeNumber, 5, 15)
        distS, serialExecutionTime = serialDijkstra.execution_time(graph)
        distP, cudaExecutionTime = parallelDijkstra.execution_time(graph, 128)
        self.executionTimeLabel.setText(f"Час послідовного виконання: {serialExecutionTime:.3f}сек. \n" \
                                        f"Час паралельного виконання: {cudaExecutionTime:.3f}сек. \n" \
                                        f"Чи результати збігаються: {np.array_equal(distS, distP)}")
        self.update_chart(cudaExecutionTime, serialExecutionTime)

    def update_chart(self, cudaExecutionTime, serialExecutionTime):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        height = [cudaExecutionTime, serialExecutionTime]
        tick_label = ['Parallel - CUDA', 'Serial - 1xCPU']

        ax.bar(tick_label, height, color=['green', 'red'])
        ax.set_xlabel('Implementation of algorithm')
        ax.set_ylabel('Time (sec)')
        self.canvas.draw()


def generate_complex_graph(n, min_weight=1, max_weight=10):
    graph = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i != j:
                weight = random.randint(min_weight, max_weight)
                graph[i][j] = weight
                graph[j][i] = weight
    print("Кількість вершин:", n)
    print("Згенерована матриця інцидентности:\n", graph)
    return graph


def generate_test_graph(n, small_distance_nodes_ratio):
    graph = np.zeros((n, n))
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
                graph[i][j] = weight
                graph[j][i] = weight

    print("Згенерована матриця інцидентности:\n", graph)
    return graph


def run_sequential_test():
    nodeNumber = int(input("Введіть кількість вершин: "))
    graph = generate_test_graph(nodeNumber, 0.1)
    distS, serialExecutionTime = serialDijkstra.execution_time(graph)
    print("Час виконання алгоритму: ", serialExecutionTime)
    print("Відстані:\n", distS)

def run_comparison_test():
    nodeNumber = int(input("Введіть кількість вершин: "))
    graph = generate_complex_graph(nodeNumber)
    for i in range(20):
        distS, serialExecutionTime = serialDijkstra.execution_time(graph)
    print("Час виконання послідовного алгоритму: ", serialExecutionTime)
    for i in [8,16,32,128]:
        for k in range(20):
            distP, cudaExecutionTime = parallelDijkstra.execution_time(graph, i)
        # print("Відстані послідовно:",distS, "\nВідстані паралельно:",distP)
        print(f"Час виконання паралельного алгоритму з {i} потоків: ", cudaExecutionTime)
        print("Чи результати послідовного та паралельного алгоритмів збігаються:", np.array_equal(distS, distP))


def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    #run_sequential_test()
    #run_comparison_test()
    main()
