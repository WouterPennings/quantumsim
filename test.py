from quantumsim_performante import Circuit
from collections import Counter
import matplotlib.pyplot as plt
import time

def histogram_of_classical_states(string_array):
    histogram = Counter(string_array)
    unique_strings = sorted(list(histogram.keys()))
    counts = [histogram[string] for string in unique_strings]
    plt.bar(unique_strings, counts)
    if len(histogram) > 8:
        plt.xticks(rotation='vertical')
    plt.xlabel('Classical states')
    plt.ylabel('Nr occurrences')
    plt.title('Number of occurrences of classical states')
    plt.show()

t1 = time.time()

n = 10
circuit = Circuit(n)
circuit.hadamard(0)
for i in range(n):
    circuit.hadamard(i)
circuit.pauli_y(0)
circuit.pauli_x(1)
circuit.pauli_z(6)
circuit.cnot(1, 2)
circuit.cnot(0, 1)
circuit.cnot(0, 2)
circuit.execute()

t2 = time.time()
print(f"Total running time: {t2-t1} seconds")

# t1 = time.perf_counter()
# circuit.execute()
# t2 = time.perf_counter()

# print(f"{round(t2-t1, 6)*1000}ms")