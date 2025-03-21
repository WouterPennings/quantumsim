from quantumsim_minima import Circuit
from collections import Counter
import matplotlib.pyplot as plt
import time
# circuit = Circuit(2)
# circuit.hadamard(0)
# circuit.cnot(0,1)
# circuit.execute(print_state=True)
# circuit.measure(print_state=True)


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


circuit = Circuit(14)
circuit.hadamard(1)
circuit.cnot(0,1)

# t1 = time.perf_counter()
# circuit.execute()
# t2 = time.perf_counter()
# print(f"{round(t2-t1, 6)*1000}ms")
