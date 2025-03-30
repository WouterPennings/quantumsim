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

# Build the circuit
circuit = Circuit(2)
circuit.hadamard(0)
circuit.cnot(0,1)
# Execute and measure the circuit 100 times
result = []
for i in range(10000):
    circuit.execute()
    circuit.measure()
    result.append(circuit.get_classical_state_as_string())
# Print the array of classical states
print(result)

histogram_of_classical_states(result)

t2 = time.time()
print(f"Total running time: {t2-t1} seconds")
