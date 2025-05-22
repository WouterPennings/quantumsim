import quantumsim_performante.quantumutil as QuantumUtil
from maxcut import show_graph, show_graph_partition, cut_size, compute_partition, compute_average_cut_size, maxcut_QAOA, maxcut_bruteforce
import networkx as nx

G = nx.watts_strogatz_graph(20, 4, 0.1)
nodes = G.nodes
edges = G.edges

show_graph(G.nodes, G.edges)

result = maxcut_QAOA(nodes, edges, noisy=False, use_GPU=True, use_lazy=True)

average_cut_size = compute_average_cut_size(nodes, edges, result)
partition = compute_partition(nodes, edges, result)
partition_cut_size = cut_size(edges, partition)

print(f"Partition {partition} has cut size {partition_cut_size}")
print(f"Average cut size is {average_cut_size}")
# QuantumUtil.histogram_of_classical_states(result)
show_graph_partition(nodes, edges, partition)