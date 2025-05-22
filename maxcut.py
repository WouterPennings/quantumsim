import random
from quantumsim_performante.quantumsim_performante import Circuit, NoisyCircuit
import quantumsim_performante.quantumutil as QuantumUtil
import numpy as np
import networkx as nx
from collections import Counter

def compute_partition(nodes:list, edges:list, result:list) -> dict:
    """
    Compute partition from the most occurring measurement

    Parameters:
    nodes  : list of nodes
    edges  : list of edges
    result : list of strings containing measurements

    Returns:
    Partition corresponding to most occurring result
    """

    # Count occurrences of each string in result
    counter = Counter(result)

    # Get the most occurring string from result
    most_common_string = counter.most_common(1)[0][0]

    # Determine partition corresponding to the most occurring string
    bit_string = most_common_string[1:-1]
    partition = {node: 0 if bit_string[node] == '0' else 1 for node in nodes}

    # Return partition corresponding to the most occurrring string in result
    return partition

def show_graph(nodes, edges, seed=42):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Layout to draw the graph
    pos = nx.spring_layout(G, seed=seed)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800, edgecolors='black')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)

def show_graph_partition(nodes, edges, partition, seed=42):
    """
    Show graph partition

    Parameters:
    nodes        : list of nodes 
    edges        : list of edges
    partition    : graph partition
    """

    # Construct the graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Layout to draw the graph
    pos = nx.spring_layout(G, seed=seed)

    # Colors for the nodes
    colors = ['skyblue' if partition[node] == 0 else 'lightgreen' for node in G.nodes()]

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800, edgecolors='black')
    nx.draw_networkx_labels(G, pos)

    # Divide the edges between `cut` edges and `same` edges
    cut_edges = [(u, v) for u, v in G.edges() if partition[u] != partition[v]]
    same_edges = [(u, v) for u, v in G.edges() if partition[u] == partition[v]]

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=same_edges, width=1.5, style='solid', edge_color='black')
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=2.5, style='dashed', edge_color='red')

# Initialize a random partition (dict: node -> 0 or 1)
def random_partition(nodes:list) -> dict:
    """
    Constructs a random partition for a given list of nodes

    Parameters:
    nodes : list of nodes 

    Returns:
    random partition
    """
    return {node: random.randint(0, 1) for node in nodes}

def cut_size(edges, partition):
    return sum(1 for u, v in edges if partition[u] != partition[v])

def compute_average_cut_size(nodes:list, edges:list, result:list) -> float:
    """
    Compute average cut size

    Parameters:
    nodes  : list of nodes
    edges  : list of edges
    result : list of measurements

    Returns:
    average cut size
    """
    occurrences = Counter(result)
    total_nr_occurrences = len(result)
    sum_cut_size = 0
    for string, count in occurrences.items():
        bit_string = string[1:-1]
        partition = {node: bit_string[node] for node in nodes}
        partition_cut_size = cut_size(edges, partition)
        sum_cut_size += partition_cut_size*count

    average_cut_size = sum_cut_size/total_nr_occurrences
    return average_cut_size

def maxcut_bruteforce(nodes, edges):
    """
    Finds an optimal solution of the max-cut problem and returns all partitions
    with their cut sizes, and the index of the partition with the maximum cut size.

    Parameters:
    nodes       : list of nodes
    edges       : list of edges

    Returns:
    all_partitions_data : list of tuples, where each tuple is (partition, cut_size)
    max_cut_index       : index in all_partitions_data corresponding to the max cut
    """
    nr_nodes = len(nodes)
    nr_partitions = 2**nr_nodes
    
    all_partitions_data = []
    max_cut_size = -1  # Initialize with a value lower than any possible cut size
    max_cut_index = -1

    # We iterate from 0 to nr_partitions - 1 to include all possible partitions.
    # The problem statement for max-cut usually includes trivial partitions (all nodes in one set).
    # If you specifically want to exclude trivial partitions (all nodes in one set),
    # you can iterate from 1 to nr_partitions - 2 as in your original code.
    # For a comprehensive list, iterating from 0 to nr_partitions - 1 is generally better.
    for i in range(nr_partitions):
        binary_digits = format(i, f'0{nr_nodes}b')
        partition = {}
        for n_idx, n in enumerate(nodes):
            partition[n] = 0 if binary_digits[n_idx] == '0' else 1 # Corrected index access

        current_cut_size = cut_size(edges, partition)
        
        # Store the current partition and its cut size
        all_partitions_data.append((partition.copy(), current_cut_size))
        
        # Check if this is the new maximum cut
        if current_cut_size > max_cut_size:
            max_cut_size = current_cut_size
            max_cut_index = len(all_partitions_data) - 1 # Store the index of the current partition
            
    return all_partitions_data, max_cut_index

# Create the QAOA circuit
def qaoa_circuit(gamma:list[float], beta:list[float], nodes:list, edges:list, p:int, use_cache=True, use_GPU=False, use_lazy=False) -> Circuit:
    """
    Creates a quantum circuit of p layers for the Quantum Approximate Optimiziation Algorithm

    Parameters:
    gamma : list of length p containing values for gamma, 0 < gamma < pi
    beta  : list of length p containing values for beta, 0 < beta < pi
    nodes : list of nodes 
    edges : list of edges
    p     : number of layers

    Returns:
    QAOA circuit with p layers
    """

    # Consistency check
    if len(gamma) != p or len(beta) != p:
        raise ValueError(f"Lists gamma and beta should be of length p = {p}")
    
    # Create circuit witn n qubits, where n is the number of nodes
    n = len(nodes)
    circuit = Circuit(n, use_cache=use_cache, use_GPU=use_GPU, use_lazy=use_lazy)
    
    # Initialize circuit by applying the Hadamard gate to all qubits
    for q in range(n):
        circuit.hadamard(q)

    # Construct p alternating cost and mixer layers
    for i in range(p):
    
        # Construct cost layer with parameter gamma[i]
        for edge in edges:
            circuit.cnot(edge[0], edge[1])
            circuit.rotate_z(2 * gamma[i], edge[1])
            circuit.cnot(edge[0], edge[1])
        
        # Construct mixer layer with parameter beta[i]
        for q in range(n):
            circuit.rotate_x(2 * beta[i], q)
    
    #return circuit
    return circuit

# Create the QAOA noisycircuit
def qaoa_circuit_noisy(gamma:list[float], beta:list[float], nodes:list, edges:list, p:int, use_cache=True, use_GPU=False) -> Circuit:
    """
    Creates a quantum circuit of p layers for the Quantum Approximate Optimiziation Algorithm

    Parameters:
    gamma : list of length p containing values for gamma, 0 < gamma < pi
    beta  : list of length p containing values for beta, 0 < beta < pi
    nodes : list of nodes 
    edges : list of edges
    p     : number of layers

    Returns:
    QAOA circuit with p layers
    """

    # Consistency check
    if len(gamma) != p or len(beta) != p:
        raise ValueError(f"Lists gamma and beta should be of length p = {p}")
    
    # Create circuit witn n qubits, where n is the number of nodes
    n = len(nodes)
    circuit = NoisyCircuit(n, use_cache=use_cache, use_GPU=use_GPU)
    
    # Initialize circuit by applying the Hadamard gate to all qubits
    for q in range(n):
        circuit.noisy_hadamard(q)

    # Construct p alternating cost and mixer layers
    for i in range(p):
    
        # Construct cost layer with parameter gamma[i]
        for edge in edges:
            circuit.noisy_cnot(edge[0], edge[1])
            circuit.rotate_z(2 * gamma[i], edge[1])
            circuit.noisy_cnot(edge[0], edge[1])
        # Construct mixer layer with parameter beta[i]
        for q in range(n):
            circuit.rotate_x(2 * beta[i], q)
    
    #return circuit
    return circuit

def maxcut_QAOA(nodes:list, edges:list, noisy=False, p=50, gamma_max=1.0, beta_max=1.0, nr_measurements=200000, use_GPU=False, use_cache=False, use_lazy=False) -> list:
    gamma = []
    beta = []
    for layer in range(p):
        gamma.append(gamma_max * (layer/p))
        beta.append(beta_max * ((p - layer)/p))
    if noisy:
        circuit = qaoa_circuit_noisy(gamma, beta, nodes, edges, p=p, use_GPU=use_GPU, use_cache=use_cache)
    else:
        circuit = qaoa_circuit(gamma, beta, nodes, edges, p=p, use_GPU=use_GPU, use_cache=use_cache, use_lazy=use_lazy)
    
    result = QuantumUtil.measure_circuit(circuit, nr_measurements=nr_measurements)
    return result