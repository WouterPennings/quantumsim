import random
import math
from quantumsim import Circuit
import numpy as np
import networkx as nx

def show_graph_partition(nodes, edges, partition):
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
    pos = nx.spring_layout(G, seed=42)

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

# Flip one random node's partition
def neighbor(partition:dict, nodes:list) -> dict:
    """
    Constructs a neighbor partition for a given partition

    Parameters:
    partition : partition for which neighbor should be constructed
    nodes     : list of nodes 

    Returns:
    neighbor partition
    """
    neighbor_partition = partition.copy()
    node_to_flip = random.choice(nodes)
    neighbor_partition[node_to_flip] ^= 1  # Flip 0 <-> 1
    return neighbor_partition

# Simulated Annealing algorithm
def simulated_annealing(nodes:list, edges:list, initial_temp:float=10.0, cooling_rate:float=0.95, max_iter:int=100) -> tuple[dict, float, list]:
    """
    Finds an approximation of the optimal solution of the max-cut problem

    Parameters:
    nodes        : list of nodes 
    edges        : list of edges
    initial_temp : initial temperature
    cooling_rate : cooling rate to reduce temperature each step
    max_iter     : maximum number of iterations

    Returns:
    partition with maximum cut size
    cut size of this partition
    history of cut sizes
    """
    current_partition = random_partition(nodes)
    current_cut_size = cut_size(edges, current_partition)
    T = initial_temp
    max_cut_partition = current_partition.copy()
    max_cut_size = current_cut_size
    history_cut_size = np.zeros(max_iter)

    for step in range(max_iter):
        candidate_partition = neighbor(current_partition, nodes)
        candidate_cut_size = cut_size(edges, candidate_partition)
        delta = candidate_cut_size - current_cut_size
        if delta > 0 or random.random() < math.exp(delta / T):
            current_partition = candidate_partition
            current_cut_size = candidate_cut_size

            if current_cut_size > max_cut_size:
                max_cut_partition = current_partition.copy()
                max_cut_size= current_cut_size

        T *= cooling_rate
        history_cut_size[step] = current_cut_size

    return max_cut_partition, max_cut_size, history_cut_size

def cut_size(edges, partition):
    return sum(1 for u, v in edges if partition[u] != partition[v])

def maxcut_bruteforce(nodes, edges):
    """
    Finds an optimal solution of the max-cut problem

    Parameters:
    nodes        : list of nodes 
    edges        : list of edges

    Returns:
    partition with maximum cut size
    cut size of this partition
    """
    nr_nodes = len(nodes)
    nr_partitions = 2**nr_nodes
    max_cut_size = 0
    for p in range(1,nr_partitions-1):
        binary_digits = format(p, f'0{nr_nodes}b')
        partition = {}
        for n in nodes:
            partition[n] = 0 if binary_digits[n] == '0' else 1
        current_cut_size = cut_size(edges, partition)
        print(f"Partition {partition} has cut size {current_cut_size}")
        if current_cut_size > max_cut_size:
            max_cut_size = current_cut_size
            max_cut_partition = partition.copy()
        
    return max_cut_partition, max_cut_size

# Create the QAOA circuit
def qaoa_circuit(gamma:list[float], beta:list[float], nodes:list, edges:list, p:int) -> Circuit:
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
    circuit = Circuit(n)
    
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