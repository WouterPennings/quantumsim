import numpy as np
import math
import cmath
import time
from timeit import timeit
import scipy.sparse as sparse

class Dirac:
    """
    Functions for the Dirac notation to describe (quantum) states and (quantum) operators.
    """
    @staticmethod
    def ket(N, a):
        """
        `|a>` is called 'ket' and represents a column vector with `1` in entry `a` and `0` everywhere else.
        """
        ket = np.zeros((N, 1))
        ket[a, 0] = 1
        return ket

    @staticmethod
    def bra(N, a):
        """
        `<a|` is called 'bra' and represents a row vector with `1` in entry `a` and `0` everywhere else.
        """
        bra = np.zeros((1, N))
        bra[0, a] = 1
        return bra

    @staticmethod
    def bra_ket(N, a, b):
        """
        `<a||b>` is the inner product of `<a|` and `|b>`, which is `1` if `a == b` and `0` `if a != b`.
        """
        bra = Dirac.bra(N, a)
        ket = Dirac.ket(N, b)
        return np.inner(bra, ket.T)

    @staticmethod
    def ket_bra(N, a, b):
        """
        `|a><b|` is the outer product of `|a>` and `<b|`, which is a matrix with `1` in entry (a,b) and `0` everywhere else.
        """
        ket = Dirac.ket(N, a)
        bra = Dirac.bra(N, b)
        return np.outer(ket, bra)
    
class QubitUnitaryOperation:
    """
    Functions to obtain 2 x 2 unitary matrices for unitary qubit operations.
    """    
    @staticmethod
    def get_identity():
        return np.array([[1, 0], [0, 1]], dtype=complex)
    
    @staticmethod
    def get_pauli_x():
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def get_pauli_y():
        return np.array([[0, complex(0,-1)], [complex(0,1), 0]], dtype=complex)
    
    @staticmethod
    def get_pauli_z():
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def get_hadamard():
        c = complex(1/np.sqrt(2), 0)
        return np.array([[c, c], [c, -c]], dtype=complex)
    
    @staticmethod
    def get_phase(theta):
        c = complex(np.cos(theta),np.sin(theta))
        return np.array([[1, 0], [0, c]], dtype=complex)
    
    @staticmethod
    def get_rotate_x(theta):
        sin = math.sin(theta/2)
        cos = math.cos(theta/2)
        return np.array([[cos, -1j * sin],[-1j * sin, cos]], dtype=complex)
    
    @staticmethod
    def get_rotate_y(theta):
        sin = math.sin(theta/2)
        cos = math.cos(theta/2)
        return np.array([[cos, -sin], [sin, cos]], dtype=complex)
    
    @staticmethod
    def get_rotate_z(theta):
        a = 0.5j * theta
        return np.array([[cmath.exp(-a), 0], [0, cmath.exp(a)]], dtype=complex)

class StateVector:
    """
    Class representing a quantum circuit of N qubits.
    """
    def __init__(self, N):
        self.N = N
        self.index = 0
        # WARNING: I HAVE FLATTEND THE STATEVECTOR
        # np.zeros(2**self.N, dtype=complex)
        # =============================================================
        # =============================================================
        # =============================================================
        self.state_vector = np.zeros(2**self.N, dtype=complex)
        # =============================================================
        # =============================================================
        # =============================================================
        self.state_vector[self.index] = 1

    def apply_unitary_operation(self, operation: sparse.coo_matrix):
        # Check if operation is a unitary matrix
        # if not np.allclose(np.eye(2**self.N), np.conj(operation.T) @ operation):
        #     raise ValueError("Input matrix is not unitary")

        self.state_vector = coo_spmv(operation.row, operation.col, operation.data, self.state_vector)

    def measure(self):
        probalities = np.square(np.abs(self.state_vector)).flatten()
        self.index = np.random.choice(len(probalities), p=probalities)

    def get_quantum_state(self):
        return self.state_vector
    
    def get_classical_state_as_string(self):
        return self.__state_as_string(self.index, self.N)
    
    def print(self):
        for i, val in enumerate(self.state_vector):
            print(f"{self.__state_as_string(i, self.N)} : {val[0]}")

    def __state_as_string(self, i,N):
        """
        Function to convert integer i, 0 <= i < N, to a quantum state in Dirac notation.
        """
        # Check if 0 <= i < 2^N
        if i < 0 or i >= 2**N:
            raise ValueError("Input i and N must satisfy 0 <= i < 2^N")
        
        binary_string = bin(i)
        state_as_string = binary_string[2:]
        state_as_string = state_as_string.zfill(N)
        return "|" + state_as_string + ">"


class CircuitUnitaryOperation:
    """
    Functions to obtain 2^N x 2^N unitary matrices for unitary operations on quantum circuits of N qubits.
    """
    @staticmethod
    def get_combined_operation_for_qubit(operation, q, N):
        identity = QubitUnitaryOperation.get_identity()

        combined_operation= sparse.coo_matrix(np.eye(1,1))

        print("Generating operation matrix: ", end="", flush=True)

        t1 = time.perf_counter()
        for i in range(0, N):
            if i == q:
                combined_operation = sparse.kron(combined_operation, operation)
            else:
                combined_operation = sparse.kron(combined_operation, identity)
        t2 = time.perf_counter()

        bytes = combined_operation.nnz * 4 * 2 + combined_operation.nnz * 16
        print(f"{round(t2-t1, 6)*1000}ms ({bytes:,} bytes)")

        # nnz = np.count_nonzero(combined_operation)
        # print(nnz)
        # print(f"{nnz}/{combined_operation.size} = {nnz/combined_operation.size*100}% sparseness")

        return combined_operation
    
    @staticmethod
    def get_combined_operation_for_identity(q, N):
        return np.array(np.eye(2**N), dtype=complex)
    
    @staticmethod
    def get_combined_operation_for_pauli_x(q, N):
        pauli_x = QubitUnitaryOperation.get_pauli_x()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(pauli_x, q, N)
    
    @staticmethod
    def get_combined_operation_for_pauli_y(q, N):
        pauli_y = QubitUnitaryOperation.get_pauli_y()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(pauli_y, q, N)
    
    @staticmethod
    def get_combined_operation_for_pauli_z(q, N):
        pauli_z = QubitUnitaryOperation.get_pauli_z()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(pauli_z, q, N)
    
    @staticmethod
    def get_combined_operation_for_hadamard(q, N):
        hadamard = QubitUnitaryOperation.get_hadamard()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(hadamard, q, N)
    
    @staticmethod
    def get_combined_operation_for_phase(theta, q, N):
        phase = QubitUnitaryOperation.get_phase(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(phase, q, N)
    
    @staticmethod
    def get_combined_operation_for_rotate_x(theta, q, N):
        rotate = QubitUnitaryOperation.get_rotate_x(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(rotate, q, N)
    
    @staticmethod
    def get_combined_operation_for_rotate_y(theta, q, N):
        rotate = QubitUnitaryOperation.get_rotate_y(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(rotate, q, N)
    
    @staticmethod
    def get_combined_operation_for_rotate_z(theta, q, N):
        rotate = QubitUnitaryOperation.get_rotate_z(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(rotate, q, N)
    
    @staticmethod
    def get_combined_operation_for_cnot(control, target, N):
        identity = QubitUnitaryOperation.get_identity()
        pauli_x = QubitUnitaryOperation.get_pauli_x()

        ket_bra_00 = Dirac.ket_bra(2,0,0)
        ket_bra_11 = Dirac.ket_bra(2,1,1)
        combined_operation_zero = sparse.coo_matrix(np.eye(1,1))
        combined_operation_one = sparse.coo_matrix(np.eye(1,1))
    
        print("Generating operation CNOT matrix: ", end="", flush=True)

        t1 = time.perf_counter()
        for i in range(0, N):
            if control == i:
                combined_operation_zero = sparse.kron(combined_operation_zero, ket_bra_00)
                combined_operation_one  = sparse.kron(combined_operation_one, ket_bra_11)
            elif target == i:
                combined_operation_zero = sparse.kron(combined_operation_zero, identity)
                combined_operation_one  = sparse.kron(combined_operation_one, pauli_x)
            else:
                combined_operation_zero = sparse.kron(combined_operation_zero, identity)
                combined_operation_one  = sparse.kron(combined_operation_one, identity)

        operation = sparse.coo_matrix(combined_operation_zero + combined_operation_one)
        t2 = time.perf_counter()

        bytes = operation.nnz * 4 * 2 + operation.nnz * 16
        print(f"{round(t2-t1, 6)*1000}ms ({bytes:,} bytes)")

        return operation
'''
Symbol for pi
'''
pi_symbol = '\u03c0'

class Circuit:
    """
    Class representing a quantum circuit of N qubits.
    """
    def __init__(self, N, cache=False):
        self.N = N
        self.state_vector = StateVector(self.N)
        self.quantum_states = [self.state_vector.get_quantum_state()]
        self.descriptions = []
        self.operations = []
        self.use_cache = cache
        self.operations_cache = {}

    def identity(self, q):
        key = key = ("identity", q)
        description = f"Hadamard on qubit {q}"

        if self.use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_identity(q, self.N)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def pauli_x(self, q):
        key = key = ("pauli_x", q)
        description = f"pauli_x on qubit {q}"

        if self.use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_x(q, self.N)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def pauli_y(self, q):
        key = key = ("pauli_y", q)
        description = f"pauli_y on qubit {q}"

        if self.use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_y(q, self.N)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def pauli_z(self, q):
        key = key = ("pauli_z", q)
        description = f"pauli_z on qubit {q}"

        if self.use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return
        
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_z(q, self.N)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def hadamard(self, q):
        key = ("hadamard", q)
        description = f"Hadamard on qubit {q}"

        if self.use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_hadamard(q, self.N)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def phase(self, theta, q):
        key = key = ("phase", theta, q)
        description = f"Phase with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"

        if self.use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_phase(theta, q, self.N)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def rotate_x(self, theta, q):
        key = key = ("rotate_x", theta, q)
        description = f"Rotate X with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"

        if self.use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_x(theta, q, self.N)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.cache_operation(key, combined_operation)
    
    def rotate_y(self, theta, q):
        key = key = ("rotate_y", theta, q)
        description = f"Rotate_y with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"

        if self.use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_y(theta, q, self.N)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.cache_operation(key, combined_operation)
    
    def rotate_z(self, theta, q):
        key = key = ("rotate_z", theta, q)
        description = f"Rotate_z with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"

        if self.use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_z(theta, q, self.N)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def cnot(self, control, target):
        key = key = ("cnot", control, target)
        description = f"CNOT with control qubit {control} and target qubit {target}"

        if self.use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_cnot(control, target, self.N)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def execute(self, print_state=False):
        self.state_vector = StateVector(self.N)
        if print_state:
            print("Initial quantum state")
            self.state_vector.print()

        for i, (operation, description) in enumerate(zip(self.operations, self.descriptions)):
            # print(f"Execute operation ({i+1}/{len(self.operations)}): ", end="", flush=True)
            t1 = time.perf_counter()
            self.state_vector.apply_unitary_operation(operation)
            self.quantum_states.append(self.state_vector.get_quantum_state())
            t2 = time.perf_counter()
            print(f"{round(t2-t1, 6)*1000}ms")

            if print_state:
                print(description)
                print(operation)
                print("Current quantum state")
                self.state_vector.print()

    def measure(self, print_state=False):
        self.state_vector.measure()
        if print_state:
            print("Measured state:")
            print(self.state_vector.get_classical_state_as_string())

    def get_classical_state_as_string(self):
        return self.state_vector.get_classical_state_as_string()

    def retrieve_operation_from_cache(self, key:tuple, description:str):
        if key in self.operations_cache:
            self.descriptions.append(description)
            self.operations.append(self.operations_cache[key])
            return True
        return False

    def cache_operation(self, key:tuple, operation:sparse.coo_matrix):
        print("Saving operation matrix to cache")
        self.operations_cache[key] = operation
        

def coo_spmv(rowIdx, colIdx, values, v):
    """
    Performs sparse matrix-vector multiplication using COO format.
    
    Parameters:
    - rowIdx (list[int]): Row indices of nonzero elements.
    - colIdx (list[int]): Column indices of nonzero elements.
    - values (list[float]): Nonzero values of the matrix.
    - v (numpy array): Dense vector for multiplication.
    
    Returns:
    - numpy array: Result vector y = A * v
    """
    out = np.zeros(len(v), dtype=np.result_type(values, v))  # Initialize output vector

    nnz = len(values)  # Number of nonzero elements

    for i in range(nnz):  # Iterate over nonzero elements
        out[rowIdx[i]] += values[i] * v[colIdx[i]]

    return out