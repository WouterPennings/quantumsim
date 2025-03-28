import numpy as np
import math
import cmath
import time
from timeit import timeit
import scipy.sparse as sparse
import cupyx.scipy.sparse as cupysparse

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

        # NOTE: Statevector normally is column-based, I made a row-based.
        # np.zeros((2**self.N,1 ), dtype=complex)
        self.state_vector = np.zeros(2**self.N, dtype=complex)

        self.state_vector[self.index] = 1

    def apply_unitary_operation(self, operation: sparse.coo_matrix):
        # Check if operation is a unitary matrix
        # if not np.allclose(np.eye(2**self.N), np.conj(operation.T) @ operation):
        #     raise ValueError("Input matrix is not unitary")

        # NOTE: A row based statevector is roughly 15% faster than matrix-vector multiplication than a column based statevector
        # print(timeit(lambda: coo_spmv_flat(operation.row, operation.col, operation.data, self.state_vector.flatten()), number=100))
        # print(timeit(lambda: coo_spmv(operation.row, operation.col, operation.data, self.state_vector), number=100))

        self.state_vector = coo_spmv_flat(operation.row, operation.col, operation.data, self.state_vector)

    def measure(self):
        probalities = np.square(np.abs(self.state_vector))
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
    def get_combined_operation_for_qubit(operation, q, N, gpu=False):
        # Converting dense numpy matrixes to sparse COO scipy matrixes
        operation =  sparse.coo_matrix(operation)
        identity = sparse.coo_matrix(QubitUnitaryOperation.get_identity())
        combined_operation= sparse.coo_matrix(np.eye(1,1))

        # "Selecting" regular scipy sparse matrix kronecker product
        kron = sparse.kron

        print("Generating operation matrix: ", end="", flush=True)
        t1 = time.perf_counter()

        if gpu:
            # Copy data to device (GPU) memory from host (CPU)
            operation = cupysparse.coo_matrix(operation)
            identity = cupysparse.coo_matrix(identity)
            combined_operation = cupysparse.coo_matrix(combined_operation)

            # "Selecting" sparse matrix GPU-accelerated matrix kronecker product
            kron = cupysparse.kron  
    	
        # Actual computation of kronecker product, this is sort of a iterative problem.
        # Size of "combined_operation" grows exponentially
        # Every qubit makes the kronecker product twice as sparse
        # Computation is done on GPU based on whether parater "GPU" is "True"
        for i in range(0, N):
            if i == q:
                combined_operation = kron(combined_operation, operation)
            else:
                combined_operation = kron(combined_operation, identity)
        
        # Copy data back from device (GPU) to host (CPU)
        if gpu: combined_operation = combined_operation.get()
            
        t2 = time.perf_counter()

        bytes = combined_operation.nnz * 4 * 2 + combined_operation.nnz * 16
        print(f"{round(t2-t1, 6)*1000}ms ({bytes:,} bytes)")

        return combined_operation

    @staticmethod
    def get_combined_operation_for_identity(q, N, gpu=False):
        return np.array(np.eye(2**N), dtype=complex)
    
    @staticmethod
    def get_combined_operation_for_pauli_x(q, N, gpu=False):
        pauli_x = QubitUnitaryOperation.get_pauli_x()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(pauli_x, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_pauli_y(q, N, gpu=False):
        pauli_y = QubitUnitaryOperation.get_pauli_y()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(pauli_y, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_pauli_z(q, N, gpu=False):
        pauli_z = QubitUnitaryOperation.get_pauli_z()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(pauli_z, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_hadamard(q, N, gpu=False):
        hadamard = QubitUnitaryOperation.get_hadamard()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(hadamard, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_phase(theta, q, N, gpu=False):
        phase = QubitUnitaryOperation.get_phase(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(phase, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_rotate_x(theta, q, N, gpu=False):
        rotate = QubitUnitaryOperation.get_rotate_x(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(rotate, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_rotate_y(theta, q, N, gpu=False):
        rotate = QubitUnitaryOperation.get_rotate_y(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(rotate, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_rotate_z(theta, q, N, gpu=False):
        rotate = QubitUnitaryOperation.get_rotate_z(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(rotate, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_cnot(control, target, N, gpu=False):
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
    def __init__(self, N, cache=False, GPU=False, lazy=False):
        self.N = N
        self.state_vector = StateVector(self.N)
        self.quantum_states = [self.state_vector.get_quantum_state()]
        self.descriptions = []
        self.operations = []
        self.__use_gpu = GPU
        self.__lazy_evaluation= lazy
        self.__use_cache = cache
        self.__operations_cache = {}

    def identity(self, q):
        key = key = ("identity", q)
        description = f"Hadamard on qubit {q}"

        if self.__use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_identity(q, self.N, gpu=self.__use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.__use_cache:
            self.cache_operation(key, combined_operation)

    def pauli_x(self, q):
        key = key = ("pauli_x", q)
        description = f"pauli_x on qubit {q}"

        if self.__use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_x(q, self.N, gpu=self.__use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.__use_cache:
            self.cache_operation(key, combined_operation)

    def pauli_y(self, q):
        key = key = ("pauli_y", q)
        description = f"pauli_y on qubit {q}"

        if self.__use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_y(q, self.N, gpu=self.__use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.__use_cache:
            self.cache_operation(key, combined_operation)

    def pauli_z(self, q):
        key = key = ("pauli_z", q)
        description = f"pauli_z on qubit {q}"

        if self.__use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return
        
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_z(q, self.N, gpu=self.__use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.__use_cache:
            self.cache_operation(key, combined_operation)

    def hadamard(self, q):
        key = ("hadamard", q)
        description = f"Hadamard on qubit {q}"

        if self.__use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_hadamard(q, self.N, gpu=self.__use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.__use_cache:
            self.cache_operation(key, combined_operation)

    def phase(self, theta, q):
        key = key = ("phase", theta, q)
        description = f"Phase with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"

        if self.__use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_phase(theta, q, self.N, gpu=self.__use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.__use_cache:
            self.cache_operation(key, combined_operation)

    def rotate_x(self, theta, q):
        key = key = ("rotate_x", theta, q)
        description = f"Rotate X with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"

        if self.__use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_x(theta, q, self.N, gpu=self.__use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.__use_cache:
            self.cache_operation(key, combined_operation)
    
    def rotate_y(self, theta, q):
        key = key = ("rotate_y", theta, q)
        description = f"Rotate_y with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"

        if self.__use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_y(theta, q, self.N, gpu=self.__use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.__use_cache:
            self.cache_operation(key, combined_operation)
    
    def rotate_z(self, theta, q):
        key = key = ("rotate_z", theta, q)
        description = f"Rotate_z with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"

        if self.__use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_z(theta, q, self.N, gpu=self.__use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.__use_cache:
            self.cache_operation(key, combined_operation)

    def cnot(self, control, target):
        key = key = ("cnot", control, target)
        description = f"CNOT with control qubit {control} and target qubit {target}"

        if self.__use_cache and self.retrieve_operation_from_cache(key, description):
            print("retrieved operation from cache")
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_cnot(control, target, self.N, gpu=self.__use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        if self.__use_cache:
            self.cache_operation(key, combined_operation)

    def execute(self, print_state=False):
        self.state_vector = StateVector(self.N)
        if print_state:
            print("Initial quantum state")
            self.state_vector.print()

        for i, (operation, description) in enumerate(zip(self.operations, self.descriptions)):
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
        if key in self.__operations_cache:
            self.descriptions.append(description)
            self.operations.append(self.__operations_cache[key])
            return True
        return False

    def cache_operation(self, key:tuple, operation:sparse.coo_matrix):
        print("Saving operation matrix to cache")
        self.__operations_cache[key] = operation
        

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
    out = np.zeros((len(v), 1), dtype=np.result_type(values, v))  # Initialize output vector

    nnz = len(values)  # Number of nonzero elements

    for i in range(nnz):  # Iterate over nonzero elements
        out[rowIdx[i], 0] += values[i] * v[colIdx[i], 0]

    return out

def coo_spmv_flat(rowIdx, colIdx, values, v):
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