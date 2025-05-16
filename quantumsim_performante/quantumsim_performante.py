# TODO
# > If matrices are too big, return to CPU processing, no GPU acceleration

try:
    import cupy
    import cupyx.scipy.sparse as cupysparse
    GPU_AVAILABLE = True
    from quantumsim_performante.algorithms import coo_kron_gpu
except:
    print("[ERROR] Cupy could not be imported, make sure that your have installed Cupy")
    print("\tIf you do not have a NVIDIA GPU, you cannot install Cupy")
    print("\tQuantumsim will still work accordingly, just less performant")
    print("\t > Installation guide: https://docs.cupy.dev/en/stable/install.html")
    GPU_AVAILABLE = False
finally:
    import time
    from typing import Union
    import scipy.sparse as sparse
    import numpy as np
    from abc import ABC, abstractmethod

    from quantumsim_performante.noisygate import NoisyGate	
    from quantumsim_performante.algorithms import coo_spmv_row
    from quantumsim_performante.device_parameters import DeviceParameters
    from quantumsim_performante.algorithms import coo_kron
    from quantumsim_performante.dirac import Dirac
    from quantumsim_performante.qubit_unitary_operation import QubitUnitaryOperation

'''
Symbol for pi
'''
pi_symbol = '\u03c0'

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
        # print(timeit(lambda: coo_spmv_row(operation.row, operation.col, operation.data, self.state_vector.flatten()), number=100))
        # print(timeit(lambda: coo_spmv_column(operation.row, operation.col, operation.data, self.state_vector), number=100))

        self.state_vector = coo_spmv_row(operation.row, operation.col, operation.data, self.state_vector)

    def apply_noisy_operation(self, operation: sparse.coo_matrix):
        # A noisy operation does not have to be a unitary matrix
        self.state_vector = coo_spmv_row(operation.row, operation.col, operation.data, self.state_vector)

    def measure_x(self, q):
        # Compute the real part of <psi|X|psi>
        X = CircuitUnitaryOperation.get_combined_operation_for_pauli_x(q, self.N)
        return np.vdot(self.state_vector, X.dot(self.state_vector)).real
    
    def measure_y(self, q):
        # Compute the real part of <psi|Y|psi>
        Y = CircuitUnitaryOperation.get_combined_operation_for_pauli_y(q, self.N)
        return np.vdot(self.state_vector, Y.dot(self.state_vector)).real

    def measure_z(self, q):
        # Compute the real part of <psi|Z|psi>
        Z = CircuitUnitaryOperation.get_combined_operation_for_pauli_z(q, self.N)
        return np.vdot(self.state_vector, Z.dot(self.state_vector)).real
    
    def measure(self):
        # print(np.abs(self.state_vector))
        probalities = np.square(np.abs(self.state_vector))
        # print(probalities)
        self.index = np.random.choice(len(probalities), p=probalities)
        # print(self.index)

    def noisy_measure(self):
        # For a noisy circuit, the sum of probabilities may not be equal to one
        probalities = np.square(np.abs(self.state_vector))
        probalities = probalities / np.sum(probalities)
        self.index = np.random.choice(len(probalities), p=probalities)

    def get_quantum_state(self):
        return self.state_vector
    
    def get_classical_state_as_string(self):
        return self.__state_as_string(self.index, self.N)
    
    def print(self):
        for i, val in enumerate(self.state_vector):
            print(f"{self.__state_as_string(i, self.N)} : {val}")

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

class NoisyStateVector(StateVector):
    def __init__(self, N):
        super().__init__(N)

    def apply_noisy_operation(self, operation: sparse.coo_matrix):
        # A noisy operation does not have to be a unitary matrix
        self.state_vector = coo_spmv_row(operation.row, operation.col, operation.data, self.state_vector)

    def noisy_measure(self):
        # For a noisy circuit, the sum of probabilities may not be equal to one
        probalities = np.square(np.abs(self.state_vector))
        probalities = probalities / np.sum(probalities)
        self.index = np.random.choice(len(probalities), p=probalities)

class CircuitUnitaryOperation:
    """
    Functions to obtain 2^N x 2^N unitary matrices for unitary operations on quantum circuits of N qubits.
    """
    
    @staticmethod
    def get_combined_operation_for_qubit(operation, q, N, gpu=False):
        # If matrices are too big of GPU memory, it will turn of GPU computation
        # if gpu:
        #     mempool = cupy.get_default_memory_pool()
        #     bytes = mempool.total_bytes()

        #     # Checks whether matrix fits on GPU memory, if not, calculations will be done on GPU
        #     # Formula to calculate memory usage for haramard (most memory intensive) for N qubits 
        #     # bytes = 96*2^(N-1)
        #     gpu = (bytes/2) > 96*2**(N-2)

        # Converting dense numpy matrixes to sparse COO scipy matrixes
        operation =  sparse.coo_matrix(operation)
        identity = sparse.coo_matrix(QubitUnitaryOperation.get_identity())
        combined_operation = sparse.coo_matrix(np.eye(1,1))

        # "Selecting" regular scipy sparse matrix kronecker product
        kron = coo_kron

        # print("[INFO] Generating operation matrix: ", end="", flush=True)
        # t1 = time.perf_counter()

        if gpu:
            # Copy data to device (GPU) memory from host (CPU)
            operation = cupysparse.coo_matrix(operation)
            identity = cupysparse.coo_matrix(identity)
            combined_operation = cupysparse.coo_matrix(combined_operation)

            # "Selecting" sparse matrix GPU-accelerated matrix kronecker product
            kron = coo_kron_gpu
    	
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

        # print(combined_operation.nnz/ (combined_operation.shape[0]**2)*100)
        
        # t2 = time.perf_counter()
        # bytes = combined_operation.nnz * 4 * 2 + combined_operation.nnz * 16
        # print(f"{round(t2-t1, 6)*1000}ms ({bytes:,} bytes)")

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
    def get_combined_operation_for_swap(a, b, N):
        combined_operation_cnot_a_b = CircuitUnitaryOperation.get_combined_operation_for_cnot(a, b, N)
        combined_operation_cnot_b_a = CircuitUnitaryOperation.get_combined_operation_for_cnot(b, a, N)

        return (combined_operation_cnot_a_b * combined_operation_cnot_b_a) * combined_operation_cnot_a_b
        # return np.dot(np.dot(combined_operation_cnot_a_b,combined_operation_cnot_b_a),combined_operation_cnot_a_b)

    @staticmethod
    def get_combined_operation_for_cnot(control, target, N, gpu=False):
        # Converting dense numpy matrixes to sparse COO scipy matrixes
        identity = sparse.coo_matrix(QubitUnitaryOperation.get_identity())
        pauli_x = sparse.coo_matrix(QubitUnitaryOperation.get_pauli_x())
        ket_bra_00 = sparse.coo_matrix(Dirac.ket_bra(2,0,0))
        ket_bra_11 = sparse.coo_matrix(Dirac.ket_bra(2,1,1))
        combined_operation_zero = sparse.coo_matrix(np.eye(1,1))
        combined_operation_one = sparse.coo_matrix(np.eye(1,1))
    
        # "Selecting" regular scipy sparse matrix kronecker product
        kron = coo_kron

        if gpu:
            # Copy data to device (GPU) memory from host (CPU)
            identity = cupysparse.coo_matrix(identity)
            pauli_x = cupysparse.coo_matrix(pauli_x)
            ket_bra_00 = cupysparse.coo_matrix(ket_bra_00)
            ket_bra_11 = cupysparse.coo_matrix(ket_bra_11)
            combined_operation_zero = cupysparse.coo_matrix(combined_operation_zero)
            combined_operation_one = cupysparse.coo_matrix(combined_operation_one)

            # "Selecting" sparse matrix GPU-accelerated matrix kronecker product
            kron = coo_kron_gpu

        # print("[INFO] Generating operation CNOT matrix: ", end="", flush=True)

        # Actual computation of kronecker product, this is sort of a iterative problem.
        # Size of "combined_operation" grows exponentially
        # Every qubit makes the kronecker product twice as sparse
        # Computation is done on GPU based on whether parater "GPU" is "True"
        t1 = time.perf_counter()
        for i in range(0, N):
            if control == i:
                combined_operation_zero = kron(combined_operation_zero, ket_bra_00)
                combined_operation_one  = kron(combined_operation_one, ket_bra_11)
            elif target == i:
                combined_operation_zero = kron(combined_operation_zero, identity)
                combined_operation_one  = kron(combined_operation_one, pauli_x)
            else:
                combined_operation_zero = kron(combined_operation_zero, identity)
                combined_operation_one  = kron(combined_operation_one, identity)

        operation = combined_operation_zero + combined_operation_one
        # Copy data back from device (GPU) to host (CPU)
        if gpu: operation = operation.get()
        operation = sparse.coo_matrix(operation)

        t2 = time.perf_counter()

        #bytes = operation.nnz * 4 * 2 + operation.nnz * 16
        # print(f"{round(t2-t1, 6)*1000}ms ({bytes:,} bytes)")

        return operation

class Circuit:
    """
    Class representing a quantum circuit of N qubits.
    """
    def __init__(self, N, use_cache=False, use_GPU=False, use_lazy=False, disk=False):
        self.N = N
        self.state_vector = StateVector(self.N)
        self.quantum_states = [self.state_vector.get_quantum_state()]
        self.descriptions = []
        self.gates = []
        self.operations: Union[list[function], list[sparse.coo_matrix]] = []
        
        # Optimization flags
        self.use_gpu = use_GPU and GPU_AVAILABLE # Only use GPU if available and enabled for use by user.
        self.lazy_evaluation = use_lazy
        self.use_cache = use_cache
        self.operations_cache = {}
        
        if use_cache:
            if use_lazy: print("[Warning] Lazy evaluation and caching cannot be both switched on. Caching is off, lazy evaluation is on")
            self.use_cache = not use_lazy
        else:
            self.use_cache = False

        if not GPU_AVAILABLE and use_GPU:
            print("[Warning] GPU will not be used. 'use_GPU' is set to 'True', but GPU is not available.")

        # "Warming up" the function, calling it compiles the function using Numba
        coo_spmv_row(np.array([0], dtype=np.int32), 
                     np.array([0], dtype=np.int32), 
                     np.array([0], dtype=np.complex128), 
                     np.array([0], dtype=np.complex128))

    def identity(self, q):
        key = ("identity", q)
        description = f"Hadamard on qubit {q}"
        gate_as_string = '.' * self.N
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            # print("[INFO] Identity operation lazely added to circuit")
            self.descriptions.append(description)
            self.operations.append(
                lambda: CircuitUnitaryOperation.get_combined_operation_for_identity(q, self.N, gpu=self.use_gpu)
                )
            
            return

        if self.use_cache: 
            if self.retrieve_operation_from_cache(key, description):
                # print("[INFO] Retrieved operation from cache, added to circuit")
                return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_identity(q, self.N, gpu=self.use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)
        
        # print("[INFO] Identity operation added to circuit")

        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def pauli_x(self, q):
        key = ("pauli_x", q)
        description = f"pauli_x on qubit {q}"
        gate_as_string = '.' * q + 'X' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            # print("[INFO] Pauli_x operation lazely added to circuit")
            self.descriptions.append(description)
            self.operations.append(
                lambda: CircuitUnitaryOperation.get_combined_operation_for_pauli_x(q, self.N, gpu=self.use_gpu)
                )
            return

        if self.use_cache: 
            if self.retrieve_operation_from_cache(key, description):
                # print("[INFO] Retrieved operation from cache, added to circuit")
                return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_x(q, self.N, gpu=self.use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        # print("[INFO] Pauli_x operation added to circuit")

        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def pauli_y(self, q):
        key = ("pauli_y", q)
        description = f"pauli_y on qubit {q}"
        gate_as_string = '.' * q + 'Y' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            # print("[INFO] Pauli_y operation lazely added to circuit")
            self.descriptions.append(description)
            self.operations.append(
                lambda: CircuitUnitaryOperation.get_combined_operation_for_pauli_y(q, self.N, gpu=self.use_gpu)
                )
            return
        
        if self.use_cache: 
            if self.retrieve_operation_from_cache(key, description):
                # print("[INFO] Retrieved operation from cache, added to circuit")
                return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_y(q, self.N, gpu=self.use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        # print("[INFO] Pauli_y operation added to circuit")
        
        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def pauli_z(self, q):
        key = ("pauli_z", q)
        description = f"pauli_z on qubit {q}"
        gate_as_string = '.' * q + 'Z' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            # print("[INFO] Pauli_z operation lazely added to circuit")
            self.descriptions.append(description)
            self.operations.append(
                lambda: CircuitUnitaryOperation.get_combined_operation_for_pauli_z(q, self.N, gpu=self.use_gpu)
                )
            return
 
        if self.use_cache: 
            if self.retrieve_operation_from_cache(key, description):
                # print("[INFO] Retrieved operation from cache, added to circuit")
                return
        
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_z(q, self.N, gpu=self.use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        # print("[INFO] Pauli_z operation added to circuit")
 
        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def hadamard(self, q):
        key = ("hadamard", q)
        description = f"Hadamard on qubit {q}"
        gate_as_string = '.' * q + 'H' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            # print("[INFO] hadamard operation lazely added to circuit")
            self.descriptions.append(description)
            self.operations.append(
                lambda: CircuitUnitaryOperation.get_combined_operation_for_hadamard(q, self.N, gpu=self.use_gpu)
                )
            return
 
        if self.use_cache: 
            if self.retrieve_operation_from_cache(key, description):
                # print("[INFO] Retrieved operation from cache, added to circuit")
                return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_hadamard(q, self.N, gpu=self.use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        # print("[INFO] Hadamard operation added to circuit")
 
        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def phase(self, theta, q):
        key = ("phase", theta, q)
        description = f"Phase with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"
        gate_as_string = '.' * q + 'S' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)
        
        if self.lazy_evaluation:
            # print("[INFO] phase operation lazely added to circuit")
            self.descriptions.append(description)
            self.operations.append(
                lambda: CircuitUnitaryOperation.get_combined_operation_for_phase(theta, q, self.N, gpu=self.use_gpu)
                )
            return
 
        if self.use_cache: 
            if self.retrieve_operation_from_cache(key, description):
                # print("[INFO] Retrieved operation from cache, added to circuit")
                return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_phase(theta, q, self.N, gpu=self.use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        # print("[INFO] phase operation added to circuit")
 
        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def rotate_x(self, theta, q):
        key = ("rotate_x", theta, q)
        description = f"Rotate X with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"
        gate_as_string = '.' * q + 'R' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)
        
        if self.lazy_evaluation:
            # print("[INFO] Rotate_x operation lazely added to circuit")
            self.descriptions.append(description)
            self.operations.append(
                lambda: CircuitUnitaryOperation.get_combined_operation_for_rotate_x(theta, q, self.N, gpu=self.use_gpu)
                )
            return
 
        if self.use_cache: 
            if self.retrieve_operation_from_cache(key, description):
                # print("[INFO] Retrieved operation from cache, added to circuit")
                return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_x(theta, q, self.N, gpu=self.use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        # print("[INFO] Rotate_x operation added to circuit")
 
        if self.use_cache:
            self.cache_operation(key, combined_operation)
    
    def rotate_y(self, theta, q):
        key = ("rotate_y", theta, q)
        description = f"Rotate_y with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"
        gate_as_string = '.' * q + 'R' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            # print("[INFO] Rotate_y operation lazely added to circuit")
            self.descriptions.append(description)
            self.operations.append(
                lambda: CircuitUnitaryOperation.get_combined_operation_for_rotate_y(theta, q, self.N, gpu=self.use_gpu)
                )
            return
 
        if self.use_cache: 
            if self.retrieve_operation_from_cache(key, description):
                # print("[INFO] Retrieved operation from cache, added to circuit")
                return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_y(theta, q, self.N, gpu=self.use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        # print("[INFO] Rotate_y operation added to circuit")
 
        if self.use_cache:
            self.cache_operation(key, combined_operation)
    
    def rotate_z(self, theta, q):
        key = ("rotate_z", theta, q)
        description = f"Rotate_z with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"
        gate_as_string = '.' * q + 'R' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            # print("[INFO] Rotate_z operation lazely added to circuit")
            self.descriptions.append(description)
            self.operations.append(
                lambda: CircuitUnitaryOperation.get_combined_operation_for_rotate_z(theta, q, self.N, gpu=self.use_gpu)
                )
            return
 
        if self.use_cache: 
            if self.retrieve_operation_from_cache(key, description):
                # print("[INFO] Retrieved operation from cache, added to circuit")
                return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_z(theta, q, self.N, gpu=self.use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        # print("[INFO] Rotate_z operation added to circuit")
 
        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def cnot(self, control, target):
        key = ("cnot", control, target)
        description = f"CNOT with control qubit {control} and target qubit {target}"
        gate_as_string = ''.join('*' if i == control else 'X' if i == target else '.' for i in range(self.N))
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            # print("[INFO] CNOT Operation operation lazely added to circuit")
            self.descriptions.append(description)
            self.operations.append(
                lambda: CircuitUnitaryOperation.get_combined_operation_for_cnot(control, target, self.N, gpu=self.use_gpu)
                )
            return
 
        if self.use_cache: 
            if self.retrieve_operation_from_cache(key, description):
                # print("[INFO] Retrieved operation from cache, added to circuit")
                return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_cnot(control, target, self.N, gpu=self.use_gpu)
        self.descriptions.append(description)
        self.operations.append(combined_operation)

        # print("[INFO] CNOT Operation added to circuit")
 
        if self.use_cache:
            self.cache_operation(key, combined_operation)

    def execute(self, print_state=False, benchmark=False):
        self.state_vector = StateVector(self.N)
        if print_state:
            print("Initial quantum state")
            self.state_vector.print()

        # Checking variable type (based on if lazy flag is True) is correct
        assert isinstance(self.operations, list), "Operations should be a list"
        if self.lazy_evaluation == True:
            assert all(isinstance(op, type(lambda: None)) for op in self.operations), "Operation matrices are lazely evaluated but the operations list is not a list of functions"
        else:
            assert all(isinstance(op, sparse.coo_matrix) for op in self.operations), "Operation matrices are evaluated but the operations list is not a list of coo_matrix"

        for i, (operation, description) in enumerate(zip(self.operations, self.descriptions)):
            # print("[INFO] Apply unitary operation: ", end="", flush=True)
            
            t1 = time.perf_counter()

            if self.lazy_evaluation:
                operation = operation()

            self.state_vector.apply_unitary_operation(operation)
            self.quantum_states.append(self.state_vector.get_quantum_state())

            t2 = time.perf_counter()
            if benchmark:
                print(f"[INFO] Apply unitary operation: {round(t2-t1, 6)*1000}ms")

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
        # print("[INFO] Saving operation matrix to cache")
        self.operations_cache[key] = operation
        
    def print_circuit(self):
        for description in self.descriptions:
            print(description)

'''
Class representing a noisy quantum circuit of N qubits.
Inherits from Circuit.
'''
class NoisyCircuit(Circuit):
    def __init__(self, N,  use_cache=False, use_GPU=False, use_lazy=False, disk=False):
        super().__init__(N, use_cache=use_cache, use_GPU=use_GPU, use_lazy=use_lazy, disk=disk)
        self.state_vector = NoisyStateVector(self.N)
        self.noisy_operations_state_prep: Union[list[function], list[sparse.coo_matrix]] = []
        self.noisy_operations_incoherent: Union[list[function], list[sparse.coo_matrix]] = []
        self.noisy_operations_readout: Union[list[function], list[sparse.coo_matrix]] = []
        self.x_measures = np.empty(self.N, dtype=object)
        self.y_measures = np.empty(self.N, dtype=object)
        self.z_measures = np.empty(self.N, dtype=object)

        # Noisy gates
        self.phi = [0 for _ in range(self.N)] # Keep a list of phi values for every qubit

        # Load in the device parameters json
        device_params = DeviceParameters()
        device_params.load_from_json("./assets/noise_parameters/QiskitKyiv_DeviceParameters.json")
        qiskit_kyiv_parameter_dict = device_params.__dict__()

        # Define a list of parameter values based on the stored device parameters
        self.parameters = {
            "T1": [float(qiskit_kyiv_parameter_dict["T1"][i % len(qiskit_kyiv_parameter_dict["T1"])]) for i in range(self.N)], # Loop over the T1 values of the device parameters to assign to each qubit
            "T2": [float(qiskit_kyiv_parameter_dict["T2"][i % len(qiskit_kyiv_parameter_dict["T2"])]) for i in range(self.N)], # Loop over the T2 values of the device parameters to assign to each qubit
            "p": [float(qiskit_kyiv_parameter_dict["p"][i % len(qiskit_kyiv_parameter_dict["p"])]) for i in range(self.N)], # Loop over the p values of the device parameters to assign to each qubit
        }

    # Define the new Pauli X gate with integrated noise 
    def noisy_pauli_x(self, q: int, p: float = None, T1: float = None, T2: float = None):
        """Adds a noisy Pauli X gate to the circuit

        Args:
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """
        
        # If any noise parameter is None use the generated value
        if p  is None: p = self.parameters["p"][q]
        if T1 is None: T1 = self.parameters["T1"][q]
        if T2 is None: T2 = self.parameters["T2"][q]

        # Theta and phi to construct Pauli X
        theta = np.pi
        phi = -self.phi[q]

        if self.lazy_evaluation:
            combined_operation = lambda: CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(theta, phi, p, T1, T2), q, self.N)
        else:
            combined_operation = CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(theta, phi, p, T1, T2), q, self.N)
        self.operations.append(combined_operation)
        self.descriptions.append(f"Noisy Pauli X on qubit {q}")

        gate_as_string = '.' * q + 'X' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

    # Define the new Pauli Y gate with integrated noise 
    def noisy_pauli_y(self, q: int, p: float= None, T1: float= None, T2: float= None):
        """Adds a noisy Pauli Y gate to the circuit

        Args:
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """

        # If any noise parameter is None use the generated value
        if p  is None: p = self.parameters["p"][q]
        if T1 is None: T1 = self.parameters["T1"][q]
        if T2 is None: T2 = self.parameters["T2"][q]

        # First execute a virtual Rz gate
        self.virtual_rotate_z(q, np.pi)

        # Theta and phi to construct Pauli Y
        theta = np.pi
        phi = -self.phi[q]

        if self.lazy_evaluation:
            combined_operation = lambda: CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(theta, phi, p, T1, T2), q, self.N)
        else:
            combined_operation = CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(theta, phi, p, T1, T2), q, self.N)
        self.operations.append(combined_operation)
        self.descriptions.append(f"Noisy Pauli Y on qubit {q}")

        gate_as_string = '.' * q + 'Y' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

    # Define the new "virtual" Pauli Z gate
    def noisy_pauli_z(self, q: int):
        """This gate is implemented virtualy and thus is not executed on the actual qubit!

        Args:
            q (int): Qubit to operate on.
        """

        # Execute a virtual Rz gate
        self.virtual_rotate_z(q, np.pi)
    
    # Define the new hadamard gate with integrated noise 
    def noisy_hadamard(self, q: int, p: float= None, T1: float= None, T2: float= None):
        """Adds a noisy hadamard gate to the circuit

        Args:
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """

        # If any noise parameter is None use the generated value
        if p  is None: p = self.parameters["p"][q]
        if T1 is None: T1 = self.parameters["T1"][q]
        if T2 is None: T2 = self.parameters["T2"][q]

        # First execute a virtual Rz gate
        self.virtual_rotate_z(q, np.pi / 2)

        # Theta for a square root X gate and phi from phi list
        theta = np.pi / 2
        phi = -self.phi[q]

        if self.lazy_evaluation:
            combined_operation = lambda: CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(theta, phi, p, T1, T2), q, self.N)
        else:
            combined_operation = CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(theta, phi, p, T1, T2), q, self.N)
        self.operations.append(combined_operation)
        self.descriptions.append(f"Noisy Hadamard on qubit {q}")

        gate_as_string = '.' * q + 'H' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

        # To complete the gate end with a virtual Rz gate
        self.virtual_rotate_z(q, np.pi / 2)
    
    def noisy_phase(self, theta: float, q: int, p: float = None, T1: float = None, T2: float = None):
        """This gate is implemented making use of perfect hadamard in combination with a X gate!

        Args:
            theta (float): Angle of rotation on the Bloch sphere.
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """

         # If any noise parameter is None use the generated value
        if p is None: p = self.parameters["p"][q]
        if T1 is None: T1 = self.parameters["T1"][q]
        if T2 is None: T2 = self.parameters["T2"][q]
        
        phi = -self.phi[q]

        # Set in hadamard basis
        self.hadamard(q)

        if self.lazy_evaluation:
            combined_operation = lambda: CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(theta, phi, p, T1, T2), q, self.N)
        else: 
            combined_operation = CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(theta, phi, p, T1, T2), q, self.N)
        self.operations.append(combined_operation)
        self.descriptions.append(f"Noisy X rotation of {theta} on qubit {q}")

        gate_as_string = '.' * q + 'S' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

        # Get out of hadamard basis
        self.hadamard(q)

    # Define the new cnot gate with integrated noise 
    def noisy_cnot(self, control: int, target: int, c_p: float= None, t_p: float= None, c_T1: float= None, t_T1: float= None, c_T2: float= None, t_T2: float= None, gate_error: float=None):
        """Adds a noisy cnot gate to the circuit with depolarizing and
        relaxation errors on both qubits during the unitary evolution.

        Args:
            c_qubit (int): Control qubit for the gate.
            t_qubit (int): Target qubit for the gate.
            c_p (float): Depolarizing error probability for the control qubit.
            t_p (float): Depolarizing error probability for the target qubit.
            c_T1 (float): Amplitude damping time in ns for the control qubit.
            t_T1 (float): Amplitude damping time in ns for the target qubit.
            c_T2 (float): Dephasing time in ns for the contorl qubit.
            t_T2 (float): Dephasing time in ns for the target qubit.
            gate_error (float): CNOT depolarizing error probability.
        """

        # If any noise parameter is None use the generated value
        if c_p  is None: c_p = self.parameters["p"][control]
        if c_T1 is None: c_T1 = self.parameters["T1"][control]
        if c_T2 is None: c_T2 = self.parameters["T2"][control]
        if t_p  is None: t_p = self.parameters["p"][target]
        if t_T1 is None: t_T1 = self.parameters["T1"][target]
        if t_T2 is None: t_T2 = self.parameters["T2"][target]
        if gate_error is None: 
            gate_error = 0.015

        gate_length = 5.61777778e-07

        # Create an identity matrix for the remaining qubits
        identity_matrix = np.eye(2**(self.N - 2))

        # Create cnot matrix
        if control < target:
            cnot_operation = NoisyGate.construct_cnot(self.phi[control], self.phi[target], gate_length, gate_error, c_p, t_p, c_T1, c_T2, t_T1, t_T2)
            self.phi[control] = self.phi[control] - np.pi/2
        else:
            cnot_operation = NoisyGate.construct_cnot_inverse(self.phi[control], self.phi[target], gate_length, gate_error, c_p, t_p, c_T1, c_T2, t_T1, t_T2)
            self.phi[control] = self.phi[control] + np.pi/2 + np.pi
            self.phi[target] = self.phi[target] + np.pi/2

        
        # FIXME: Use GPU is self.__use_gpu is True
        # Perform the Kronecker product to expand the CNOT operation
        # if GPU_AVAILABLE and self.__use_gpu:
        #     cnot_operation = coo_kron_gpu(cnot_operation, identity_matrix)
        # else:
        #     cnot_operation = coo_kron(cnot_operation, identity_matrix)
        # cnot_operation = coo_kron(cnot_operation, identity_matrix)

        cnot_operation = np.kron(cnot_operation, identity_matrix)
        cnot_operation = sparse.csr_matrix(cnot_operation)

        # Create swap matrices
        if control < target:
            # control qubit should swap with qubit 0
            swap_control = CircuitUnitaryOperation.get_combined_operation_for_swap(0, control, self.N) if control != 0 else np.eye(2**self.N)
            swap_target = CircuitUnitaryOperation.get_combined_operation_for_swap(1, target, self.N) if target != 1 else np.eye(2**self.N)
        else:
            # target qubit should swap with qubit 0
            swap_target = CircuitUnitaryOperation.get_combined_operation_for_swap(0, target, self.N) if target != 0 else np.eye(2**self.N)
            swap_control = CircuitUnitaryOperation.get_combined_operation_for_swap(1, control, self.N) if control != 1 else np.eye(2**self.N)
        
        # FIXME: ALL THESE MATRICES ARE CSR NOT COO
        # print(type(swap_control))
        # print(type(swap_target))
        # print(type(cnot_operation))

        # Construct the full CNOT operation with swaps
        operation = swap_control @ swap_target @ cnot_operation @ swap_target.T.conj() @ swap_control.T.conj() 
        operation = sparse.coo_matrix(operation)    

        self.descriptions.append(f"Noisy CNOT with target qubit {target} and control qubit {control}")
        self.operations.append(operation)

        gate_as_string = ''.join('*' if i == control else 'X' if i == target else '.' for i in range(self.N))
        self.gates.append(gate_as_string)

    # Define a virtual Rz gate to mimic the "quantum-gates" package
    def virtual_rotate_z(self, q: int, theta: float):
        """ This gate is implemented virtualy and thus is not executed on the actual qubit!
        Update the phase to implement virtual Rz(theta) gate on qubit i

        Args:
            q: index of the qubit
            theta: angle of rotation on the Bloch sphere

        Returns:
             None
        """
        self.phi[q] += theta

    def add_noisy_operation_state_prep(self, p, q):
        noisy_operation_state_prep = (1-p)*Dirac.ket_bra(2,0,0) + p*Dirac.ket_bra(2,1,1)
        noisy_operation_state_prep = noisy_operation_state_prep.astype(np.complex128)
        
        if self.lazy_evaluation:
            combined_noisy_operation_state_prep = lambda: CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_state_prep, q, self.N)
        else:
            combined_noisy_operation_state_prep = CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_state_prep, q, self.N)

        self.noisy_operations_state_prep.append(combined_noisy_operation_state_prep)

    def add_noisy_operation_coherent_x(self, theta, q):
        theta_radians = (theta/180)*np.pi
        noisy_operation_coherent = QubitUnitaryOperation.get_rotate_x(theta_radians)
        
        if self.lazy_evaluation:
            combined_noisy_operation_coherent = lambda: CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_coherent, q, self.N)
        else:
            combined_noisy_operation_coherent = CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_coherent, q, self.N)

        self.operations.append(combined_noisy_operation_coherent)
        self.descriptions.append(f"Coherent noise rot_X {theta} deg")
        gate_as_string = '.' * q + 'N' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

    def add_noisy_operation_coherent_y(self, theta, q):
        theta_radians = (theta/180)*np.pi
        noisy_operation_coherent = QubitUnitaryOperation.get_rotate_y(theta_radians)
        
        if self.lazy_evaluation:
            combined_noisy_operation_coherent = lambda: CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_coherent, q, self.N)
        else:
            combined_noisy_operation_coherent = CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_coherent, q, self.N)

        self.operations.append(combined_noisy_operation_coherent)
        self.descriptions.append(f"Coherent noise rot_Y {theta} deg")
        gate_as_string = '.' * q + 'N' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

    def add_noisy_operation_coherent_z(self, theta, q):
        theta_radians = (theta/180)*np.pi
        noisy_operation_coherent = QubitUnitaryOperation.get_rotate_z(theta_radians)

        if self.lazy_evaluation:
            combined_noisy_operation_coherent = lambda: CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_coherent, q, self.N)
        else:
            combined_noisy_operation_coherent = CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_coherent, q, self.N)

        self.operations.append(combined_noisy_operation_coherent)
        self.descriptions.append(f"Coherent noise rot_Z {theta} deg")
        gate_as_string = '.' * q + 'N' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

    def add_noisy_operation_incoherent(self, px, py, pz, q):
        I = QubitUnitaryOperation.get_identity()
        X = QubitUnitaryOperation.get_pauli_x()
        Y = QubitUnitaryOperation.get_pauli_y()
        Z = QubitUnitaryOperation.get_pauli_z()
        
        noisy_operation_incoherent = (1-px-py-pz)*I + px*X + py*Y +pz*Z
        
        if self.lazy_evaluation:
            combined_noisy_operation_incoherent = lambda: CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_incoherent, q, self.N)
        else:
            combined_noisy_operation_incoherent = CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_incoherent, q, self.N)
        
        self.noisy_operations_incoherent.append(combined_noisy_operation_incoherent)

    def add_noisy_operation_readout(self, epsilon, nu, q):
        noisy_operation_readout = np.array([[1-epsilon,nu],[epsilon,1-nu]], dtype=np.complex128)
        
        if self.lazy_evaluation:
            combined_noisy_operation_readout = lambda: CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_readout, q, self.N)
        else:
            combined_noisy_operation_readout = CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_readout, q, self.N)

        self.noisy_operations_readout.append(combined_noisy_operation_readout)

    # Override method execute() from class Circuit
    def execute(self, print_state=False):
        # Checking variable type (based on if lazy flag is True) is correct
        assert isinstance(self.operations, list), "Operations should be a list"
        if self.lazy_evaluation == True:
            assert all(isinstance(op, type(lambda: None)) for op in self.operations), "Operation matrices are lazely evaluated but the operations list is not a list of functions"
        else:
            assert all(isinstance(op, sparse.coo_matrix) for op in self.operations), "Operation matrices are evaluated but the operations list is not a list of coo_matrix"

        # Checking variable type (based on if lazy flag is True) is correct
        assert isinstance(self.noisy_operations_incoherent, list), "Operations should be a list"
        if self.lazy_evaluation == True:
            assert all(isinstance(op, type(lambda: None)) for op in self.noisy_operations_incoherent), "noisy_operations_incoherent matrices are lazely evaluated but the operations list is not a list of functions"
        else:
            assert all(isinstance(op, sparse.coo_matrix) for op in self.noisy_operations_incoherent), "noisy_operations_incoherent matrices are evaluated but the operations list is not a list of coo_matrix"
        
        # Checking variable type (based on if lazy flag is True) is correct
        assert isinstance(self.noisy_operations_state_prep, list), "Operations should be a list"
        if self.lazy_evaluation == True:
            assert all(isinstance(op, type(lambda: None)) for op in self.noisy_operations_state_prep), "noisy_operations_state_prep matrices are lazely evaluated but the operations list is not a list of functions"
        else:
            assert all(isinstance(op, sparse.coo_matrix) for op in self.noisy_operations_state_prep), "noisy_operations_state_prep matrices are evaluated but the operations list is not a list of coo_matrix"

        self.state_vector = StateVector(self.N)
        for noisy_operation in self.noisy_operations_state_prep:
            if self.lazy_evaluation: noisy_operation = noisy_operation()
            self.state_vector.apply_noisy_operation(noisy_operation)

        self.quantum_states = [self.state_vector.get_quantum_state()]

        for q in range(self.N):
            self.x_measures[q] = [self.state_vector.measure_x(q)]
            self.y_measures[q] = [self.state_vector.measure_y(q)]
            self.z_measures[q] = [self.state_vector.measure_z(q)]

        if print_state:
            print("Initial quantum state")
            self.state_vector.print()

        for i, (operation, description) in enumerate(zip(self.operations, self.descriptions)):
            if self.lazy_evaluation: operation = operation()

            # Apply operation to statevector   
            self.state_vector.apply_unitary_operation(operation)
            self.quantum_states.append(self.state_vector.get_quantum_state())

            if "Coherent noise" not in description:
                for noisy_operation in self.noisy_operations_incoherent:
                    if self.lazy_evaluation: noisy_operation = noisy_operation()
                    self.state_vector.apply_noisy_operation(noisy_operation)

                for q in range(self.N):
                    self.x_measures[q].append(self.state_vector.measure_x(q))
                    self.y_measures[q].append(self.state_vector.measure_y(q))
                    self.z_measures[q].append(self.state_vector.measure_z(q))

            # Logging
            if print_state:
                print(description)
                print(operation)
                print("Current quantum state")
                self.state_vector.print()

    # Override method measure() from class Circuit
    def measure(self, print_state=False):
        # Checking variable type (based on if lazy flag is True) is correct
        assert isinstance(self.noisy_operations_readout, list), "Operations should be a list"
        if self.lazy_evaluation == True:
            assert all(isinstance(op, type(lambda: None)) for op in self.noisy_operations_readout), "noisy_operations_readout matrices are lazely evaluated but the operations list is not a list of functions"
        else:
            assert all(isinstance(op, sparse.coo_matrix) for op in self.noisy_operations_readout), "noisy_operations_readout matrices are evaluated but the operations list is not a list of coo_matrix"

        # Apply noise operation matrix to statevector
        for noisy_operation in self.noisy_operations_readout:
            if self.lazy_evaluation: noisy_operation = noisy_operation()
            self.state_vector.apply_noisy_operation(noisy_operation)

        # Do actual measurement
        self.state_vector.noisy_measure()

        # Logging
        if print_state:
            print("Measured state:")
            print(self.state_vector.get_classical_state_as_string())

    def get_x_measures(self, q):
        return self.x_measures[q]
    
    def get_y_measures(self, q):
        return self.y_measures[q]
    
    def get_z_measures(self, q):
        return self.z_measures[q]

class NoisyGateInstruction(ABC):
    @abstractmethod
    def getNoisyOperation(self):
        pass

class NoisyPauliX(NoisyGateInstruction):
    def __init__(self, q: int, totalQubits: int, p: float = None, T1: float = None, T2: float = None, ):
        self.q = q
        self.N = totalQubits
        self.p = p
        self.T1 = T1
        self.T2 = T2

    def setTheta(self, theta: float):
        self.theta = theta
        
    def setPhi(self, phi: float):
        self.phi = phi

    def getNoisyOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(self.theta, self.phi, self.p, self.T1, self.T2),self.q, self.N)
    
class NoisyPauliY(NoisyGateInstruction):
    def __init__(self, q: int, totalQubits: int, p: float = None, T1: float = None, T2: float = None, ):
        self.q = q
        self.N = totalQubits
        self.p = p
        self.T1 = T1
        self.T2 = T2

    def setTheta(self, theta: float):
        self.theta = theta
        
    def setPhi(self, phi: float):
        self.phi = phi
        
    def getNoisyOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(self.theta, self.phi, self.p, self.T1, self.T2),self.q, self.N)

class NoisyPauliZ(NoisyGateInstruction):
    def __init__(self, q: int, totalQubits: int, p: float = None, T1: float = None, T2: float = None, ):
        self.q = q
        self.N = totalQubits
        self.p = p
        self.T1 = T1
        self.T2 = T2

    def setTheta(self, theta: float):
        self.theta = theta
        
    def setPhi(self, phi: float):
        self.phi = phi

    def getNoisyOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(self.theta, self.phi, self.p, self.T1, self.T2),self.q, self.N)

class NoisyPhase(NoisyGateInstruction):
    def __init__(self, theta: float, q: int, totalQubits: int, p: float = None, T1: float = None, T2: float = None, ):
        self.q = q
        self.theta = theta
        self.N = totalQubits
        self.p = p
        self.T1 = T1
        self.T2 = T2

    def setPhi(self, phi: float):
        self.phi = phi

    def getNoisyOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(self.theta, self.phi, self.p, self.T1, self.T2),self.q, self.N)

class NoisyHadamard(NoisyGateInstruction):
    def __init__(self, q: int, totalQubits: int, p: float = None, T1: float = None, T2: float = None, ):
        self.q = q
        self.N = totalQubits
        self.p = p
        self.T1 = T1
        self.T2 = T2

    def setTheta(self, theta: float):
        self.theta = theta
        
    def setPhi(self, phi: float):
        self.phi = phi
        
    def getNoisyOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(self.theta, self.phi, self.p, self.T1, self.T2),self.q, self.N)
    
class NoisyCNOT(NoisyGateInstruction):
    def __init__(self, c_qubit: int, t_qubit: int, N: int, c_p: float= None, t_p: float= None, c_T1: float= None, t_T1: float= None, c_T2: float= None, t_T2: float= None, gate_error: float=None):
        self.c_qubit = c_qubit
        self.t_qubit = t_qubit
        self.N = N
        self.c_p = c_p
        self.t_p = t_p
        self.c_T1 = c_T1
        self.t_T1 = t_T1
        self.c_T2 = c_T2
        self.t_T2 = t_T2
        self.gate_error = gate_error

    def setTheta(self, theta: float):
        self.theta = theta
        
    def setPhiControl(self, phi: float):
        self.c_phi = phi

    def setPhiTarget(self, phi: float):
        self.t_phi = phi

    def getNoisyOperation(self) -> CircuitUnitaryOperation:
        # This is a solution for mimicking noise of two qubits by adding single qubit gates, returns a pauli x gate
        # return CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(self.theta, self.c_phi, self.p, self.T1, self.T2),self.q, self.N)

        gate_length = 5.61777778e-07

        # Create an identity matrix for the remaining qubits
        identity_matrix = np.eye(2**(self.N - 2))

        # Create cnot matrix
        if self.c_qubit < self.t_qubit:
            cnot_operation = NoisyGate.construct_cnot(self.c_phi, self.t_phi, gate_length, self.gate_error, self.c_p, self.t_p, self.c_T1, self.c_T2, self.t_T1, self.t_T2)
            self.c_phi = self.c_phi - np.pi/2
        else:
            cnot_operation = NoisyGate.construct_cnot_inverse(self.c_phi, self.t_phi, gate_length, self.gate_error, self.c_p, self.t_p, self.c_T1, self.c_T2, self.t_T1, self.t_T2)
            self.c_phi = self.c_phi + np.pi/2 + np.pi
            self.t_phi = self.t_phi + np.pi/2
        
        # Perform the Kronecker product to expand the CNOT operation
        cnot_operation = np.kron(cnot_operation, identity_matrix)

        # Create swap matrices
        if self.c_qubit < self.t_qubit:
            # control qubit should swap with qubit 0
            swap_control = CircuitUnitaryOperation.get_combined_operation_for_swap(0, self.c_qubit, self.N) if self.c_qubit != 0 else np.eye(2**self.N)
            swap_target = CircuitUnitaryOperation.get_combined_operation_for_swap(1, self.t_qubit, self.N) if self.t_qubit != 1 else np.eye(2**self.N)
        else:
            # target qubit should swap with qubit 0
            swap_target = CircuitUnitaryOperation.get_combined_operation_for_swap(0, self.t_qubit, self.N) if self.t_qubit != 0 else np.eye(2**self.N)
            swap_control = CircuitUnitaryOperation.get_combined_operation_for_swap(1, self.c_qubit, self.N) if self.c_qubit != 1 else np.eye(2**self.N)

        # Construct the full CNOT operation with swaps
        operation = swap_control @ swap_target @ cnot_operation @ swap_target.T.conj() @ swap_control.T.conj() 
        return operation
    
class NoisyReset(NoisyGateInstruction):
    def __init__(self, q: int, readBit: int, totalQubits: int, p: float = None, T1: float = None, T2: float = None):
        self.q = q
        self.readBit = readBit
        self.N = totalQubits
        self.p = p
        self.T1 = T1
        self.T2 = T2

    def setTheta(self, theta: float):
        self.theta = theta
        
    def setPhi(self, phi: float):
        self.phi = phi

    def getNoisyOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(self.theta, self.phi, self.p, self.T1, self.T2),self.q, self.N)
