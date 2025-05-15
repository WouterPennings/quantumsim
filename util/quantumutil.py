import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.animation as animation
from quantumsim_performante import *

"""
Supporting functions for execution, measurement, and visualisation of intermediate quantum states.
"""
class QuantumUtil:
    """
    Function to run a quantum circuit and measure the classical state.
    """
    @staticmethod
    def run_circuit(circuit:Circuit, nr_runs=1000):
        # if(circuit.save_instructions):
        #     raise Exception("Direct Operation Execution is enabled, QuantumUtil not supported with this flag")
        result = []
        for i in range(nr_runs):
            circuit.execute()
            circuit.measure()
            result.append(circuit.get_classical_state_as_string())
        return result

    """
    Function to run a quantum circuit once and measure the classical state many times.
    """
    @staticmethod
    def measure_circuit(circuit:Circuit, nr_measurements=1000, little_endian_formatted: bool=False):
        circuit.execute()
        result = []
        for i in range(nr_measurements):
            circuit.measure()
            result.append(circuit.get_classical_state_as_string())
        return result

    """"
    Function to run a quantum circuit many times and measure its classical register state many times
    """
    @staticmethod
    def measure_circuit_bit_register(circuit:Circuit, nr_measurements=100, beginBit: int=0, endBit: int = 0):
        result = []
        for i in range(nr_measurements):
            circuit.execute()
            result.append(circuit.classicalBitRegister.toString(beginBit, endBit))
        return result

    """
    Function to plot a histogram of all classical states after executing the circuit multiple times.
    """
    @staticmethod
    def histogram_of_classical_states(ideal_string_array, noisy_string_array=None):
        ideal_histogram = Counter(ideal_string_array)
        ideal_unique_strings = sorted(list(ideal_histogram.keys()))
        ideal_counts = [ideal_histogram[string] for string in ideal_unique_strings]

        if noisy_string_array is None:
            plt.bar(ideal_unique_strings, ideal_counts)
            if len(ideal_histogram) > 8:
                plt.xticks(rotation='vertical')
            plt.xlabel('Classical states')
            plt.ylabel('Nr occurrences')
            plt.title('Number of occurrences of classical states')
            plt.show()
        else:
            width = 0.4  # Width of the bars

            # Combine and sort all unique strings
            noisy_histogram = Counter(noisy_string_array)
            all_unique_strings = sorted(set(ideal_unique_strings + list(noisy_histogram.keys())))
            ideal_counts = [ideal_histogram.get(string, 0) for string in all_unique_strings]
            noisy_counts = [noisy_histogram.get(string, 0) for string in all_unique_strings]

            # Generate x positions for the bars
            x = np.arange(len(all_unique_strings))

            # Plot ideal and noisy bars side by side
            plt.bar(x - width / 2, ideal_counts, width, label='Ideal')
            plt.bar(x + width / 2, noisy_counts, width, label='Noisy', color='#eb4034')

            # Set x-tick labels to the classical state strings
            plt.xticks(x, all_unique_strings, rotation='vertical' if len(all_unique_strings) > 8 else 'horizontal')

            # Add labels and title
            plt.xlabel('Classical states')
            plt.ylabel('Nr occurrences')
            plt.title('Number of occurrences of classical states')
            plt.tight_layout()
            plt.legend(loc='upper right')
            plt.show()


    """
    Function to plot a all intermediate (quantum) states of the last execution of a circuit.
    """
    @staticmethod
    def show_all_intermediate_states(circuit:Circuit, show_description=True, show_colorbar=True):
        matrix_of_all_states = np.zeros((2**circuit.N, len(circuit.quantum_states)), dtype=complex)
        i = 0
        for state_vector in circuit.quantum_states:
            matrix_of_all_states[:,i] = state_vector.flatten()
            i = i + 1

        fig_width  = 4 + circuit.N
        fig_height = 4 + 0.5*len(circuit.operations)
        fig, ax = plt.subplots()
        fig.set_size_inches(fig_width, fig_height)
        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        radius_circle = 0.45
        length_arrow = 0.4
        color_map = mcol.LinearSegmentedColormap.from_list('CmapBlueRed',['b','r'])
        norm = plt.Normalize(vmin=0, vmax=1)

        for (x, y), c in np.ndenumerate(matrix_of_all_states):
            r = abs(c)
            phase = cmath.phase(c)
            color = color_map(int(r*256))
            circle = plt.Circle([x + 0.5, y + 0.5], radius_circle, facecolor=color, edgecolor='black')
            dx = length_arrow * np.cos(phase)
            dy = length_arrow * np.sin(phase)
            arrow = plt.Arrow(x + 0.5 - dx, y + 0.5 - dy, 2*dx, 2*dy, facecolor='lightgray', edgecolor='black')
            ax.add_patch(circle)
            ax.add_patch(arrow)

        ax.autoscale_view()
        ax.invert_yaxis()

        positions_x = []
        all_states_as_string = []
        for i in range(0,2**circuit.N):
            positions_x.append(i + 0.5)
            all_states_as_string.append(Dirac.state_as_string(i,circuit.N))
        plt.xticks(positions_x, all_states_as_string, rotation='vertical')

        j = 0.5
        positions_y = [j]
        if show_description:
            all_operations_as_string = ['Initial state  ' + '.'*circuit.N]
        else:
            all_operations_as_string = ['.'*circuit.N]
        j = j + 1
        for description, gate in zip(circuit.descriptions, circuit.gates):
            positions_y.append(j)
            if show_description:
                all_operations_as_string.append(f"{description}  {gate}")
            else:
                all_operations_as_string.append(f"{gate}")
            j = j + 1
        plt.yticks(positions_y, all_operations_as_string)

        if show_colorbar:
            sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
            sm.set_array([])
            ax = plt.gca()
            divider = ax.get_position()
            shrink = divider.height
            cbar = plt.colorbar(sm, ax=ax, shrink=shrink)
        
        plt.title('Intermediate quantum states')
        plt.show()


    """
    Function to plot a all intermediate probabilities of the last execution of a circuit.
    """
    @staticmethod
    def show_all_probabilities(circuit:Circuit, show_description=True, show_colorbar=True):
        matrix_of_probabilities = np.zeros((2**circuit.N,len(circuit.quantum_states)))
        i = 0
        for state_vector in circuit.quantum_states:
            probalities = np.square(np.abs(state_vector)).flatten()
            matrix_of_probabilities[:,i] = probalities
            i = i + 1

        fig_width  = 4 + circuit.N
        fig_height = 4 + 0.5*len(circuit.operations)
        fig, ax = plt.subplots()
        fig.set_size_inches(fig_width, fig_height)
        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        size = 0.9
        color_map = mcol.LinearSegmentedColormap.from_list('CmapBlueRed',['b','r'])
        norm = plt.Normalize(vmin=0, vmax=1)

        for (x, y), w in np.ndenumerate(matrix_of_probabilities):
            color = color_map(int(w*256))
            rect = plt.Rectangle([x - size/2, y - size/2], size, size,
                                facecolor=color, edgecolor='black')
            ax.add_patch(rect)

        ax.autoscale_view()
        ax.invert_yaxis()
            
        positions_x = []
        all_states_as_string = []
        for i in range(0,2**circuit.N):
            positions_x.append(i)
            all_states_as_string.append(Dirac.state_as_string(i, circuit.N))
        plt.xticks(positions_x, all_states_as_string, rotation='vertical')

        positions_y = [0]
        if show_description:
            all_operations_as_string = ['Initial state  ' + '.'*circuit.N]
        else:
            all_operations_as_string = ['.'*circuit.N]
        j = 1
        for description, gate in zip(circuit.descriptions, circuit.gates):
            positions_y.append(j)
            if show_description:
                all_operations_as_string.append(f"{description}  {gate}")
            else:
                all_operations_as_string.append(f"{gate}")
            j = j + 1
        plt.yticks(positions_y, all_operations_as_string)

        if show_colorbar:
            sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
            sm.set_array([])
            ax = plt.gca()
            divider = ax.get_position()
            shrink = divider.height
            cbar = plt.colorbar(sm, ax=ax, shrink=shrink)
        
        plt.title('Intermediate probabilities')
        plt.show()

    """
    Function to plot x, y, and z-values for each qubit during the last execution of a circuit.
    If parameter noisy_circuit is defined, the x, y, and z values for that circuit will be shown in red.
    """
    @staticmethod
    def plot_intermediate_states_per_qubit(ideal_circuit:Circuit, noisy_circuit:Circuit=None):
        for q in range(ideal_circuit.N):
            x_measures_ideal = ideal_circuit.get_x_measures(q)
            y_measures_ideal = ideal_circuit.get_y_measures(q)
            z_measures_ideal = ideal_circuit.get_z_measures(q)
           
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
            ax1.plot(x_measures_ideal, 'b', label='x ideal')
            ax1.set_ylim(-1.0, 1.0)
            ax1.set_title(f'X for qubit {q}')
            ax1.set_ylabel('X')
            ax2.plot(y_measures_ideal, 'b', label='y ideal')
            ax2.set_ylim(-1.0, 1.0)
            ax2.set_title(f'Y for qubit {q}')
            ax2.set_ylabel('Y')
            ax3.plot(z_measures_ideal, 'b', label='z ideal')
            ax3.set_ylim(-1.0, 1.0)
            ax3.set_title(f'Z for qubit {q}')
            ax3.set_xlabel('Circuit depth')
            ax3.set_ylabel('Z')

            if not noisy_circuit is None:
                x_measures_noisy = noisy_circuit.get_x_measures(q)
                y_measures_noisy = noisy_circuit.get_y_measures(q)
                z_measures_noisy = noisy_circuit.get_z_measures(q)
                ax1.plot(x_measures_noisy, 'r', label='x noisy')
                ax2.plot(y_measures_noisy, 'r', label='y noisy')
                ax3.plot(z_measures_noisy, 'r', label='z noisy')
                ax1.legend(loc='upper right')
                ax2.legend(loc='upper right')
                ax3.legend(loc='upper right')


    """
    Function to create an animation of the execution of a noisy quantum circuit using Bloch spheres.
    Parameter ideal_circuit should have the same number of qubits and the same gate operations as noisy_circuit, 
    but without decoherence or quantum noise.
    """
    @staticmethod
    def create_animation(ideal_circuit:NoisyCircuit, noisy_circuit:NoisyCircuit=None):

        # Define the number of frames for the animation
        num_frames = len(ideal_circuit.get_x_measures(0))

        # Create a figure for the plot
        fig_width = 3 * ideal_circuit.N
        fig_height = 4
        fig = plt.figure()
        fig.set_size_inches(fig_width, fig_height)

        # Create a Bloch sphere object for each qubit
        b = []
        for q in range(ideal_circuit.N):
            ax = fig.add_subplot(1, ideal_circuit.N, q+1, projection='3d')
            b.append(Bloch(fig=fig, axes=ax))

        # Function to update the Bloch sphere for each frame
        def animate(i):
            for q in range(ideal_circuit.N):
                # Clear the previous vectors and points
                b[q].clear()  

                # Define the state vector for the ideal circuit
                x = ideal_circuit.get_x_measures(q)[i]
                y = ideal_circuit.get_y_measures(q)[i]
                z = ideal_circuit.get_z_measures(q)[i]
                ideal_state_vector = np.array([x, y, z])

                # Add the ideal state to the Bloch sphere
                b[q].add_vectors(ideal_state_vector)

                # Define the state vector for the noisy circuit
                if not noisy_circuit is None:
                    x = noisy_circuit.get_x_measures(q)[i]
                    y = noisy_circuit.get_y_measures(q)[i]
                    z = noisy_circuit.get_z_measures(q)[i]
                    noisy_state_vector = np.array([x, y, z])

                    # Add the noisy state to the Bloch sphere
                    b[q].add_vectors(noisy_state_vector)

                # Green is ideal state, red is noisy state
                b[q].vector_color = ['g', 'r']

                # Redraw the Bloch sphere
                b[q].make_sphere()  

        # Create an animation
        ani = animation.FuncAnimation(fig, animate, frames=num_frames, repeat=False)

        return ani