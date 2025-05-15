import json
import numpy as np
import os

class DeviceParameters(object):
    """Snapshot of the noise of the IBM backend. Can load and save the properties.

    Args:
        qubits_layout (list[int]): Layout of the qubits.

    Attributes:
        qubits_layout (list[int]): Layout of the qubits.
        nr_of_qubits (int): Number of qubits to be used.
        T1 (np.array): T1 time.
        T2 (np.array): T2 time.
        p (np.array): To be added.
        rout (np.array): To be added.
        p_int (np.array): Error probabilites in the 2 qubit gate.
        p_int (np.array): Gate time to implement controlled not operations in the 2 qubit gate.
        tm (np.array): To be added.
        dt (np.array): To be added.
        
    """

    def __init__(self):
        self.qubits_layout = None
        self.nr_of_qubits = None
        self.T1 = None
        self.T2 = None
        self.p = None
        self.rout = None
        self.p_int = None
        self.t_int = None
        self.tm = None
        self.dt = None
        self.metadata = None
        self._names = ["T1", "T2", "p", "rout", "p_int", "t_int", "tm", "dt", "metadata"]
        self._f_txt = ["T1.txt", "T2.txt", "p.txt", "rout.txt", "p_int.txt", "t_int.txt", "tm.txt", "dt.txt",
                       "metadata.json"]

    def load_from_json(self, location: str):
        """ Load device parameters from single json file at the location.
        """
        # Verify that it exists
        self._json_exists_at_location(location)

        # Load
        f = open(location)
        data_dict = json.load(f)

        # Check json keys
        if any((name not in data_dict for name in self._names)):
            raise Exception("Loading of device parameters from json not successful: At least one quantity is missing.")

        # Add lists to instance as arrays
        self.qubits_layout = np.array(data_dict["metadata"]["qubits_layout"])
        self.nr_of_qubits = data_dict["metadata"]["config"]["n_qubits"]
        self.T1 = np.array(data_dict["T1"])
        self.T2 = np.array(data_dict["T2"])
        self.p = np.array(data_dict["p"])
        self.rout = np.array(data_dict["rout"])
        self.p_int = np.array(data_dict["p_int"])
        self.t_int = np.array(data_dict["t_int"])
        self.tm = np.array(data_dict["tm"])
        self.dt = np.array(data_dict["dt"])
        self.metadata = data_dict["metadata"]

        # Verify
        if not self.is_complete():
            raise Exception("Loading of device parameters from json was not successful: Did not pass verification.")

        return

    def load_from_texts(self, location: str):
        """ Load device parameters from many text files at the location.
        """

        # Verify that exists
        self._texts_exist_at_location(location)

        # Load -> If the text has only one line, we have to make it into an 1x1 array explicitely.
        if self.nr_of_qubits == 1:
            # Here we use 'array' because with only one qubit 'loadtxt' doesn't load an array
            self.T1 = np.array([np.loadtxt(location + self.f_T1)])
            self.T2 = np.array([np.loadtxt(location + self.f_T2)])
            self.p = np.array([np.loadtxt(location + self.f_p)])
            self.rout = np.array([np.loadtxt(location + self.f_rout)])
            self.p_int = np.array([np.loadtxt(location + self.f_p_int)])
            self.t_int = np.array([np.loadtxt(location + self.f_t_int)])
            self.tm = np.array([np.loadtxt(location + self.f_tm)])
        else:
            self.T1 = np.loadtxt(location + self.f_T1)
            self.T2 = np.loadtxt(location + self.f_T2)
            self.p = np.loadtxt(location + self.f_p)
            self.rout = np.loadtxt(location + self.f_rout)
            self.p_int = np.loadtxt(location + self.f_p_int)
            self.t_int = np.loadtxt(location + self.f_t_int)
            self.tm = np.loadtxt(location + self.f_tm)
        self.dt = np.array([np.loadtxt(location + self.f_dt)])
        with open(location + self.f_metadata, "r") as metadata_file:
            self.metadata = json.load(metadata_file)

        # Verify
        if not self.is_complete():
            raise Exception("Loading of device parameters from text files was not successful: Did not pass verification.")

        return

    def get_as_tuple(self) -> tuple:
        """ Get the parameters as a tuple. The parameters have to be already loaded.
        """
        if not self.is_complete():
            raise Exception("Exception in DeviceParameters.get_as_tuble(): At least one of the parameters is None.")
        return self.T1, self.T2, self.p, self.rout, self.p_int, self.t_int, self.tm, self.dt, self.metadata

    def is_complete(self) -> bool:
        """ Returns whether all device parameters have been successfully initialized.
        """
        # Check not None
        return not any((
                self.T1 is None,
                self.T2 is None,
                self.p is None,
                self.rout is None,
                self.p_int is None,
                self.t_int is None,
                self.tm is None,
                self.dt is None,
                self.metadata is None))

    def check_T1_and_T2_times(self, do_raise_exception: bool) -> bool:
        """ Checks the T1 and T2 times. Raises an exception in case of invalid T1, T2 times if the flag is set. Returns
            whether or not all qubits are flawless.
        """

        print("Verifying the T1 and T2 times of the device: ")
        nr_bad_qubits = 0
        for i, (T1, T2) in enumerate(zip(self.T1, self.T2)):
            if T1 >= 2*T2:
                nr_bad_qubits += 1
                print('The qubit n.', self.qubits_layout[i], 'is bad.')
                print('Delete the affected qubit from qubits_layout and change the layout.')

        if nr_bad_qubits:
            print(f'Attention, there are {nr_bad_qubits} bad qubits.')
            print('In case of side effects contact Jay Gambetta.')
        else:
            print('All right!')

        if nr_bad_qubits and do_raise_exception:
            raise Exception(f'Stop simulation: The DeviceParameters class found {nr_bad_qubits} bad qubits.')

        return nr_bad_qubits == 0

    def _texts_exist_at_location(self, location):
        """ Checks if the text files with the device parameters exist at the expected location. Raises an exception
            if more than one text is missing.
        """
        missing = [f for f in self._f_txt if not os.path.exists(location + f)]
        if len(missing) > 0:
            raise FileNotFoundError(
                f"DeviceParameter found that at {location} the files {missing} are missing."
            )
        return

    def _json_exists_at_location(self, location):
        """ Checks if the json files with the device parameters exist, otherwise raises an exception.
        """
        if not os.path.exists(location):
            raise FileNotFoundError(
                f"DeviceParameter found that at {location} the file is missing."
            )
        return

    def __dict__(self):
        """ Get dict representation. """
        return {
            "T1": self.T1,
            "T2": self.T2,
            "p": self.p,
            "rout": self.rout,
            "p_int": self.p_int,
            "t_int": self.t_int,
            "tm": self.tm,
            "dt": self.dt,
            "metadata": self.metadata
        }

    def __str__(self):
        """ Representation as str. """
        return json.dumps(self.__dict__(), indent=4, default=default_serializer)

    def __eq__(self, other):
        """ Allows us to compare instances. """
        return self.__str__() == other.__str__()
