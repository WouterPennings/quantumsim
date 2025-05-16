import numpy as np
import scipy
import scipy.sparse as sparse
from quantumsim_performante.pulse import *

constant_pulse = ConstantPulse()
constant_pulse_numerical = ConstantPulseNumerical()
gaussian_pulse = GaussianPulse(loc=0.5, scale=0.25)

""" 
Evaluates the integrals coming up in the Noisy gates approach for different pulse waveforms.
Because many of the integrals are evaluated many times with the same parameters, we can apply caching to speed things
up.
"""
class Integrator(object):
    """Calculates the integrals for a specific pulse parametrization.

    Args:
        pulse (Pulse): Object specifying the pulse waveform and parametrization.

    Attributes:
        pulse_parametrization (callable): Function F: [0,1] -> [0,1] representing the parametrization of the pulse.
        use_lookup (bool): Tells whether or not the lookup table of the analytical solution should be used.
    """

    _INTEGRAL_LOOKUP = {
        "sin(theta/a)**2": lambda theta, a: np.sin(theta/a)**2,
        "sin(theta/(2*a))**4": lambda theta, a: np.sin(theta/(2*a))**4,
        "sin(theta/a)*sin(theta/(2*a))**2": lambda theta, a: np.sin(theta/a)*np.sin(theta/(2*a))**2,
        "sin(theta/(2*a))**2": lambda theta, a: np.sin(theta/(2*a))**2,
        "cos(theta/a)**2": lambda theta, a: np.cos(theta/a)**2,
        "sin(theta/a)*cos(theta/a)": lambda theta, a: np.sin(theta/a)*np.cos(theta/a),
        "sin(theta/a)": lambda theta, a: np.sin(theta/a),
        "cos(theta/(2*a))**2": lambda theta, a: np.cos(theta/(2*a))**2
    }
    # For each key (integrand), we calculated the result (parametric integral from 0 to theta) using the parametrization
    # theta(t,t0) = omega(t-t0)/a, corresponding to a square pulse, which is one that has constant magnitude.
    _RESULT_LOOKUP = {
        "sin(theta/a)**2": lambda theta, a: a*(2*theta - np.sin(2*theta))/(4*theta),
        "sin(theta/(2*a))**4": lambda theta, a: a*(6*theta-8*np.sin(theta)+np.sin(2*theta))/(16*theta),
        "sin(theta/a)*sin(theta/(2*a))**2": lambda theta, a: a*((np.sin(theta/2))**4)/theta,
        "sin(theta/(2*a))**2": lambda theta, a: a*(theta - np.sin(theta))/(2 * theta),
        "cos(theta/a)**2": lambda theta, a: a*(2*theta + np.sin(2*theta))/(4*theta),
        "sin(theta/a)*cos(theta/a)": lambda theta, a: a*(np.sin(theta))**2/(2*theta),
        "sin(theta/a)": lambda theta, a: a*(1-np.cos(theta))/theta,
        "cos(theta/(2*a))**2": lambda theta, a: a*(theta + np.sin(theta))/(2*theta)
    }

    def __init__(self, pulse: Pulse):
        self.pulse_parametrization = pulse.get_parametrization()
        self.use_lookup = pulse.use_lookup
        self._cache = dict()

    def integrate(self, integrand: str, theta: float, a: float) -> float:
        """ Evaluates the integrand provided as string from zero to a based on the implicit pulse shape scaled by theta.

        If the pulse (pulse_parametrization) is None, we assume that the pulse height is constant. In this case, we do
        not perform numerical calculation but just lookup the result.

        Args:
            integrand (str): Name of the integrand.
            theta (str): Upper limit of the integration. Total area of the pulse waveform.
            a (str): Scaling parameter.

        Returns:
            Integration result as float.
        """

        # Caching
        if (integrand, theta, a) in self._cache:
            return self._cache[(integrand, theta, a)]

        # Input validation
        assert integrand in self._INTEGRAL_LOOKUP.keys(), "Unknown integrand."
        assert a > 0, f"Require non-vanishing gate time but found a = {a}."

        # Pulse is constant -> We can lookup the analytical result
        if self.use_lookup:
            y = self._analytical_integration(integrand, theta, a)

        # Pulse is variable
        else:
            y = self._numerical_integration(integrand, theta, a)

        # Caching
        self._cache[(integrand, theta, a)] = y

        return y

    def _analytical_integration(self, integrand_str: str, theta: float, a: float) -> float:
        """Lookups up the result of the integration for the case that the parametrization is None.

        Note:
            This method can/should only be used when the pulse height is constant. Otherwise, the result would be wrong.

        Args:
            integrand_str (str): Name of the integrand.
            theta (float): Upper limit of the integration. Total area of the pulse waveform.
            a (float): Scaling parameter.
        """
        integral = self._RESULT_LOOKUP[integrand_str]
        return integral(theta, a)

    def _numerical_integration(self, integrand_name: str, theta: float, a: float) -> float:
        """Looks up the integrand as function and performs numerical integration from 0 to theta.

        Uses the the parametrization specified in the class instance.

        Args:
            integrand_name (str): Name of the integrand.
            theta (float): Upper limit of the integration. Total area of the pulse waveform.
            a (float): Scaling parameter.

        Returns:
            Result of the integration as float.
        """
        integrand = self._INTEGRAL_LOOKUP[integrand_name]

        # The parametrization is a monotone function with param(t=0) == 0 and param(t=1) == 1.
        param = self.pulse_parametrization

        # We scale this parametrization such that scaled_param(t=0) == 0 and scaled_param(t=1) == theta.
        scaled_param = lambda t: param(t) * theta

        # We parametrize the integrand and integrate it from 0 to a. Integral should go from 0 to a.
        integrand_p = lambda t: integrand(scaled_param(t), a)
        y, abserr = scipy.integrate.quad(integrand_p, 0, a)

        return y
    
# Create the integrator for the noisy gates to use
integrator = Integrator(constant_pulse)


# Functions in this class are adapted from `quantum-gates`:
# Source: https://pypi.org/project/quantum-gates/
# License: MIT License
# Original Authors: M. Grossi, G. D. Bartolomeo, M. Vischi, P. Da Rold, R. Wixinger
class NoisyGate:
    @staticmethod
    def __get_unitary_contribution(theta, phi):
        """Unitary contribution due to drive Hamiltonian.

        Args:
            theta (float): Angle of rotation on the Bloch sphere.
            phi (float): Phase of the drive defining axis of rotation on the Bloch sphere.

        Returns:
            Array representing the unitary contribution due to drive Hamiltonian.
        """
        return np.array(
            [[np.cos(theta/2), - 1J * np.sin(theta/2) * np.exp(-1J * phi)],
             [- 1J * np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )
    
    @staticmethod
    def __ito_integrals_for_X_Y_sigma_min(theta):
        """Ito integrals.

        Ito integrals for the following processes:
            * depolarization for X(t)
            * depolarization for Y(t)
            * relaxation for sigma_min(t).

        As illustration, we leave the variables names for X(t) in the calculation.

        Args:
            theta (float): Angle of rotation on the Bloch sphere.

        Returns:
            Tuple of floats representing sampled results of the Ito integrals.
        """
        # Integral of sin(theta)**2
        Vdx_1 = integrator.integrate("sin(theta/a)**2", theta, 1)

        # Integral of sin**4(theta/2)
        Vdx_2 = integrator.integrate("sin(theta/(2*a))**4", theta, 1)

        # Integral of sin(theta) sin**2(theta/2)
        Covdx_12 = integrator.integrate("sin(theta/a)*sin(theta/(2*a))**2", theta, 1)

        # Integral of sin(theta)
        Covdx_1Wdx = integrator.integrate("sin(theta/a)", theta, 1)

        # Integral of sin**2(theta/2)
        Covdx_2Wdx = integrator.integrate("sin(theta/(2*a))**2", theta, 1)

        # Mean and covariance
        meand_x = np.array([0, 0, 0])
        covd_x = np.array([[Vdx_1, Covdx_12, Covdx_1Wdx], [Covdx_12, Vdx_2, Covdx_2Wdx], [Covdx_1Wdx, Covdx_2Wdx, 1]])

        # Sampling
        sample_dx = np.random.multivariate_normal(meand_x, covd_x, 1) # The variance of Wr is 1
        Idx1 = sample_dx[0,0]
        Idx2 = sample_dx[0,1]
        Wdx = sample_dx[0,2]

        return Idx1, Idx2, Wdx

    @staticmethod
    def __ito_integrals_for_Z(theta):
        """Ito integrals.

        Ito integrals for the following processes:
            * depolarization for Z(t)
            * relaxation for Z(t).

        As illustration, we leave the variable names for the depolarization Itô processes depending on Z(t).

        Args:
            theta (float): angle of rotation on the Bloch sphere.

        Returns:
             Tuple of floats representing sampled results of the Ito integrals.
        """

        # Integral of cos(theta)**2
        Vdz_1 = integrator.integrate("cos(theta/a)**2", theta, 1)

        # Integral of sin(theta)**2
        Vdz_2 = integrator.integrate("sin(theta/a)**2", theta, 1)

        # Integral of sin(theta)*cos(theta)
        Covdz_12 = integrator.integrate("sin(theta/a)*cos(theta/a)", theta, 1)

        # Mean and covariance
        meand_z = np.array([0,0])
        covd_z = np.array(
            [[Vdz_1,Covdz_12],
             [Covdz_12, Vdz_2]]
        )

        # Sampling
        sample_dz = np.random.multivariate_normal(meand_z, covd_z, 1)
        Idz1 = sample_dz[0,0]
        Idz2 = sample_dz[0,1]

        return Idz1, Idz2

    @staticmethod
    def __get_depolarization_contribution(theta, phi, ed):
        # Variances and covariances for depolarization Itô processes depending on X(t)
        Idx1, Idx2, Wdx = NoisyGate.__ito_integrals_for_X_Y_sigma_min(theta)
        Idx = ed * np.array([[np.sin(phi)*Idx1,Wdx + (np.exp(-2*1J*phi)-1)*Idx2],[Wdx + (np.exp(+2*1J*phi)-1)*Idx2,-np.sin(phi)*Idx1]])

        #Variances and covariances for depolarization Itô processes depending on Y(t)
        Idy1, Idy2, Wdy = NoisyGate.__ito_integrals_for_X_Y_sigma_min(theta)
        Idy = ed * np.array([[-np.cos(phi)*Idy1, -1J*Wdy + 1J*(np.exp(-2*1J*phi)+1)*Idy2], [1J*Wdy - 1J*(np.exp(2*1J*phi)+1)*Idy2, np.cos(phi)*Idy1]])

        # Variances and covariances for depolarization Itô processes depending on Z(t)
        Idz1, Idz2 = NoisyGate.__ito_integrals_for_Z(theta)
        Idz = ed * np.array(
            [[Idz1, -1J * np.exp(-1J*phi) * Idz2],
             [1J * np.exp(1J*phi) * Idz2, -Idz1]]
        )

        return Idx, Idy, Idz

    @staticmethod
    def __deterministic_relaxation(theta):
        """Deterministic contribution given by relaxation

        Args:
            theta (float): angle of rotation on the Bloch sphere.

        Returns:
            Array representing the deterministic part of the relaxation process.
        """

        # Integral of sin(theta/(2*a))**2
        det1 = integrator.integrate("sin(theta/(2*a))**2", theta, 1)

        # Integral of sin(theta)
        det2 = integrator.integrate("sin(theta/a)", theta, 1)

        # Integral of cos(theta/2)**2
        det3 = integrator.integrate("cos(theta/(2*a))**2", theta, 1)

        return det1, det2, det3

    @staticmethod
    def __get_relaxation_contribution(theta, phi, ep, e1):
        # Variances and covariances for relaxation Itô processes depending on sigma_min(t)
        Ir1, Ir2, Wr = NoisyGate.__ito_integrals_for_X_Y_sigma_min(theta)
        Ir = e1 * np.array([[-1J/2 * np.exp(1J*phi) * Ir1, Wr - Ir2], [np.exp(2*1J*phi)*Ir2,1J/2* np.exp(1J*phi) * Ir1]])

        # Deterministic contribution given by relaxation
        det1, det2, det3 = NoisyGate.__deterministic_relaxation(theta)
        deterministic = -e1**2/2 * np.array([[det1, 1J/2*np.exp(-1J*phi)*det2], [-1J/2*np.exp(1J*phi)*det2, det3]])

        # Variances and covariances for relaxation Itô processes depending on Z(t)
        Ip1, Ip2 = NoisyGate.__ito_integrals_for_Z(theta)
        Ip = ep * np.array([[Ip1, -1J * np.exp(-1J*phi) * Ip2], [1J * np.exp(1J*phi) * Ip2, -Ip1]])

        return Ir, deterministic, Ip
    
    @staticmethod
    def construct(theta, phi, p, T1, T2):
        """Constructs a noisy single qubit gate. 

        Args:
            theta (float): Angle of rotation on the Bloch sphere.
            phi (float): Phase of the drive defining axis of rotation on the Bloch sphere.
            lam (float): Z rotation.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.

        Returns:
              Array representing a general single-qubit noisy quantum gate.
        """

        """ 0) CONSTANTS """

        tg = 35 * 10**(-9)
        ed = np.sqrt(p/4)

        # Amplitude damping time is zero
        if T1 == 0:
            e1 = 0
        else:
            e1 = np.sqrt(tg/T1)

        # Dephasing time is zero
        if T2 == 0:
            ep = 0
        else:
            e2 = np.sqrt(tg/T2)
            ep = np.sqrt((1/2) * (e2**2 - e1**2/2))

        """ 2) DEPOLARIZATION CONTRIBUTION """
        Idx, Idy, Idz = NoisyGate.__get_depolarization_contribution(theta, phi, ed)

        """ 3) RELAXATION CONTRIBUTION """
        Ir, deterministic, Ip = NoisyGate.__get_relaxation_contribution(theta, phi, ep, e1)

        """ 4) COMBINE CONTRIBUTIONS """
        return NoisyGate.__get_unitary_contribution(theta, phi) @ scipy.linalg.expm(deterministic) @ scipy.linalg.expm(1J * Idx + 1J * Idy + 1J * Idz + 1J * Ir + 1J * Ip)
    
    @staticmethod
    def _ito_integrals_for_depolarization_process(omega, phi, a) -> tuple[float]:
        """ Ito integrals.

         Used for the depolarization Itô processes depending on one of
            * [tensor(ID,Z)](t)
            * [tensor(X,ID)](t)
            * [tensor(Y,ID)](t)
            * [tensor(sigma_min,ID)](t)

        As illustration, we leave the variable names from the version with [tensor(ID,Z)](t).

        Args:
            omega: integral of theta from t0 to t1.
            phi: phase of the drive defining axis of rotation on the Bloch sphere.
            a: fraction representing CR gate time / gate time.

        Returns:
            Tuple of floats representing sampled results of the Ito integrals.
        """

        # Integral of cos(omega/a)**2
        Vp_trg_1 = integrator.integrate("cos(theta/a)**2", omega, a)

        # Integral of sin(omega/a)**2
        Vp_trg_2 = integrator.integrate("sin(theta/a)**2", omega, a)

        # Integral of sin(omega/a)*cos(omega/a)
        Covp_trg_12 = integrator.integrate("sin(theta/a)*cos(theta/a)", omega, a)

        # Mean and covariance
        meanp_trg = [0, 0]
        covp_trg = [[Vp_trg_1, Covp_trg_12], [Covp_trg_12, Vp_trg_2]]

        # Sample
        sample_p_trg = np.random.multivariate_normal(meanp_trg, covp_trg, 1)
        Ip_trg_1 = sample_p_trg[0,0]
        Ip_trg_2 = sample_p_trg[0,1]

        return Ip_trg_1, Ip_trg_2

    @staticmethod
    def _ito_integrals_for_depolarization_process_reversed_tensor(omega, a) -> tuple[float]:
        """ Ito integrals.

        Used for the depolarization Itô processes depending on one of
            * [tensor(ID,X)](t)
            * [tensor(ID,Y)](t)

        As illustration, we leave the variable names from the version with [tensor(ID,Y)](t).

        Args:
            omega (float): Integral of theta from t0 to t1.
            a (float): Fraction representing CR gate time / gate time.

        Returns:
            Tuple of floats representing sampled results of the Ito integrals.
        """

        # Integral of sin**2(omega/a)
        Vdy_trg_1 = integrator.integrate("sin(theta/a)**2", omega, a)

        # Integral of sin(omega/(2*a))**4
        Vdy_trg_2 = integrator.integrate("sin(theta/(2*a))**4", omega, a)

        # Integral of sin(omega/a) sin**2(omega/(2*a))
        Covdy_trg_12 = integrator.integrate("sin(theta/a)*sin(theta/(2*a))**2", omega, a)

        # Integral of sin(omega/a)
        Covdy_trg_1Wdy = integrator.integrate("sin(theta/a)", omega, a)

        # Integral of sin(omega/(2*a))**2
        Covdy_trg_2Wdy = integrator.integrate("sin(theta/(2*a))**2", omega, a)

        meandy_trg = np.array([0, 0, 0])
        covdy_trg = np.array(
            [[Vdy_trg_1, Covdy_trg_12, Covdy_trg_1Wdy],
             [Covdy_trg_12, Vdy_trg_2, Covdy_trg_2Wdy],
             [Covdy_trg_1Wdy, Covdy_trg_2Wdy, a]]
        )

        # The variance of Wdy is a
        sample_dy_trg = np.random.multivariate_normal(meandy_trg, covdy_trg, 1)

        Idy_trg_1 = sample_dy_trg[0,0]
        Idy_trg_2 = sample_dy_trg[0,1]
        Wdy_trg = sample_dy_trg[0,2]

        return Idy_trg_1, Idy_trg_2,  Wdy_trg

    @staticmethod
    def __get_cr_gate_contribution(theta, phi, t_cr, p, c_T1, c_T2, t_T1, t_T2):
        """Generates a CR gate.

        This is the 2 order approximated solution, non-unitary matrix. It implements the CR two-qubit noisy quantum gate
        with depolarizing and relaxation errors on both qubits during the unitary evolution.

        Args:
            theta (float): Angle of rotation on the Bloch sphere.
            phi (float): Phase of the drive defining axis of rotation on the Bloch sphere.
            t_cr (float): CR gate time in ns.
            p (float): Depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.

        Returns:
              CR two-qubit noisy quantum gate (numpy array)
        """

        """ 0) CONSTANTS """

        tg = 35 * 10**(-9)
        omega = theta
        a = t_cr / tg
        assert t_cr > 0, f"Expected t_cr to be > 0 but found {t_cr}"
        assert tg > 0, f"Expected tg to be > 0 but found {tg}"
        ed_cr = np.sqrt(p/(4*a))

        if c_T1 == 0:
            e1_ctr = 0
        else:
            e1_ctr = np.sqrt(tg/c_T1)

        if c_T2 == 0:
            ep_ctr = 0
        else:
            e2_ctr = np.sqrt(tg/c_T2)
            ep_ctr = np.sqrt((1/2) * (e2_ctr**2 - e1_ctr**2/2))

        if t_T1 == 0:
            e1_trg = 0
        else:
            e1_trg = np.sqrt(tg/t_T1)

        if t_T2 == 0:
            ep_trg = 0
        else:
            e2_trg = np.sqrt(tg/t_T2)
            ep_trg = np.sqrt((1/2) * (e2_trg**2 - e1_trg**2/2))

        U = np.array(
            [[np.cos(theta/2), -1J*np.sin(theta/2) * np.exp(-1J * phi), 0, 0],
             [-1J*np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2), 0, 0],
             [0, 0, np.cos(theta/2), 1J*np.sin(theta/2) * np.exp(-1J * phi)],
             [0, 0, 1J*np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )

        """ 1) RELAXATION CONTRIBUTIONS """

        # Variances and covariances for amplitude damping Itô processes depending on [tensor(sigma_min,ID)](t)
        Ir_ctr_1, Ir_ctr_2 = NoisyGate._ito_integrals_for_depolarization_process(omega, phi, a)

        Ir_ctr = e1_ctr * np.array(
            [[0, 0, Ir_ctr_1, 1J*Ir_ctr_2 * np.exp(-1J * phi)],
             [0, 0, 1J*Ir_ctr_2 * np.exp(1J * phi), Ir_ctr_1],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        )

        # Variances and covariances for amplitude damping Itô processes depending on [tensor(ID,sigma_min)](t)
        Ir_trg_1, Ir_trg_2, Wr_trg = NoisyGate._ito_integrals_for_depolarization_process_reversed_tensor(omega, a)
        Ir_trg = e1_trg * np.array(
            [[-1J*(1/2)*Ir_trg_1*np.exp(1J*phi), Wr_trg-Ir_trg_2, 0, 0],
             [Ir_trg_2*np.exp(2*1J*phi), 1J*(1/2)*Ir_trg_1*np.exp(1J*phi), 0, 0],
             [0, 0, 1J*(1/2)*Ir_trg_1*np.exp(1J*phi),Wr_trg-Ir_trg_2],
             [0, 0, Ir_trg_2*np.exp(2*1J*phi), -1J*(1/2)*Ir_trg_1*np.exp(1J*phi)]]
        )

        # Variances and covariances for phase damping Itô processes depending on [tensor(Z,ID)](t)
        Wp_ctr = np.random.normal(0, np.sqrt(a))
        Ip_ctr = ep_ctr * np.array(
            [[Wp_ctr, 0, 0, 0],
             [0, Wp_ctr, 0, 0],
             [0, 0, -Wp_ctr, 0],
             [0, 0, 0, -Wp_ctr]]
        )

        # Variances and covariances for phase damping Itô processes depending on [tensor(ID,Z)](t)
        Ip_trg_1, Ip_trg_2 = NoisyGate._ito_integrals_for_depolarization_process(omega, phi, a)
        Ip_trg = ep_trg * np.array(
            [[Ip_trg_1, -1J*Ip_trg_2*np.exp(-1J*phi), 0, 0],
             [1J*Ip_trg_2*np.exp(1J*phi), -Ip_trg_1, 0, 0],
             [0, 0, Ip_trg_1, 1J*Ip_trg_2*np.exp(-1J*phi)],
             [0, 0, -1J*Ip_trg_2*np.exp(1J*phi), -Ip_trg_1]]
        )

        #Deterministic contribution given by relaxation
        det1 = (a*omega-a*np.sin(omega))/(2*omega)
        det2 = (a/omega)*(1-np.cos(omega))
        det3 = a/(2*omega)*(omega+np.sin(omega))

        deterministic_r_ctr = -e1_ctr**2/2 * np.array([[0,0,0,0],[0,0,0,0],[0,0,a,0],[0,0,0,a]])
        deterministic_r_trg = -e1_trg**2/2 * np.array(
            [[det1,1J*(1/2)*det2*np.exp(-1J*phi),0,0],
             [-1J*(1/2)*det2*np.exp(1J*phi),det3,0,0],
             [0,0,det1,-1J*(1/2)*det2*np.exp(-1J*phi)],[0,0,1J*(1/2)*det2*np.exp(1J*phi),det3]]
        )

        """ 2) DEPOLARIZATION CONTRIBUTIONS """

        # Variances and covariances for depolarization Itô processes depending on [tensor(X,ID)](t)
        Idx_ctr_1, Idx_ctr_2 = NoisyGate._ito_integrals_for_depolarization_process(omega, phi, a)
        Idx_ctr = ed_cr * np.array(
            [[0, 0, Idx_ctr_1, 1J*Idx_ctr_2 * np.exp(-1J * phi)],
             [0, 0, 1J*Idx_ctr_2 * np.exp(1J * phi), Idx_ctr_1],
             [Idx_ctr_1, -1J*Idx_ctr_2 * np.exp(-1J * phi), 0, 0],
             [-1J*Idx_ctr_2 * np.exp(1J * phi), Idx_ctr_1, 0, 0]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(Y,ID)](t)
        Idy_ctr_1, Idy_ctr_2 = NoisyGate._ito_integrals_for_depolarization_process(omega, phi, a)
        Idy_ctr = ed_cr * np.array(
            [[0, 0, -1J*Idy_ctr_1, Idy_ctr_2 * np.exp(-1J * phi)],
             [0, 0, Idy_ctr_2 * np.exp(1J * phi), -1J*Idy_ctr_1],
             [1J*Idy_ctr_1, Idy_ctr_2 * np.exp(-1J * phi), 0, 0],
             [Idy_ctr_2 * np.exp(1J * phi), 1J*Idy_ctr_1, 0, 0]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(Z,ID)](t)
        Wdz_ctr = np.random.normal(0, np.sqrt(a))
        Idz_ctr = ed_cr * np.array(
            [[Wdz_ctr, 0, 0, 0],
             [0, Wdz_ctr, 0, 0],
             [0, 0, -Wdz_ctr, 0],
             [0, 0, 0, -Wdz_ctr]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(ID,X)](t)
        Idx_trg_1, Idx_trg_2, Wdx_trg = NoisyGate._ito_integrals_for_depolarization_process_reversed_tensor(omega, a)

        Idx_trg = ed_cr * np.array(
            [[Idx_trg_1 * np.sin(phi), Wdx_trg + (np.exp(-2*1J*phi)-1)*Idx_trg_2, 0, 0],
             [Wdx_trg + (np.exp(2*1J*phi)-1)*Idx_trg_2, -Idx_trg_1*np.sin(phi), 0, 0],
             [0,  0, -Idx_trg_1 * np.sin(phi), Wdx_trg + (np.exp(-2*1J*phi)-1)*Idx_trg_2],
             [0, 0, Wdx_trg + (np.exp(2*1J*phi)-1)*Idx_trg_2, Idx_trg_1 * np.sin(phi)]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(ID,Y)](t)
        Idy_trg_1, Idy_trg_2,  Wdy_trg = NoisyGate._ito_integrals_for_depolarization_process_reversed_tensor(omega, a)
        Idy_trg = ed_cr * np.array(
            [[-Idy_trg_1*np.cos(phi), -1J*Wdy_trg + 1J * (np.exp(-2*1J*phi)+1)*Idy_trg_2, 0, 0],
             [1J*Wdy_trg - 1J * (np.exp(2*1J*phi)+1)*Idy_trg_2, Idy_trg_1*np.cos(phi), 0, 0],
             [0, 0, Idy_trg_1*np.cos(phi), -1J*Wdy_trg + 1J * (np.exp(-2*1J*phi)+1)*Idy_trg_2],
             [0, 0, 1J*Wdy_trg - 1J * (np.exp(2*1J*phi)+1)*Idy_trg_2, -Idy_trg_1*np.cos(phi)]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(ID,Z)](t)
        Idz_trg_1, Idz_trg_2 = NoisyGate._ito_integrals_for_depolarization_process(omega, phi, a)
        Idz_trg = ed_cr * np.array(
            [[Idz_trg_1, -1J*Idz_trg_2*np.exp(-1J*phi), 0, 0],
             [1J*Idz_trg_2*np.exp(1J*phi), -Idz_trg_1, 0, 0],
             [0, 0, Idz_trg_1, 1J*Idz_trg_2*np.exp(-1J*phi)],
             [0, 0, -1J*Idz_trg_2*np.exp(1J*phi), -Idz_trg_1]]
        )

        """ 4) COMBINE CONTRIBUTIONS """
        return U @ scipy.linalg.expm(deterministic_r_ctr + deterministic_r_trg) \
                 @ scipy.linalg.expm(
            1J * Ir_ctr + 1J * Ir_trg
            + 1J * Ip_ctr + 1J * Ip_trg
            + 1J * Idx_ctr + 1J * Idy_ctr + 1J * Idz_ctr
            + 1J * Idx_trg + 1J * Idy_trg + 1J * Idz_trg
        )
    
    @staticmethod
    def __get_relaxation_gate_contribution(Dt, T1, T2):
        """Generates the noisy gate for combined amplitude and phase damping.

        This is the exact solution, a non-unitary matrix. It implements the single-qubit relaxation error on idle
        qubits.

        Args:
            Dt (float): idle time in ns.
            T1 (float): qubit's amplitude damping time in ns.
            T2 (float): qubit's dephasing time in ns.

        Returns:
              Array representing the amplitude and phase damping noise gate.
        """
        # Constants
        # tg = 561.778 # Gate execution time in nanoseconds as provided by Qiskit's ibmb_kyiv device gate time median
        tg = 35 * 10**(-9)
        Dt = Dt / tg

        # Helper function
        def V(Dt) -> float:
            return 1-np.exp(-e1**2 * Dt)

        # Calculations
        if T1 == 0:
            e1 = 0
        else:
            e1 = np.sqrt(tg/T1)

        if T2 == 0:
            ep = 0
        else:
            e2 = np.sqrt(tg/T2)
            ep = np.sqrt((1/2) * (e2**2 - e1**2/2))

        W = np.random.normal(0, np.sqrt(Dt))
        I = np.random.normal(0, np.sqrt(V(Dt)))
        result = np.array(
            [[np.exp(1J * ep * W), 1J * I * np.exp(-1J * ep * W)],
             [0, np.exp(-e1**2/2 * Dt) * np.exp(-1J * ep * W)]]
        )
        return result

    @staticmethod
    def construct_cnot(c_phi, t_phi, t_cnot, p_cnot, c_p, t_p,
                  c_T1, c_T2, t_T1, t_T2):
        """Generates a noisy CNOT gate.

        This is a 2nd order approximated solution, a non-unitary matrix. It implements the CNOT two-qubit noisy quantum
        gate with depolarizing and relaxation errors on both qubits during the unitary evolution.

        Args:
            c_phi (float): Control qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_phi (float): Target qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_cnot (float): CNOT gate time in ns
            p_cnot (float): CNOT depolarizing error probability.
            c_p (float): Qubit depolarizing error probability for contorl qubit.
            t_p (float): Qubit depolarizing error probability for target qubit.
            c_T1 (float): Qubit's amplitude damping time in ns for control qubit.
            t_T1 (float): Qubit's amplitude damping time in ns for target qubit.
            c_T2 (float): Qubit's dephasing time in ns for control qubit.
            t_T2 (float): Qubit's dephasing time in ns for target qubit.

        Returns:
              Array representing a CNOT two-qubit noisy quantum gate.
        """

        """ 0) CONSTANTS """
        tg = 35*10**(-9)
        t_cr = t_cnot/2-tg
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_cnot)**2 / ((1-(3/4)*c_p)**2 * (1-(3/4)*t_p)))))

        """ 1) CR gate contributions """
        first_cr = NoisyGate.__get_cr_gate_contribution(-np.pi/4, -t_phi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)
        second_cr = NoisyGate.__get_cr_gate_contribution(np.pi/4, -t_phi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)

        """ 2) X/Sqrt(X) contributions """
        x_gate = NoisyGate.construct(np.pi, -c_phi+np.pi/2, c_p, c_T1, c_T2)
        sx_gate = NoisyGate.construct(np.pi / 2, -t_phi, t_p, t_T1, t_T2)
        Y_Rz = NoisyGate.construct(-np.pi, -c_phi + np.pi/2 + np.pi/2, c_p, c_T1, c_T2)

        """ 3) Relaxation contribution """
        relaxation_gate = NoisyGate.__get_relaxation_gate_contribution(tg, t_T1, t_T2)

        """ 4) COMBINE CONTRIBUTIONS """
        r = first_cr @ np.kron(x_gate, relaxation_gate) @ second_cr @ np.kron(Y_Rz, sx_gate)
        r = sparse.coo_matrix(r)
        return r
    
    @staticmethod
    def construct_cnot_inverse(c_phi, t_phi, t_cnot, p_cnot, c_p, t_p,
                  c_T1, c_T2, t_T1, t_T2):
        """Generates an reversed noisy CNOT gate.

        This is a 2nd order approximated solution, a non-unitary matrix. It implements the CNOT two-qubit noisy quantum
        gate with depolarizing and relaxation errors on both qubits during the unitary evolution.

        Args:
            c_phi (float): Control qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_phi (float): Target qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_cnot (float): CNOT gate time in ns
            p_cnot (float): CNOT depolarizing error probability.
            c_p (float): Qubit depolarizing error probability for contorl qubit.
            t_p (float): Qubit depolarizing error probability for target qubit.
            c_T1 (float): Qubit's amplitude damping time in ns for control qubit.
            t_T1 (float): Qubit's amplitude damping time in ns for target qubit.
            c_T2 (float): Qubit's dephasing time in ns for control qubit.
            t_T2 (float): Qubit's dephasing time in ns for target qubit.

        Returns:
              Array representing the reverse CNOT two-qubit noisy quantum gate.
        """

        """ 0) CONSTANTS """
        tg = 35*10**(-9)
        t_cr = (t_cnot-3*tg)/2
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_cnot)**2 / ((1-(3/4)*c_p)**2 * (1-(3/4)*t_p)**3))))

        """ 1) CR gate contributions """
        first_cr = NoisyGate.__get_cr_gate_contribution(-np.pi/4, -c_phi-np.pi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)
        second_cr = NoisyGate.__get_cr_gate_contribution(np.pi/4, -c_phi-np.pi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)

        """ 2) X/Sqrt(X) contributions """
        Ry = NoisyGate.construct(-np.pi/2, -t_phi-np.pi/2+np.pi/2, t_p, t_T1, t_T2)
        Y_Z = NoisyGate.construct(np.pi/2, -c_phi-np.pi+np.pi/2, c_p, c_T1, c_T2)
        x_gate = NoisyGate.construct(np.pi, -t_phi-np.pi/2, t_p, t_T1, t_T2)
        first_sx_gate = NoisyGate.construct(np.pi/2, -c_phi - np.pi - np.pi/2, c_p, c_T1, c_T2)
        second_sx_gate = NoisyGate.construct(np.pi/2, -t_phi - np.pi/2, c_p, c_T1, c_T2)

        """ 3) Relaxation contribution """
        relaxation_gate = NoisyGate.__get_relaxation_gate_contribution(tg, c_T1, c_T2)

        """ 4) COMBINE CONTRIBUTIONS """
        r = np.kron(Ry, first_sx_gate) @ first_cr @ np.kron(x_gate, relaxation_gate) @ second_cr @ np.kron(second_sx_gate, Y_Z)
        r = sparse.coo_matrix(r)
        return r

    @staticmethod
    def construct_ecr(c_phi, t_phi, t_ecr, p_ecr, c_p, t_p,
                  c_T1, c_T2, t_T1, t_T2):
        """Generates a noisy ECR gate.

            This is a 2nd order approximated solution, a non-unitary matrix. It implements the ECR two-qubit noisy quantum
            gate with depolarizing and relaxation errors on both qubits during the unitary evolution.

            Args:
                c_phi (float): Control qubit phase of the drive defining axis of rotation on the Bloch sphere.
                t_phi (float): Target qubit phase of the drive defining axis of rotation on the Bloch sphere.
                t_ecr (float): ECR gate time in ns.
                p_ecr (float): ECR depolarizing error probability.
                c_p (float): Control qubit depolarizing error probability.
                t_p (float): Target qubit depolarizing error probability.
                c_T1 (float): Control qubit's amplitude damping time in ns.
                c_T2 (float): Control qubit's dephasing time in ns.
                t_T1 (float): Target qubit's amplitude damping time in ns.
                t_T2 (float): Target qubit's dephasing time in ns.

            Returns:
                Array representing a ECR two-qubit noisy quantum gate.
            """
        
        """ 0) CONSTANTS """
        tg = 35*10**(-9)
        t_cr = t_ecr/2-tg
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_ecr)**2 / ((1-(3/4)*c_p)**2 * (1-(3/4)*t_p)))))

        """ 1) CR gate contributions """
        first_cr = NoisyGate.__get_cr_gate_contribution(np.pi/4, np.pi-t_phi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)
        second_cr = NoisyGate.__get_cr_gate_contribution(-np.pi/4, np.pi-t_phi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)

        """ 2) X contribution """
        x_gate = -1J* NoisyGate.construct(np.pi, np.pi-c_phi, c_p, c_T1, c_T2)

        """ 3) Relaxation contribution """
        relaxation_gate = NoisyGate.__get_relaxation_gate_contribution(tg, t_T1, t_T2)
        
        """ 4) COMBINE CONTRIBUTIONS """
        return (first_cr @ np.kron(x_gate , relaxation_gate) @ second_cr)
    
    @staticmethod
    def construct_ecr_inverse(c_phi, t_phi, t_ecr, p_ecr, c_p, t_p,
                  c_T1, c_T2, t_T1, t_T2):
        """Generates a noisy inverse ECR gate.

        This is a 2nd order approximated solution, a non-unitary matrix. It implements the reverse ECR two-qubit noisy quantum
        gate with depolarizing and relaxation errors on both qubits during the unitary evolution.

        Args:
            c_phi (float): Control qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_phi (float): Target qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_ecr (float): ECR gate time in ns.
            p_ecr (float): ECR depolarizing error probability.
            c_p (float): Control qubit depolarizing error probability.
            t_p (float): Target qubit depolarizing error probability.
            c_T1 (float): Control qubit's amplitude damping time in ns.
            c_T2 (float): Control qubit's dephasing time in ns.
            t_T1 (float): Target qubit's amplitude damping time in ns.
            t_T2 (float): Target qubit's dephasing time in ns.

        Returns:
              Array representing a reverse ECR two-qubit noisy quantum gate.
        """
        """ 0) CONSTANTS """
        tg = 35*10**(-9)
        t_cr = t_ecr/2-tg
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_ecr)**2 / ((1-(3/4)*c_p)**2 * (1-(3/4)*t_p)))))

        """ 1) CR gate contributions """
        first_cr = NoisyGate.__get_cr_gate_contribution(np.pi/4, np.pi-t_phi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)
        second_cr = NoisyGate.__get_cr_gate_contribution(-np.pi/4, np.pi-t_phi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)

        """ 2) X/Sqrt(X) contributions """
        x_gate = -1J* NoisyGate.construct(np.pi, np.pi-c_phi, c_p, c_T1, c_T2)
        sx_gate_ctr_1 =  NoisyGate.construct(np.pi/2, -np.pi/2-c_phi, c_p, c_T1, c_T2)
        sx_gate_trg_1 =  NoisyGate.construct(np.pi/2, -np.pi/2-t_phi, t_p, t_T1, t_T2)
        sx_gate_ctr_2 =  NoisyGate.construct(np.pi/2, -np.pi/2-c_phi, c_p, c_T1, c_T2)
        sx_gate_trg_2 =  NoisyGate.construct(np.pi/2, -np.pi/2-t_phi, t_p, t_T1, t_T2)

        """ 3) Relaxation contribution """
        relaxation_gate = NoisyGate.__get_relaxation_gate_contribution(tg, t_T1, t_T2)

        """ 4) COMBINE CONTRIBUTIONS """
        return 1j * np.kron(sx_gate_ctr_1, sx_gate_trg_1) @ (first_cr @ np.kron(x_gate , relaxation_gate) @ second_cr ) @ np.kron(sx_gate_ctr_2, sx_gate_trg_2)
    