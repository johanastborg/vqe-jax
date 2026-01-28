import jax.numpy as jnp
from .utils import pauli_string_to_matrix

class Hamiltonian:
    """
    Represents a Hamiltonian as a sum of Pauli strings.
    H = sum(c_i * P_i)
    """
    def __init__(self, terms):
        """
        Args:
            terms: List of tuples (coefficient, pauli_string)
                   e.g. [(0.5, "Z"), (0.5, "X")]
        """
        self.terms = terms
        self.matrix = None

    def to_matrix(self):
        """Constructs the dense matrix representation of the Hamiltonian."""
        if self.matrix is not None:
            return self.matrix

        # Infer size from first term
        if not self.terms:
            raise ValueError("No terms in Hamiltonian")

        first_op = pauli_string_to_matrix(self.terms[0][1])
        dim = first_op.shape[0]
        H = jnp.zeros((dim, dim), dtype=complex)

        for coeff, p_str in self.terms:
            mat = pauli_string_to_matrix(p_str)
            H = H + coeff * mat

        self.matrix = H
        return H

    def expectation(self, state):
        """Calculates the expectation value <psi|H|psi>."""
        H = self.to_matrix()
        # <psi|H|psi> = state^dagger @ H @ state
        return jnp.real(jnp.vdot(state, jnp.dot(H, state)))
