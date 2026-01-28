import jax.numpy as jnp
from jax import jit
from .utils import tensor_product, I

def ry_matrix(theta):
    """Returns the matrix for a rotation around the Y axis."""
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.array([[c, -s], [s, c]], dtype=complex)

def rz_matrix(theta):
    """Returns the matrix for a rotation around the Z axis."""
    phase = jnp.exp(1j * theta / 2)
    return jnp.array([[jnp.conj(phase), 0], [0, phase]], dtype=complex)

def cnot_matrix():
    """Returns the matrix for a CNOT gate."""
    return jnp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

def apply_layer_ry(state, n_qubits, params):
    """Applies a layer of Ry gates to all qubits."""
    # This can be optimized by kronecker product of all Rys
    ops = [ry_matrix(p) for p in params]
    full_op = tensor_product(ops)
    return jnp.dot(full_op, state)

def apply_entangling_layer(state, n_qubits):
    """Applies CNOTs in a ladder structure (0-1, 1-2, ...)."""
    # Naive full matrix construction
    current_state = state
    for i in range(n_qubits - 1):
        # Alternative: Use a helper to expand 2-qubit gate to full space
        full_op = _expand_gate(cnot_matrix(), [i, i+1], n_qubits)
        current_state = jnp.dot(full_op, current_state)
    return current_state

def _expand_gate(gate_matrix, targets, n_qubits):
    """
    Expands a gate to the full n-qubit Hilbert space.
    Only supports 1-qubit and adjacent 2-qubit gates for simplicity in this demo.
    """
    if len(targets) == 1:
        ops = [I] * n_qubits
        ops[targets[0]] = gate_matrix
        return tensor_product(ops)
    elif len(targets) == 2:
        # Assuming targets are [i, i+1]
        t1, t2 = targets
        if t2 != t1 + 1:
             raise NotImplementedError("Only adjacent 2-qubit gates supported in this simple demo.")

        # Correct approach for adjacent:
        # Kron product of Is before, Gate, Is after
        mats = []
        for k in range(t1):
            mats.append(I)
        mats.append(gate_matrix)
        for k in range(t2 + 1, n_qubits):
            mats.append(I)

        return tensor_product(mats)
    else:
        raise ValueError("Only 1 or 2 qubit gates supported.")


class RyAnsatz:
    """
    Hardware efficient ansatz with Ry rotations and entangling layers.
    """
    def __init__(self, n_qubits, depth):
        self.n_qubits = n_qubits
        self.depth = depth
        self.n_params = n_qubits * (depth + 1)

    def get_initial_state(self):
        state = jnp.zeros(2**self.n_qubits, dtype=complex)
        state = state.at[0].set(1.0)
        return state

    def forward(self, params):
        """
        Executes the ansatz circuit.
        Args:
            params: Flat array of parameters.
        Returns:
            Final state vector.
        """
        state = self.get_initial_state()

        # Reshape params for easier indexing: (depth + 1, n_qubits)
        params_reshaped = params.reshape((self.depth + 1, self.n_qubits))

        # Initial rotation layer
        state = apply_layer_ry(state, self.n_qubits, params_reshaped[0])

        for d in range(self.depth):
            # Entangling layer
            state = apply_entangling_layer(state, self.n_qubits)

            # Rotation layer
            state = apply_layer_ry(state, self.n_qubits, params_reshaped[d+1])

        return state
