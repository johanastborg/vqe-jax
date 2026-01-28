import jax.numpy as jnp

# Pauli Matrices
I = jnp.array([[1, 0], [0, 1]], dtype=complex)
X = jnp.array([[0, 1], [1, 0]], dtype=complex)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=complex)
Z = jnp.array([[1, 0], [0, -1]], dtype=complex)

PAULI_MATS = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

def tensor_product(mats):
    """Computes the tensor product of a list of matrices."""
    if not mats:
        return jnp.array([[1]], dtype=complex)
    res = mats[0]
    for m in mats[1:]:
        res = jnp.kron(res, m)
    return res

def pauli_string_to_matrix(pauli_string):
    """
    Converts a Pauli string (e.g., "IXZ") to a matrix.
    """
    mats = [PAULI_MATS[p] for p in pauli_string]
    return tensor_product(mats)
