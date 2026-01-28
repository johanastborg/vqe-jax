import unittest
import jax
import jax.numpy as jnp
from jax_vqe.hamiltonian import Hamiltonian
from jax_vqe.circuit import RyAnsatz
from jax_vqe.vqe import VQE
from jax_vqe.optimizer import get_optimizer
from jax_vqe.utils import PAULI_MATS

class TestVQE(unittest.TestCase):
    def test_hamiltonian_matrix(self):
        # H = 1.0 * Z
        # Z = [[1, 0], [0, -1]]
        h = Hamiltonian([(1.0, "Z")])
        mat = h.to_matrix()
        expected = jnp.array([[1, 0], [0, -1]], dtype=complex)
        self.assertTrue(jnp.allclose(mat, expected))

    def test_simple_vqe_optimization(self):
        # H = Z. Minimal energy is -1 (state |1>)
        h = Hamiltonian([(1.0, "Z")])

        # Ansatz: 1 qubit, depth 1
        ansatz = RyAnsatz(n_qubits=1, depth=1)

        # Optimizer
        optimizer = get_optimizer("adam", learning_rate=0.1)

        vqe = VQE(ansatz, optimizer)

        # Initialize params
        key = jax.random.PRNGKey(0)
        initial_params = jax.random.normal(key, shape=(ansatz.n_params,))

        result = vqe.run(h, initial_params, steps=200)

        final_energy = result['optimal_value']
        # The ground state of Z is |1>, energy -1.
        self.assertAlmostEqual(final_energy, -1.0, places=3)

    def test_two_qubit_vqe(self):
        # H = Z0 * Z1. Ground states |01> (-1) or |10> (-1)?
        # Z x Z:
        # |00> -> 1*1 = 1
        # |01> -> 1*-1 = -1
        # |10> -> -1*1 = -1
        # |11> -> -1*-1 = 1
        # Ground energy is -1.

        h = Hamiltonian([(1.0, "ZZ")])
        ansatz = RyAnsatz(n_qubits=2, depth=2)
        optimizer = get_optimizer("adam", learning_rate=0.1)
        vqe = VQE(ansatz, optimizer)

        key = jax.random.PRNGKey(1)
        initial_params = jax.random.normal(key, shape=(ansatz.n_params,))

        result = vqe.run(h, initial_params, steps=200)

        self.assertLess(result['optimal_value'], -0.95)

if __name__ == '__main__':
    unittest.main()
