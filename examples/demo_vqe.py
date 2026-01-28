import jax
import jax.numpy as jnp
from jax_vqe.hamiltonian import Hamiltonian
from jax_vqe.circuit import RyAnsatz
from jax_vqe.vqe import VQE
from jax_vqe.optimizer import get_optimizer

def main():
    print("Running VQE Demo...")

    # Define a simple Hamiltonian for 2 qubits
    # H = -1.0 * Z0 * Z1 - 1.0 * X0
    # This represents a simple Ising model term + transverse field

    terms = [
        (-1.0, "ZZ"),
        (-1.0, "XI")
    ]
    hamiltonian = Hamiltonian(terms)

    print("Hamiltonian terms:", terms)

    # Define Ansatz
    n_qubits = 2
    depth = 2
    ansatz = RyAnsatz(n_qubits, depth)
    print(f"Ansatz: RyAnsatz with {n_qubits} qubits and depth {depth}")

    # Define Optimizer
    optimizer = get_optimizer("adam", learning_rate=0.05)

    # Initialize VQE
    vqe = VQE(ansatz, optimizer)

    # Initial Parameters
    key = jax.random.PRNGKey(42)
    initial_params = jax.random.normal(key, shape=(ansatz.n_params,))

    # Run
    print("Starting optimization...")
    result = vqe.run(hamiltonian, initial_params, steps=300)

    print("Optimization complete.")
    print(f"Optimal Energy: {result['optimal_value']:.6f}")
    print("Optimal Parameters:", result['optimal_params'])

    # Verify with Exact Diagonalization
    H_mat = hamiltonian.to_matrix()
    eigenvalues, _ = jnp.linalg.eigh(H_mat)
    min_eigenvalue = eigenvalues[0]
    print(f"Exact Ground State Energy: {min_eigenvalue:.6f}")

    diff = abs(result['optimal_value'] - min_eigenvalue)
    print(f"Difference: {diff:.6f}")

if __name__ == "__main__":
    main()
