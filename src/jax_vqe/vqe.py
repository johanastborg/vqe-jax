import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax

class VQE:
    """
    Main VQE driver class.
    """
    def __init__(self, ansatz, optimizer):
        """
        Args:
            ansatz: An instance of an Ansatz class (must have forward(params) method).
            optimizer: An optax optimizer (GradientTransformation).
        """
        self.ansatz = ansatz
        self.optimizer = optimizer

    def run(self, hamiltonian, initial_params, steps=100):
        """
        Runs the VQE optimization loop.

        Args:
            hamiltonian: The Hamiltonian to minimize.
            initial_params: Initial parameters for the ansatz.
            steps: Number of optimization steps.

        Returns:
            Dictionary containing optimal parameters and energy history.
        """

        # Pre-compute the Hamiltonian matrix to avoid re-computing it inside the JIT loop
        # But wait, hamiltonian.expectation() calls to_matrix().
        # If to_matrix() caches, it's fine.
        # But passing the 'hamiltonian' object into JIT might be tricky if it's not a PyTree.
        # It's better to extract the matrix first or define the cost function using the matrix directly.

        H_matrix = hamiltonian.to_matrix()

        @jit
        def cost_fn(params):
            state = self.ansatz.forward(params)
            # Expectation <psi|H|psi>
            return jnp.real(jnp.vdot(state, jnp.dot(H_matrix, state)))

        opt_state = self.optimizer.init(initial_params)
        params = initial_params

        @jit
        def update_step(params, opt_state):
            loss, grads = value_and_grad(cost_fn)(params)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        energy_history = []
        for i in range(steps):
            params, opt_state, loss = update_step(params, opt_state)
            energy_history.append(float(loss))
            if i % 10 == 0:
                pass # logging could be added here

        return {
            "optimal_params": params,
            "optimal_value": energy_history[-1],
            "history": energy_history
        }
