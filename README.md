# JAX-VQE

A Variational Quantum Eigensolver (VQE) backend implementation using JAX and Python. This library is designed for molecular ground state calculations and other quantum chemistry applications.

## Features

- **JAX-based**: Utilizes JAX for automatic differentiation and JIT compilation.
- **Modular**: Separate modules for circuits/ansatz, Hamiltonians, and optimization.
- **Extensible**: Easy to add new ansatzes or optimizers.

## Installation

```bash
pip install -r requirements.txt
```

## Structure

- `src/jax_vqe`: Core library code.
- `tests`: Unit tests.
- `examples`: Usage examples.

## Usage

See `examples/demo_vqe.py` for a complete example.
