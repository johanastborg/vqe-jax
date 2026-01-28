import optax

def get_optimizer(name="adam", learning_rate=0.01):
    """
    Returns an optax optimizer.

    Args:
        name: Name of the optimizer ('adam', 'sgd', etc.)
        learning_rate: Learning rate for the optimizer.

    Returns:
        optax.GradientTransformation
    """
    if name.lower() == "adam":
        return optax.adam(learning_rate)
    elif name.lower() == "sgd":
        return optax.sgd(learning_rate)
    elif name.lower() == "adagrad":
        return optax.adagrad(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
