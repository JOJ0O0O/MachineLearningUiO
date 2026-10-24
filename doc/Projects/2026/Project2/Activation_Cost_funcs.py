import numpy as np
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that results agree with numpy to ~1e-15
import jax.numpy as jnp                    # numpy-compatible functions that JAX can differentiate

# Defining activation functions and their derivative
def ReLU(z):
    return jnp.where(z > 0, z, 0.0)

# Derivative of the ReLU function
def ReLU_der(z):
    return jnp.where(z > 0, 1.0, 0.0)

def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

# Derivative of the sigmoid function
def sigmoid_der(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time."""
    e_z = jnp.exp(z - jnp.max(z))
    return e_z / jnp.sum(e_z)

def mse(predict, target):
    return jnp.mean((predict - target) ** 2)

def mse_der(predict, target):
    return 2 * (predict - target) / predict.size

dict_act_der = {ReLU : ReLU_der,
                sigmoid: sigmoid_der,
                softmax: None}

dict_cost_der = {mse : mse_der}