import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that results agree with numpy to ~1e-15
import jax.numpy as jnp                    # numpy-compatible functions that JAX can differentiate
"""
Defining activation functions and their derivative
Most of these functions are directly taken from the lecture book with np replaced by jnp where applicable
"""

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

def cross_entropy(predict, target):
    return jnp.sum(-target * jnp.log(predict))

def mse_der(predict, target):
    return 2 * (predict - target) / predict.size

def leaky_relu(z, a=0.01): return jnp.where(z > 0, z, a * z)
def leaky_relu_prime(z, a=0.01): return jnp.where(z > 0, 1.0, a)
def elu(z, a=1.0): # Eq. (8.elu)
    return jnp.where(z > 0, z, a * (jnp.exp(jnp.minimum(z, 0)) - 1))
def elu_prime(z, a=1.0):
    return jnp.where(z > 0, 1.0, a * jnp.exp(jnp.minimum(z, 0)))
def tanh_(z): return jnp.tanh(z)
def tanh_prime(z): return 1 - jnp.tanh(z)**2
def identity(z): return z
def identity_prime(z): return jnp.ones_like(z)
def softplus(z): return jnp.log1p(jnp.exp(-jnp.abs(z))) + jnp.maximum(z, 0)
def softplus_prime(z): return sigmoid(z)
def gelu(z):
    """Exact GELU, Eq. (8.gelu); erf comes from scipy."""
    from scipy.special import erf
    return z * 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
def gelu_prime(z):
    from scipy.special import erf
    Phi = 0.5 * (1.0 + erf(z / jnp.sqrt(2.0)))
    phi = jnp.exp(-0.5 * z**2) / jnp.sqrt(2.0 * np.pi) # the normal density
    return Phi + z * phi # d/dz [z Phi(z)]
def gelu_tanh(z):
    """The approximation of Eq. (8.geluapprox) used by most frameworks."""
    return 0.5 * z * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (z + 0.044715 * z**3)))
def swish(z): return z * sigmoid(z) # Eq. (8.swish)
def swish_prime(z):
    s = sigmoid(z); return s + z * s * (1 - s)
def mish(z): # Eq. (8.mish)
    sp = jnp.log1p(jnp.exp(-jnp.abs(z))) + jnp.maximum(z, 0)
    return z * jnp.tanh(sp)
def mish_prime(z, h=1e-6):
    return (mish(z + h) - mish(z - h)) / (2 * h) # numerical is adequate

def ReLU_weights(rng, n_in, n_out):
    return rng.normal(0, np.sqrt(2/ n_in), size = (n_in, n_out))

def Sigmoid_weights(rng, n_in, n_out):
    return rng.normal(0, np.sqrt(2/ (n_in + n_out)), size = (n_in, n_out))

def leaky_ReLU_weights(rng, n_in, n_out, a=0.01):
    return rng.normal(0, np.sqrt(2 / ((1 + a**2) * n_in)), size=(n_in, n_out))

dict_act_der = {ReLU : ReLU_der,
                sigmoid: sigmoid_der,
                leaky_relu: leaky_relu_prime,
                elu: elu_prime,
                tanh_: tanh_prime,
                identity : identity_prime,
                softplus : softplus_prime,
                gelu: gelu_prime,
                swish: swish_prime,
                mish : mish_prime,
                softmax : None,
                softmax_vec : None}

dict_act_weights = {ReLU : ReLU_weights,
                sigmoid: Sigmoid_weights,
                leaky_relu: leaky_ReLU_weights,
                elu: ReLU_weights,
                tanh_: Sigmoid_weights,
                identity : Sigmoid_weights,
                softplus : ReLU_weights,
                gelu_tanh: ReLU_weights,
                gelu: ReLU_weights,
                swish: ReLU_weights,
                mish : ReLU_weights,
                softmax: Sigmoid_weights,
                softmax_vec: Sigmoid_weights}


dict_cost_der = {mse : mse_der,
                 cross_entropy : None}