import numpy as np
import jax
import Activation_Cost_funcs as acf
import jax.numpy as jnp
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone


class NeuralNetwork(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        cost_fun,
        optimizer,
        gamma=0.1,
        seed=2026,
        regularization = None,
        lam = 0.0
    ):
        self.activation_funcs = activation_funcs
        n_layers = len(layer_output_sizes)
        if n_layers == 0:
            raise ValueError("layer_output_sizes must contain at least one layer")
        if len(activation_funcs) < n_layers:
            raise ValueError("activation_funcs must provide an activation for every layer")
        self.activation_ders = [
            acf.dict_act_der[activation_fun]
            for activation_fun in self.activation_funcs
        ]
        self.cost_fun = cost_fun
        self.cost_der = acf.dict_cost_der[cost_fun]
        self.weight_ini = [
            acf.dict_act_weights[key] for key in self.activation_funcs
        ]
        self.seed = seed 
        self.optimizer = optimizer          #This will be an optimizer object, which is created from a different file
        self.layer_output_sizes = layer_output_sizes
        self.network_input_size = network_input_size
        self.gamma = gamma
        if regularization in ["L1", "L2", None]:
            self.regularization = regularization
            self.lam = lam
        else: raise ValueError(regularization, "is not a valid regularization the options are : L1, L2 and None")

    def predict(self, X):
        # Simple feed forward pass
        return np.asarray(self.jax_compliant_predict(self.layers, X)).reshape(-1)

    def cost(self, inputs, targets):
        return self.cost_fun(self.predict(inputs), targets)

    def _feed_forward_saver(self, inputs):
        layer_inputs, zs = [], []
        a = inputs
        for i, (W, b) in enumerate(self.layers):
            f = self.activation_funcs[self._activation_index(i, len(self.layers))]
            layer_inputs.append(a)
            z = a @ W + b
            a = f(z)
            zs.append(z)
        return layer_inputs, zs, a

    def compute_gradient(self, inputs, targets):
        layer_inputs, zs, predict = self._feed_forward_saver(inputs)
        layer_grads = [() for _ in self.layers]
        L = len(self.layers)
        for i in reversed(range(L)):
            layer_input, z = layer_inputs[i], zs[i]
            if i == L - 1:
                dC_da = self.cost_der(predict, targets)
            else:
                W_next, _ = self.layers[i + 1]
                dC_da = dC_dz @ W_next.T
            derivative_index = self._activation_index(i, L)
            dC_dz = dC_da * self.activation_ders[derivative_index](z)
            layer_grads[i] = (layer_input.T @ dC_dz, np.sum(dC_dz, axis=0))
        if self.regularization == "L1":
            layer_grads = [(dW + self.lam * np.sign(W), db +  self.lam * np.sign(b)) for (dW, db),(W,b) in zip(layer_grads, self.layers)]
        elif self.regularization == "L2":
            layer_grads = [(dW + 2 * self.lam * (W), db + 2 * self.lam * (b)) for (dW, db),(W,b) in zip(layer_grads, self.layers)]
        
        return layer_grads

    def train_jax(self, inputs, targets, epochs, sgd = True):
        """Train the model using the input optimizer and jax"""
        """Note: Consider using time decay and early stopping in sgd???"""
        self.init_layers()
        if(sgd):
            rng = np.random.default_rng(2026)
            for _ in range(epochs):
                for batch in self.shuffeled_batches(len(inputs), 128, rng):
                    grad = self.jax_gradient(inputs[batch], targets[batch])
                    step =  self._get_optimizer().step(grad)
                    self.update_weights(step)

        else:
            for _ in range(epochs):
                grad = self.jax_gradient(inputs, targets)
                step =  self._get_optimizer().step(grad)
                self.update_weights(step)
        
        print("Training with ", epochs, " epochs is done.")

    def fit(self, X, y):
        #Wrapper for scikit learn
        self.init_layers(reset=True)
        self.optimizer_ = self.optimizer(self.gamma)
        self.train(X, np.asarray(y).reshape(-1, 1), epochs=1000)
        return self

    def train(self, inputs, targets, epochs, sgd = True):
        """Train the model using the input optimizer"""
        self.init_layers()
        if(sgd):
            rng = np.random.default_rng(2026)
            for _ in range(epochs):
                for batch in self.shuffeled_batches(len(inputs), 16, rng):
                    grad = self.compute_gradient(inputs[batch], targets[batch])
                    step =  self._get_optimizer().step(grad)
                    self.update_weights(step)

        else:
            for _ in range(epochs):
                grad = self.compute_gradient(inputs, targets)
                step =  self._get_optimizer().step(grad)
                self.update_weights(step)
        
        print("Training with ", epochs, " epochs is done.")

    def update_weights(self, layer_updates):
        """Apply the updates (one (dW, db) per layer) computed by an optimizer outside the class."""
        self.layers = [(W - dW, b - db) for (W, b), (dW, db) in zip(self.layers, layer_updates)]

    # These last two methods are not needed in the project, but they can be nice to have!
    # The first one has a layers parameter so that you can use jax.grad on it.
    def jax_compliant_predict(self, layers, inputs):
        a = inputs
        for i, (W, b) in enumerate(layers):
            f = self.activation_funcs[self._activation_index(i, len(layers))]
            a = f(a @ W + b)
        return a

    def _activation_index(self, layer_index, n_layers):
        """
            This helper function is AI generated, to merge what scikit learns CV functionalities need with the previous architecture of my neural network.
            This sone is specifically used so we can use a CV over the number of layers, while giving the network only one set of activation functions, with the last / output activation being the last one in the list.
            For this I have used gpt-6 luna.
            I have implemented, tested and modified this function myself.
            This therefore corresponds to usage level 2.
        """
        if len(self.activation_funcs) < n_layers:
            raise ValueError(
                "activation_funcs must provide an activation for every layer "
                f"({n_layers} required, {len(self.activation_funcs)} provided)"
            )
        return layer_index if layer_index < n_layers - 1 else len(self.activation_funcs) - 1

    def jax_gradient(self, inputs, targets):
        if self.regularization == "L1":
            def cost_of_layers(layers):
                return self.cost_fun(self.jax_compliant_predict(layers, inputs), targets) + self.lam * sum(
                                                                                            jnp.sum(jnp.where(W == 0, 0.0, jnp.abs(W)))
                                                                                            + jnp.sum(jnp.where(b == 0, 0.0, jnp.abs(b))) for (W, b) in layers)
            return jax.grad(cost_of_layers)(self.layers)
        elif self.regularization == "L2":
            def cost_of_layers(layers):
                        return self.cost_fun(self.jax_compliant_predict(layers, inputs), targets) + self.lam * sum(sum(sum(W*W)) + sum(b*b) for (W, b) in layers)
            return jax.grad(cost_of_layers)(self.layers)
        else:
            def cost_of_layers(layers):
                        return self.cost_fun(self.jax_compliant_predict(layers, inputs), targets)
            return jax.grad(cost_of_layers)(self.layers)

    @staticmethod
    def shuffeled_batches(n, M, rng):
        idx = rng.permutation(n)
        for k in range(0, n, M):
            yield idx[k:k + M]

    def _get_optimizer(self):
        """
        This helper function is AI generated, to merge what scikit learns CV functionalities neeed with the previous architecture of my neural network
        For this I have used gpt-6 luna.
        I have implemented, tested and modified this function myself.
        This therefore corresponds to usage level 2.
        """
        if not hasattr(self, "optimizer_"):
            self.optimizer_ = self.optimizer(self.gamma)
        return self.optimizer_

    def init_layers(self, reset=False):
        """
        This helper function is AI generated, to merge what scikit learns CV functionalities neeed with the previous architecture of my neural network
        For this I have used gpt-6 luna.
        I have implemented, tested and modified this function myself.
        This therefore corresponds to usage level 2.
        """
        """
        Initilizes all weights and biases newly
        If the weights and biases are reset, the optimizer will also be reset
        Needed if the Network is used inside a Crossvalidation from scikit-learn
        """
        sizes = [self.network_input_size] + list(self.layer_output_sizes)
        n_layers = len(sizes) - 1
        if len(self.activation_funcs) < n_layers:
            raise ValueError(
                "activation_funcs must provide an activation for every layer "
                f"({n_layers} required, {len(self.activation_funcs)} provided)"
            )
        expected_shapes = [((n_in, n_out), (n_out,))
                           for n_in, n_out in zip(sizes[:-1], sizes[1:])]
        current_shapes = [(W.shape, b.shape) for W, b in getattr(self, "layers", [])]
        if reset or current_shapes != expected_shapes:
            rng = np.random.default_rng(self.seed)
            self.layers = []

            for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
                initializer_index = self._activation_index(i, n_layers)
                initializer = self.weight_ini[initializer_index]
                weights = initializer(rng, fan_in, fan_out)
                biases = np.zeros(fan_out)
                self.layers.append((weights, biases))
            if hasattr(self, "optimizer_"):
                del self.optimizer_
