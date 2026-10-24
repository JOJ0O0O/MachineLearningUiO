import numpy as np
import jax
import Activation_Cost_funcs as acf
import jax.numpy as jnp

class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        cost_fun,
        optimizer,
        output_error=None,
        seed=2026,
        regularization = None,
        lam = 0.0
    ):
        self.activation_funcs = activation_funcs
        self.activation_ders = [acf.dict_act_der[activation_fun] for activation_fun in activation_funcs]
        self.cost_fun = cost_fun
        self.cost_der = acf.dict_cost_der[cost_fun]
        self.output_error = output_error     # e.g. a - y for softmax + cross entropy; None otherwise
        rng = np.random.default_rng(seed)
        sizes = [network_input_size] + list(layer_output_sizes)
        self.layers = [(rng.standard_normal((n_in, n_out)) / np.sqrt(n_in), np.zeros(n_out))
                       for n_in, n_out in zip(sizes[:-1], sizes[1:])]
        self.optimizer = optimizer
        if regularization in ["L1", "L2", None]:
            self.regularization = regularization
            self.lam = lam
        else: raise ValueError(regularization, "is not a valid regularization the options are : L1, L2 and None")

    def predict(self, inputs):
        # Simple feed forward pass
        return self.jax_compliant_predict(self.layers, inputs)

    def cost(self, inputs, targets):
        return self.cost_fun(self.predict(inputs), targets)

    def _feed_forward_saver(self, inputs):
        layer_inputs, zs = [], []
        a = inputs
        for (W, b), f in zip(self.layers, self.activation_funcs):
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
            if i == L - 1 and self.output_error is not None:
                dC_dz = self.output_error(predict, targets)
            else:
                if i == L - 1:
                    dC_da = self.cost_der(predict, targets)
                else:
                    W_next, _ = self.layers[i + 1]
                    dC_da = dC_dz @ W_next.T
                dC_dz = dC_da * self.activation_ders[i](z)
            layer_grads[i] = (layer_input.T @ dC_dz, np.sum(dC_dz, axis=0))
        if self.regularization == "L1":
            layer_grads = [(dW + self.lam * np.sign(W), db +  self.lam * np.sign(b)) for (dW, db),(W,b) in zip(layer_grads, self.layers)]
        elif self.regularization == "L2":
            layer_grads = [(dW + 2 * self.lam * (W), db + 2 * self.lam * (b)) for (dW, db),(W,b) in zip(layer_grads, self.layers)]
        
        return layer_grads

    def train_jax(self, inputs, targets, epochs, sgd = True):
        """Train the model using the input optimizer and jax"""
        """Note: Consider using time decay and early stopping in sgd???"""
        if(sgd):
            rng = np.random.default_rng(2026)
            for _ in range(epochs):
                
                for batch in self.shuffeled_batches(len(inputs), 128, rng):
                    grad = self.jax_gradient(inputs[batch], targets[batch])
                    step = self.optimizer.step(grad)
                    self.update_weights(step)

        else:
            for _ in range(epochs):
                grad = self.jax_gradient(inputs, targets)
                step = self.optimizer.step(grad)
                self.update_weights(step)
        
        print("Training with ", epochs, " epochs is done.")

    def train(self, inputs, targets, epochs, sgd = True):
        """Train the model using the input optimizer"""
        if(sgd):
            print(len(inputs))
            rng = np.random.default_rng(2026)
            for _ in range(epochs):
                for batch in self.shuffeled_batches(len(inputs), 128, rng):
                    grad = self.compute_gradient(inputs[batch], targets[batch])
                    step = self.optimizer.step(grad)
                    self.update_weights(step)

        else:
            for _ in range(epochs):
                grad = self.compute_gradient(inputs, targets)
                step = self.optimizer.step(grad)
                self.update_weights(step)

        print("Training with ", epochs, " epochs is done.")

    def update_weights(self, layer_updates):
        """Apply the updates (one (dW, db) per layer) computed by an optimizer outside the class."""
        self.layers = [(W - dW, b - db) for (W, b), (dW, db) in zip(self.layers, layer_updates)]

    # These last two methods are not needed in the project, but they can be nice to have!
    # The first one has a layers parameter so that you can use jax.grad on it.
    def jax_compliant_predict(self, layers, inputs):
        a = inputs
        for (W, b), f in zip(layers, self.activation_funcs):
            a = f(a @ W + b)
        return a

    def jax_gradient(self, inputs, targets):
        if self.regularization == "L1":
            def cost_of_layers(layers):
                return self.cost_fun(self.jax_compliant_predict(layers, inputs), targets) + self.lam * sum(sum(sum(jnp.abs(W))) + sum(jnp.abs(b)) for (W, b) in layers)
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