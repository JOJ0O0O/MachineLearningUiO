import numpy as np
import jax
import Activation_Cost_funcs as acf

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone

from tqdm import tqdm


class NeuralNetwork(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        cost_fun,
        optimizer,
        epochs=1000,
        gamma=0.001,
        lmbd=0.0,
        output_error=None,
        seed=2026,
    ):
        # AI usage: Claude (Anthropic) restructured this method to follow the
        # scikit-learn estimator contract: __init__ only stores its arguments
        # unchanged (required by clone(), cross_val_score and GridSearchCV).
        # Weight, derivative and optimizer construction moved to _initialize().
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.cost_fun = cost_fun
        self.optimizer = optimizer          # the optimizer class, e.g. AdamOptimizer
        self.epochs = epochs
        self.gamma = gamma
        self.lmbd = lmbd
        self.output_error = output_error    # e.g. a - y for softmax + cross entropy; None otherwise
        self.seed = seed
 
    def _initialize(self):
        # AI usage: Claude (Anthropic) wrote this method by moving the original
        # initialisation code out of __init__ and building the optimizer
        # instance from the optimizer class.
        self.activation_ders = [acf.dict_act_der[f] for f in self.activation_funcs]
        self.cost_der = acf.dict_cost_der[self.cost_fun]
        rng = np.random.default_rng(self.seed)
        sizes = [self.network_input_size] + list(self.layer_output_sizes)
        self.layers = [(rng.standard_normal((n_in, n_out)) / np.sqrt(n_in), np.zeros(n_out))
                       for n_in, n_out in zip(sizes[:-1], sizes[1:])]
        self.optimizer_ = self.optimizer(gamma=self.gamma)
 
    def get_weights(self):
        
        return self.layers
 
    def set_weights(self, new_weights):
        
        self.layers = new_weights
 
    def predict(self, X):
        
        # Simple feed forward pass
        input = X.copy()
        return self.jax_compliant_predict(self.layers, input)
 
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
        # AI usage: Claude (Anthropic) suggested adding the L2 penalty term
        # (lmbd * W) to the weight gradient. Backpropagation itself: no AI usage.
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
            layer_grads[i] = (layer_input.T @ dC_dz + self.lmbd * self.layers[i][0],
                              np.sum(dC_dz, axis=0))
        return layer_grads
 
    def train_jax(self, inputs, targets):
        """Train the model using the input optimizer and jax"""
        # AI usage: Claude (Anthropic) pointed out two bugs, fixed here: the
        # gradient was computed only once outside the loop, and jax_gradient
        # was called with `self` passed twice. Also uses self.optimizer_.
        for _ in tqdm(range(self.epochs)):
            grad = self.jax_gradient(inputs, targets)
            step = self.optimizer_.step(grad)
            self.update_weights(step)
        print("Training with ", self.epochs, " epochs is done.")
 
    def train(self, inputs, targets):
        """Train the model using the input optimizer"""
        # AI usage: Claude (Anthropic) changed self.optimizer to self.optimizer_
        # (the instance built in _initialize). Training loop otherwise: no AI usage.
        for _ in tqdm(range(self.epochs)):
            grad = self.compute_gradient(inputs, targets)
            step = self.optimizer_.step(grad)
            self.update_weights(step)
        print("Training with ", self.epochs, " epochs is done.")
 
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
        
        def cost_of_layers(layers):
            return self.cost_fun(self.jax_compliant_predict(layers, inputs), targets)
        return jax.grad(cost_of_layers)(self.layers)
 
    def fit(self, inputs, targets):
        # AI usage: Claude (Anthropic) diagnosed the NaN CV scores / NotFittedError
        # and suggested calling _initialize(), setting the fitted flag
        # `is_fitted_` (needed by sklearn's check_is_fitted inside Pipeline)
        # and returning self.
        self._initialize()
        self.train(inputs, targets)
        self.is_fitted_ = True
        return self

 