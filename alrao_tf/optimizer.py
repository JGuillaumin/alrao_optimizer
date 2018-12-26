import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.training import training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.ops import resource_variable_ops


def generate_random_lr(min_lr, max_lr, shape):
    lr = np.random.uniform(low=0., high=1., size=shape)
    lr = np.exp(lr*(np.log(max_lr) - np.log(min_lr)) + np.log(min_lr))
    return lr


class AlraoGradientDescentOptimizer(optimizer.Optimizer):

    def __init__(self, min_lr=1e-5, max_lr=10., use_locking=False, name="AlraoGradientDescent"):
        super(AlraoGradientDescentOptimizer, self).__init__(use_locking, name)
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._dict_learning_rate = dict()

    def _create_slots(self, var_list):
        for v in var_list:
            var_name = v.name.split(":")[0]
            lr = generate_random_lr(min_lr=self._min_lr,
                                    max_lr=self._max_lr,
                                    shape=v.shape.as_list())

            self._dict_learning_rate[v.name] = ops.convert_to_tensor(lr, name=var_name+"-lr", dtype=v.dtype.base_dtype)

    def _prepare(self):
        # nothing to do !
        pass

    def _apply_dense(self, grad, var):
        # learning rate is already cast to var.dtype
        # element-wise multiplication between grads and learning rates.
        scaled_grad = math_ops.multiply(self._dict_learning_rate[var.name], grad)

        # training_ops.apply_gradient_descent() requires a scalar learning rate. Here set to 1. since grads are
        # already scaled with random learning rate.
        return training_ops.apply_gradient_descent(var, 1., scaled_grad,
                                                   use_locking=self._use_locking,).op

    def _resource_apply_dense(self, grad, handle):
        scaled_grad = math_ops.multiply(self._dict_learning_rate[handle.name], grad)
        return training_ops.resource_apply_gradient_descent(handle.handle,
                                                            1.,
                                                            grad,
                                                            use_locking=self._use_locking)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        return resource_variable_ops.resource_scatter_add(
            handle.handle, indices,
            -math_ops.multiply(grad, self._dict_learning_rate[handle.name]))

    def _apply_sparse_duplicate_indices(self, grad, var):
        delta = ops.IndexedSlices(math_ops.multiply(grad.values, self._dict_learning_rate[var.name]),
                                  grad.indices, grad.dense_shape)
        return var.scatter_sub(delta, use_locking=self._use_locking)


class AlraoMomentumOptimizer(optimizer.Optimizer):
    def __init__(self, min_lr=1e-5, max_lr=10.,
                 momentum=0.9, use_locking=False,
                 name="AlraoMomentum", use_nesterov=False):
        super(AlraoMomentumOptimizer, self).__init__(use_locking, name)
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._momentum = momentum
        self._use_nesterov = use_nesterov
        self._dict_learning_rate = dict()
        self._momentum_tensor = None

    def _create_slots(self, var_list):
        for v in var_list:
            var_name = v.name.split(":")[0]
            lr = generate_random_lr(min_lr=self._min_lr,
                                    max_lr=self._max_lr,
                                    shape=v.shape.as_list())

            self._dict_learning_rate[v.name] = ops.convert_to_tensor(lr, name=var_name+"-lr", dtype=v.dtype.base_dtype)

            self._zeros_slot(v, "momentum", self._name)

    def _prepare(self):
        momentum = self._momentum
        if callable(momentum):
            momentum = momentum()
        self._momentum_tensor = ops.convert_to_tensor(momentum, name="momentum")

    def _apply_dense(self, grad, var):
        scaled_grad = math_ops.multiply(self._dict_learning_rate[var.name], grad)
        mom = self.get_slot(var, "momentum")
        return training_ops.apply_momentum(var, mom,
                                           1.,
                                           scaled_grad,
                                           math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
                                           use_locking=self._use_locking,
                                           use_nesterov=self._use_nesterov).op

    def _resource_apply_dense(self, grad, handle):
        scaled_grad = math_ops.multiply(self._dict_learning_rate[handle.name], grad)
        mom = self.get_slot(handle, "momemtum")
        return training_ops.resource_apply_momentum(handle.handle, mom.handle,
                                                    1.,
                                                    scaled_grad,
                                                    math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
                                                    use_locking=self._use_locking,
                                                    use_nesterov=self._use_nesterov)

    def _apply_sparse(self, grad, var):
        scaled_grad = math_ops.multiply(self._dict_learning_rate[var.name], grad)
        mom = self.get_slot(var, "momemtum")
        return training_ops.sparse_apply_momentum(var, mom,
                                                  1.,
                                                  scaled_grad.value, scaled_grad.indices,
                                                  math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
                                                  use_locking=self._use_locking,
                                                  use_nesterov=self._use_nesterov)

    def _resource_apply_sparse(self, grad, handle, indices):
        scaled_grad = math_ops.multiply(self._dict_learning_rate[handle.name], grad)
        mom = self.get_slot(handle, "momemtum")
        return training_ops.resource_sparse_apply_momentum(handle.handle, mom.handle,
                                                           1.,
                                                           scaled_grad, indices,
                                                           math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
                                                           use_locking=self._use_locking,
                                                           use_nesterov=self._use_nesterov)
