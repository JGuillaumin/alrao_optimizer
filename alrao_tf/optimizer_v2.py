import numpy as np
import tensorflow as tf
from tensorflow import keras as kr

def generate_random_lr(min_lr:float, max_lr:float, shape:list, name:str, dtype:object):
    return tf.Variable(np.exp(
                np.random.uniform(low=0., high=1., size=shape)
                *(np.log(max_lr) - np.log(min_lr)) + np.log(min_lr)),
                name=name, dtype=dtype
    )

# It only has been tested for _resource_apply_dense, but since I (saisua) did not write
# the code (it was JGuillaumin), I expect everithing to work, as I only did update a few
# calls.

class AlraoMomentumOptimizer(kr.optimizers.Optimizer):
    def __init__(self, min_lr:float=1e-5, max_lr:float=10, 
                        momentum:float=0.9, use_nesterov:bool=False, use_locking:bool=False,
                        name="Alraom", **kwargs):
        super().__init__(name, **kwargs)

        self._min_lr = min_lr
        self._max_lr = max_lr
        self._momentum = momentum
        self._use_nesterov = use_nesterov
        self._dict_learning_rate = dict()
        self._momentum_tensor = None
        self._use_nesterov = use_nesterov
        self._use_locking = use_locking
        self._is_first = True


    def _prepare(self, var_list):
        if(self._is_first):
            momentum = self._momentum
            if callable(momentum):
                momentum = momentum()

            self._momentum_tensor = tf.Variable(momentum, name="momentum")
            self._is_first = False

            for v in var_list:
                self._dict_learning_rate[v.name] = generate_random_lr(min_lr=self._min_lr,
                                                                        max_lr=self._max_lr,
                                                                        shape=v.shape.as_list(),
                                                                        name=f"{v.name.split('/')[0]}-lr",
                                                                        dtype=v.dtype.base_dtype
                                                                        )

    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, "momentum")

    @tf.function
    def _apply_dense(self, var, grad):
        tf.raw_ops.ApplyMomentum(
                                var=var, 
                                accum=self.get_slot(var, "momentum"),
                                lr=1.,
                                grad=tf.multiply(self._dict_learning_rate[var.name], grad),
                                momentum=tf.cast(self._momentum_tensor, var.dtype.base_dtype),
                                use_locking=self._use_locking,
                                use_nesterov=self._use_nesterov)

    @tf.function
    def _resource_apply_dense(self, grad, handle):
        tf.raw_ops.ResourceApplyMomentum(var=handle.handle, 
                                        accum=self.get_slot(handle, "momentum").handle,
                                        lr=1.,
                                        grad=tf.multiply(self._dict_learning_rate[handle.name], grad),
                                        momentum=tf.cast(self._momentum_tensor, grad.dtype.base_dtype),
                                        use_locking=self._use_locking,
                                        use_nesterov=self._use_nesterov)

    @tf.function
    def _apply_sparse(self, grad, var):
        scaled_grad = tf.multiply(self._dict_learning_rate[var.name], grad)

        tf.raw_ops.SparseApplyMomentum(var=var, 
                                        accum=self.get_slot(var, "momentum"),
                                        lr=1.,
                                        grad=scaled_grad.value, 
                                        indices=scaled_grad.indices,
                                        momentum=tf.cast(self._momentum_tensor, grad.dtype.base_dtype),
                                        use_locking=self._use_locking,
                                        use_nesterov=self._use_nesterov)

    @tf.function
    def _resource_apply_sparse(self, grad, handle, indices):
        tf.raw_ops.ResourceSparseApplyMomentum(var=handle.handle, 
                                                accum=self.get_slot(handle, "momentum").handle,
                                                lr=1.,
                                                grad=tf.multiply(self._dict_learning_rate[handle.name], grad), 
                                                indices=indices,
                                                momentum=tf.cast(self._momentum_tensor, grad.dtype.base_dtype),
                                                use_locking=self._use_locking,
                                                use_nesterov=self._use_nesterov)


    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
        }

if(__name__ == "__main__"):
    m = kr.Sequential([
        kr.layers.InputLayer((1,)),
        kr.layers.Dense(1)
    ])

    m.compile(optimizer=AlraoMomentumOptimizer(),#learning_rate=initial_lr, clipnorm=0.5),
                        loss=kr.losses.mse)
    m.fit([1], [2], epochs=1)