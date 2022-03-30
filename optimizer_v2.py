import tensorflow as tf
from tensorflow import keras as kr
import re


# It only has been tested for _resource_apply_dense in momentum, but since I (saisua) did
# not write the code (it was JGuillaumin), I expect everithing to work, as I only did update
# a few calls.
# The default version of the code did not have learning rate implemented, but I have found
# it useful to reduce on plateau. It should initialize as 1. so if not needed, keep in mind
# it is not meant to be used

def generate_random_lr(min_lr:float, max_lr:float, shape:list, name:str, dtype:object):
    return tf.Variable(tf.exp(
                    tf.random.uniform(shape, maxval=1., minval=0.)*
                    (tf.math.log(max_lr) - tf.math.log(min_lr)) + tf.math.log(min_lr)
                ), name=name, dtype=dtype
    )
    
dense_num = re.compile("dense_(\d+)/.*?")
class AlraoGradientDescentOptimizer(kr.optimizers.Optimizer):
    def __init__(self, min_lr:float=1e-5, max_lr:float=10, learning_rate:float=1.0,
                        use_locking:bool=False, name="AlraoGD", **kwargs):
        super().__init__(name, **kwargs)

        self._set_hyper("learning_rate", learning_rate)

        self._min_lr = float(min_lr)
        self._max_lr = float(max_lr)
        self._dict_learning_rate = dict()
        self._use_locking = use_locking
        self._is_first = True

    def _prepare(self, var_list):
        if(self._is_first):
            self._is_first = False

            highest = 0
            for v in var_list:
                self._dict_learning_rate[v.name] = generate_random_lr(min_lr=self._min_lr,
                                                                        max_lr=self._max_lr,
                                                                        shape=v.shape.as_list(),
                                                                        name=f"{v.name.split('/')[0]}-lr",
                                                                        dtype=v.dtype.base_dtype
                                                                        )

    def _create_slots(self, var_list):
        if(self._is_first):
            for v in var_list:
                self._dict_learning_rate[v.name] = generate_random_lr(min_lr=self._min_lr,
                                                                        max_lr=self._max_lr,
                                                                        shape=v.shape.as_list(),
                                                                        name=f"{v.name.split('/')[0]}-lr",
                                                                        dtype=v.dtype.base_dtype
                                                                        )


    def _apply_dense(self, grad, var):
        # learning rate is already cast to var.dtype
        # element-wise multiplication between grads and learning rates.
        # training_ops.apply_gradient_descent() requires a scalar learning rate. Here set to 1. since grads are
        # already scaled with random learning rate.
        return tf.raw_ops.ApplyGradientDescent(var=var, 
                                                alpha=self._get_hyper("learning_rate"), 
                                                delta=tf.multiply(self._dict_learning_rate[var.name], grad),
                                                use_locking=self._use_locking,)

    def _resource_apply_dense(self, grad, handle):
        return tf.raw_ops.ResourceApplyGradientDescent(var=handle.handle,
                                                        alpha=self._get_hyper("learning_rate"),
                                                        delta=tf.multiply(self._dict_learning_rate[handle.name], grad),
                                                        use_locking=self._use_locking)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        return tf.raw_ops.ResourceScatterAdd(resource=handle.handle, 
                                        indices=indices,
                                        updates=-tf.multiply(grad, self._dict_learning_rate[handle.name]))


    def _apply_sparse_duplicate_indices(self, grad, var):
        return var.scatter_sub(tf.IndexedSlices(tf.multiply(grad.values, self._dict_learning_rate[var.name]),
                                                grad.indices, 
                                                grad.dense_shape), 
                                use_locking=self._use_locking)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
        }



class AlraoMomentumOptimizer(kr.optimizers.Optimizer):
    def __init__(self, min_lr:float=1e-5, max_lr:float=10, learning_rate:float=1.0,
                        momentum:float=0.9, use_nesterov:bool=True, use_locking:bool=False,
                        name="Alraom", **kwargs):
        super().__init__(name, **kwargs)

        self._set_hyper("learning_rate", learning_rate)

        self._min_lr = float(min_lr)
        self._max_lr = float(max_lr)
        self._momentum = float(momentum)
        self._use_nesterov = use_nesterov
        self._dict_learning_rate = dict()
        self._momentum_tensor = None
        self._use_locking = use_locking
        self._is_first = True


    def _prepare(self, var_list):
        if(self._is_first):
            self._is_first = False

            self._momentum_tensor = tf.Variable(self._momentum() if callable(self._momentum) else self._momentum, name="momentum")

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
        tf.raw_ops.ApplyMomentum(var=var, 
                                accum=self.get_slot(var, "momentum"),
                                lr=self._get_hyper("learning_rate"),
                                grad=tf.multiply(self._dict_learning_rate[var.name], grad),
                                momentum=tf.cast(self._momentum_tensor, var.dtype.base_dtype),
                                use_locking=self._use_locking,
                                use_nesterov=self._use_nesterov)

    @tf.function
    def _resource_apply_dense(self, grad, handle):
        tf.raw_ops.ResourceApplyMomentum(var=handle.handle, 
                                        accum=self.get_slot(handle, "momentum").handle,
                                        lr=self._get_hyper("learning_rate"),
                                        grad=tf.multiply(self._dict_learning_rate[handle.name], grad),
                                        momentum=tf.cast(self._momentum_tensor, grad.dtype.base_dtype),
                                        use_locking=self._use_locking,
                                        use_nesterov=self._use_nesterov)

    @tf.function
    def _apply_sparse(self, grad, var):
        scaled_grad = tf.multiply(self._dict_learning_rate[var.name], grad)

        tf.raw_ops.SparseApplyMomentum(var=var, 
                                        accum=self.get_slot(var, "momentum"),
                                        lr=self._get_hyper("learning_rate"),
                                        grad=scaled_grad.value, 
                                        indices=scaled_grad.indices,
                                        momentum=tf.cast(self._momentum_tensor, grad.dtype.base_dtype),
                                        use_locking=self._use_locking,
                                        use_nesterov=self._use_nesterov)

    @tf.function
    def _resource_apply_sparse(self, grad, handle, indices):
        tf.raw_ops.ResourceSparseApplyMomentum(var=handle.handle, 
                                                accum=self.get_slot(handle, "momentum").handle,
                                                lr=self._get_hyper("learning_rate"),
                                                grad=tf.multiply(self._dict_learning_rate[handle.name], grad), 
                                                indices=indices,
                                                momentum=tf.cast(self._momentum_tensor, grad.dtype.base_dtype),
                                                use_locking=self._use_locking,
                                                use_nesterov=self._use_nesterov)


    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
        }

if(__name__ == "__main__"):
    m = kr.Sequential([
        kr.layers.InputLayer((1,)),
        kr.layers.Dense(64),
        kr.layers.PReLU(),
        kr.layers.Dense(64),
        kr.layers.PReLU(),
        kr.layers.Dense(64),
        kr.layers.PReLU(),
        kr.layers.Dense(64),
        kr.layers.PReLU(),

        kr.layers.Dense(1),
        kr.layers.PReLU(),
    ])

    m.compile(optimizer=AlraoGradientDescentOptimizer(max_lr=0.01, min_lr=1e-5, learning_rate=1.),
                        loss=kr.losses.mse)
    m.fit([1,4,5,6,7,8,9,10], [2,8,10,12,14,16,18,20], epochs=100, verbose=2)