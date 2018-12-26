import tensorflow as tf
from pprint import  pprint
import numpy as np

from .optimizer import AlraoMomentumOptimizer, AlraoGradientDescentOptimizer


class AlraoModel:
    def __init__(self,
                 pre_classifier_scope=None,
                 nb_classifiers=10,
                 num_classes=10,
                 weight_decay=0.,
                 min_lr=1e-5,
                 max_lr=10.,
                 momentum=0.9,
                 verbose=True):
        self.pre_classifier_scope = pre_classifier_scope
        self.nb_classifiers = nb_classifiers
        self.num_classes = num_classes
        self.list_classifiers = []
        self.verbose = verbose
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.momentum = momentum

        self.averaged_probs = None
        self.list_probs = []
        self.list_logits = []
        self.list_losses = []
        self.averaged_loss = None

        self.list_metric_cxent = []
        self.list_update_cxent_op = []
        self.list_metric_l2 = []
        self.list_update_l2_op = []
        self.list_metric_acc = []
        self.list_update_acc_op = []

        self.metric_averaged_acc, self.update_averaged_acc_op = None, None
        self.metric_averaged_cxent, self.update_averaged_cxent_op = None, None
        self.metric_l2_pre_cls, self.update_l2_pre_cls_op = None, None

    def apply(self, features):

        for i in range(self.num_classes):
            name_classifier = "SubClassifier-{}".format(i)
            with tf.variable_scope(name_classifier) as scope_classifier:
                logits = tf.layers.dense(inputs=features, units=self.num_classes)
                probs = tf.nn.softmax(logits, axis=1, name="probs_cls_{}".format(i))

                self.list_logits.append(logits)
                self.list_probs.append(probs)
                self.list_classifiers.append(scope_classifier)

        # deterministic model averaging !
        self.averaged_probs = tf.reduce_mean(tf.stack(self.list_probs, axis=-1), axis=-1, name="averaged_probs")

        if self.verbose:
            print("list_probs:")
            pprint(self.list_probs)
            print("averaged_probs : {}".format(self.averaged_probs))

    def compute_losses(self, labels):

        for i in range(self.nb_classifiers):
            with tf.name_scope("Loss-cls-{}".format(i)):
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                                     logits=self.list_logits[i]),
                                      name="cxent")
                metric_cxent, update_cxent_op = tf.metrics.mean(loss, name="metric-cxent-cls-{}".format(i))
                self.list_metric_cxent.append(metric_cxent)
                self.list_update_cxent_op.append(update_cxent_op)
                self.list_losses.append(loss)

                if self.weight_decay > 0.:
                    l2_loss = tf.add_n([tf.nn.l2_loss(var) for
                                        var in tf.trainable_variables(scope=self.list_classifiers[i].name)],
                                       name='l2-loss-{}'.format(i))

                    metric_l2, update_l2_op = tf.metrics.mean(l2_loss, name="metric-l2-cls-{}".format(i))
                    self.list_metric_l2.append(metric_l2)
                    self.list_update_l2_op.append(update_l2_op)

                # compute accuracy for this classifier
                preds = tf.cast(tf.argmax(self.list_logits[i], axis=1), dtype=tf.int32)
                metric_acc,  update_acc_op = tf.metrics.accuracy(labels, preds,
                                                                 name='metric-acc-cls-{}'.format(i))

                self.list_metric_acc.append(metric_acc)
                self.list_update_acc_op.append(update_acc_op)

        self.averaged_loss = tf.reduce_mean(self.list_losses, name="averaged-cxent")
        self.metric_averaged_cxent, self.update_averaged_cxent_op = tf.metrics.mean(self.averaged_loss,
                                                                                    name='metric-averaged-cxent')

        if self.weight_decay > 0:
            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables(scope=self.pre_classifier_scope.name)],
                               name="l2-loss-pre-classifier")
            self.metric_l2_pre_cls, self.update_l2_pre_cls_op = tf.metrics.mean(l2_loss, name="metric-l2-pre-cls")

        averaged_preds = tf.cast(tf.argmax(self.averaged_probs, 1), dtype=tf.int32, name='averaged-preds')
        self.metric_averaged_acc, self.update_averaged_acc_op = tf.metrics.accuracy(labels, averaged_preds,
                                                                                    name="metric-averaged-acc")

    def get_train_op(self):

        classifiers_lr = [np.exp(np.log(self.min_lr) + \
                                 k / (self.nb_classifiers - 1) * (np.log(self.max_lr) - np.log(self.min_lr)) \
                                 ) for k in range(self.nb_classifiers)]

        list_train_op = list()

        for i in range(self.nb_classifiers):
            with tf.variable_scope('Optimizer-cls-{}'.format(i)):
                lr = classifiers_lr[i]
                if self.momentum > 0.:
                    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=self.momentum,
                                                     name="MomentumOptimizer-Cls-{}".format(i))
                else:
                    opt = tf.train.GradientDescentOptimizer(learning_rate=lr, name='SGDOptimizer-Cls-{}'.format(i))

                train_op = opt.minimize(self.list_losses[i],
                                        var_list=tf.trainable_variables(scope=self.list_classifiers[i].name))
                list_train_op.append(train_op)

        with tf.variable_scope("Optimizer-Pre-Classifier"):
            if self.momentum > 0:
                opt = AlraoMomentumOptimizer(min_lr=self.min_lr, max_lr=self.max_lr, momentum=self.momentum)
            else:
                opt = AlraoGradientDescentOptimizer(min_lr=self.min_lr, max_lr=self.max_lr)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op_pre_classifier = opt.minimize(self.averaged_loss,
                                                       var_list=tf.trainable_variables(
                                                           scope=self.pre_classifier_scope.name))
                with tf.control_dependencies([train_op_pre_classifier, ]):
                    main_op = tf.group(*list_train_op)

        return main_op
