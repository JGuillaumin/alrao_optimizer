import tensorflow as tf
import os
from argparse import ArgumentParser
import numpy as np
import sys

from mobile_net_v2 import MobileNetv2
from utils import get_generators, COLORS, get_best_model_ckpt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args):

    print("TF version : ", tf.__version__)
    with_cuda = tf.test.is_built_with_cuda()
    with_gpu = tf.test.is_gpu_available()
    if with_cuda and with_gpu:
        data_format = "channels_first"
    else:
        data_format = "channels_last"
    print("Built with CUDA : ", with_cuda)
    print("Available GPU : ", with_gpu)
    print("data_format : ", data_format)

    if tf.gfile.Exists(args.output_dir):
        tf.gfile.DeleteRecursively(args.output_dir)
    tf.gfile.MakeDirs(args.output_dir)

    # create TensorFlow session
    config = tf.ConfigProto(allow_soft_placement=True)
    if with_cuda and with_gpu:
        config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    iterator_train, iterator_valid, iterator_test, \
        steps_per_epoch_train, steps_per_epoch_val, steps_per_epoch_test = get_generators(batch_size=args.batch_size,
                                                                                          data_format=data_format,
                                                                                          verbose=args.verbose)
    shape = [None, 32, 32, 3] if data_format == "channels_last" else [None, 3, 32, 32]
    with tf.name_scope("Inputs"):
        batch_x = tf.placeholder(shape=shape, dtype=tf.float32, name="batch_x")
        batch_y = tf.placeholder(shape=[None, ], dtype=tf.int32, name="batch_x")
        is_training_bn = tf.placeholder(shape=[], dtype=tf.bool)
        global_step = tf.train.get_or_create_global_step()

    with tf.variable_scope("MobileNetv2"):
        model = MobileNetv2(num_classes=10,
                            gamma=1.,
                            data_format="channels_last",
                            is_training_bn=is_training_bn,
                            expansion=[1, 6, 6, 6, 6, 6, 6],
                            fan_out=[16, 24, 32, 64, 96, 160, 320],
                            num_blocks=[1, 2, 3, 4, 3, 3, 1],
                            stride=[1, 1, 2, 2, 1, 2, 1])
        logits = model.complete_inference(inputs=batch_x)
        print("logits : {}".format(logits))

    with tf.name_scope("Loss"):
        cxent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batch_y),
                               name="cxent")
        total_loss = cxent

        if args.weight_decay > 0.:
            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='L2_loss')
            total_loss += args.weight_decay*l2_loss
        else:
            l2_loss = None

    with tf.name_scope("Optimizer"):
        opt = tf.train.MomentumOptimizer(learning_rate=args.lr, momentum=args.momentum, name='MomentumOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss=total_loss, global_step=global_step)

    with tf.name_scope("Summaries"):
        update_metric_ops = []

        predictions = tf.cast(tf.argmax(logits, 1), dtype=tf.int32)
        metric_acc, update_op_acc = tf.metrics.accuracy(batch_y, predictions, name="metric_acc")
        metric_cxent, update_op_cxent = tf.metrics.mean(cxent, name="metric_cxent")
        update_metric_ops.append(update_op_acc)
        update_metric_ops.append(update_op_cxent)

        if l2_loss is not None:
            metric_l2, update_op_l2 = tf.metrics.mean(l2_loss, name="metric_l2_loss")
            update_metric_ops.append(update_op_l2)
        else:
            metric_l2, update_op_l2 = None, None

        local_update_ops = tf.group(*update_metric_ops, name="local_update_ops")

    tf.summary.scalar('ACCURACY', metric_acc)
    tf.summary.scalar('CXENT', metric_cxent)
    merged_summaries_valid = tf.summary.merge_all()

    if l2_loss is not None:
        tf.summary.scalar('L2-LOSS', metric_l2)
    merged_summaries_train = tf.summary.merge_all()

    best_saver = tf.train.Saver(var_list=tf.global_variables(scope="MobileNetv2"), max_to_keep=1)
    best_model_ckpt_path = os.path.join(args.output_dir, 'best_model_ckpt')

    train_writer = tf.summary.FileWriter(args.output_dir + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter(args.output_dir + '/valid')
    test_writer = tf.summary.FileWriter(args.output_dir + '/test')

    best_acc = 0.
    list_acc_val = []
    list_acc_train = []

    # initialize model
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, args.epochs+1):

        # ======================= TRAIN ==================================
        # re-initialize local variables in streaming metrics
        sess.run(tf.local_variables_initializer())
        feed_dict = {is_training_bn: True}
        for _ in range(steps_per_epoch_train):
            x, y = iterator_train.next()
            feed_dict[batch_x] = x
            feed_dict[batch_y] = y
            _ = sess.run([train_op, local_update_ops], feed_dict=feed_dict)

        acc_v, cxent_v, summaries = sess.run([metric_acc, metric_cxent, merged_summaries_train], feed_dict=feed_dict)
        train_writer.add_summary(global_step=epoch, summary=summaries)
        train_writer.flush()
        list_acc_train.append(acc_v)
        # ==================================================================

        # ======================= VALIDATION ==================================
        # re-initialize local variables in streaming metrics
        sess.run(tf.local_variables_initializer())
        feed_dict = {is_training_bn: False}
        for _ in range(steps_per_epoch_val):
            x, y = iterator_valid.next()
            feed_dict[batch_x] = x
            feed_dict[batch_y] = y
            _ = sess.run([local_update_ops], feed_dict=feed_dict)

        feed_dict = {is_training_bn: False}
        acc_v, cxent_v, summaries = sess.run([metric_acc, metric_cxent, merged_summaries_valid], feed_dict=feed_dict)
        valid_writer.add_summary(global_step=epoch, summary=summaries)
        valid_writer.flush()
        list_acc_val.append(acc_v)
        # ==================================================================

        if acc_v > best_acc:
            best_acc = acc_v
            color = COLORS['green']
            best_saver.save(sess, best_model_ckpt_path, global_step=epoch)
        else:
            color = COLORS['red']
        print("EPOCH {} | TRAIN acc={:.4f} | {}VALID acc={:.5f}{}".format(epoch, list_acc_train[-1],
                                                                          color[0], list_acc_val[-1], color[1]))

    # ======================= TEST ==================================
    # load weights from best model and make inference on test set
    best_ckpt = get_best_model_ckpt(best_model_ckpt_path)
    best_epoch = np.argmax(list_acc_val)
    if best_ckpt is not None:
        print("Load best model {}, snapshot at epoch={}".format(best_ckpt, best_epoch))
        best_saver.restore(sess, best_ckpt)
    # re-initialize local variables in streaming metrics
    sess.run(tf.local_variables_initializer())
    feed_dict = {is_training_bn: False}
    for _ in range(steps_per_epoch_test):
        x, y = iterator_test.next()
        feed_dict[batch_x] = x
        feed_dict[batch_y] = y
        _ = sess.run([local_update_ops], feed_dict=feed_dict)

    feed_dict = {is_training_bn: False}
    acc_test, cxent_test, summaries = sess.run([metric_acc, metric_cxent, merged_summaries_valid],
                                               feed_dict=feed_dict)
    test_writer.add_summary(global_step=best_epoch, summary=summaries)
    test_writer.flush()
    print("EPOCH {} | TEST acc={:.4f}".format(best_epoch, acc_test))
    # ==================================================================


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--epochs", dest="epochs", type=int, default=50)

    parser.add_argument("--lr", dest="lr", type=float, default=0.01)
    parser.add_argument("--momentum", dest="momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default=0.0001)

    parser.add_argument("--output_dir", dest="output_dir", type=str, default="logs/baseline/")
    parser.add_argument("--verbose", dest="verbose", type=bool, default=True)

    args = parser.parse_args(sys.argv[1:])
    main(args)