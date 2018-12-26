import tensorflow as tf


class MobileNetv2:
    def __init__(self,
                 num_classes=10,
                 gamma=1.,
                 data_format="channels_last",
                 is_training_bn=True,
                 expansion=[1, 6, 6, 6, 6, 6, 6],
                 fan_out=[16, 24, 32, 64, 96, 160, 320],
                 num_blocks=[1, 2, 3, 4, 3, 3, 1],
                 stride=[1, 1, 2, 2, 1, 2, 1]):

        if len(expansion) != len(fan_out):
            raise ValueError("expansion and fan_out must have the same number of elements : "
                             "{} != {}".format(len(expansion), len(fan_out)))
        if len(expansion) != len(num_blocks):
            raise ValueError("expansion and num_blocks must have the same number of elements : "
                             "{} != {}".format(len(expansion), len(num_blocks)))

        if len(expansion) != len(stride):
            raise ValueError("expansion and stride must have the same number of elements : "
                             "{} != {}".format(len(expansion), len(stride)))

        self.num_classes = num_classes
        self.gamma = gamma
        self.data_format = data_format
        self.is_training_bn = is_training_bn
        self.expansion = expansion
        self.fan_out = fan_out
        self.num_blocks = num_blocks
        self.stride = stride

        if data_format == "channels_last":
            self.axis_bn = -1
        else:
            self.axis_bn = 1

    def block(self, inputs, expansion, fan_in, fan_out,  stride):
        fan = int(expansion*fan_in)

        x = tf.layers.conv2d(inputs=inputs, filters=fan, kernel_size=1, strides=1, padding="same", use_bias=False,
                             data_format=self.data_format, name="conv1-1x1")
        x = tf.layers.batch_normalization(inputs=x, axis=self.axis_bn, training=self.is_training_bn,
                                          fused=True, name="bn1")
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(inputs=x, filters=fan, kernel_size=3, strides=stride, padding="same", use_bias=False,
                             data_format=self.data_format, name="conv2d-3x3-not-grouped")
        # this is a naive implementation of grouped conv. To optimize later.
        # x = grouped_convolution(inputs=x, groups=fan, kernel_size=3, strides=stride, padding="same", use_bias=False,
        #                         data_format=self.data_format, name="conv2d-3x3-grouped")

        x = tf.layers.batch_normalization(inputs=x, axis=self.axis_bn, training=self.is_training_bn,
                                          fused=True, name="bn2")
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(inputs=x, filters=fan_out, kernel_size=1, strides=1, padding="same", use_bias=False,
                             data_format=self.data_format, name="conv3-1x1")
        x = tf.layers.batch_normalization(inputs=x, axis=self.axis_bn, training=self.is_training_bn,
                                          fused=True, name="bn3")

        if stride == 1 and fan_in != fan_out:
            shortcut = tf.layers.conv2d(inputs=inputs, filters=fan_out, kernel_size=1, strides=1, padding='same',
                                        use_bias=False, data_format=self.data_format, name="conv-shortcut")
            shortcut = tf.layers.batch_normalization(inputs=shortcut, axis=self.axis_bn, training=self.is_training_bn,
                                                     fused=True, name="bn-shortcut")
        else:
            shortcut = inputs
        out = x + shortcut if stride == 1 else x

        return out

    def partial_inference(self, inputs):

        with tf.variable_scope('InputBlock'):
            x = tf.layers.conv2d(inputs=inputs, filters=int(self.gamma*32), kernel_size=3, strides=1,
                                 padding='same', use_bias=False, data_format=self.data_format, name="conv1")
            x = tf.layers.batch_normalization(inputs=x, axis=self.axis_bn, training=self.is_training_bn,
                                              fused=True, name="bn1")
            x = tf.nn.relu(x, name='relu1')
            print("InputBlocks : {}".format(x))
        fan_in = int(self.gamma*32)

        for i in range(len(self.expansion)):
            with tf.variable_scope("BlockMobileNet-{}".format(i)):

                strides = [self.stride[i]] + [1]*(self.num_blocks[i] -1)
                for j, stride in enumerate(strides):
                    with tf.variable_scope("SubBlockMobileNet-{}".format(j)):

                        x = self.block(inputs=x,
                                       expansion=self.expansion[i],
                                       fan_in=fan_in,
                                       fan_out=self.fan_out[i],
                                       stride=stride)
                fan_in = self.fan_out[i]
                print("BlockMobileNet-{} : {}".format(i, x))

        with tf.variable_scope('OutputBlock'):
            x = tf.layers.conv2d(inputs=x, filters=int(self.gamma * 1280), kernel_size=1, strides=1,
                                 padding='same', use_bias=False, data_format=self.data_format, name="conv2")
            x = tf.layers.batch_normalization(inputs=x, axis=self.axis_bn, training=self.is_training_bn,
                                              fused=True, name="bn2")
            x = tf.nn.relu(x, name='relu')
            # print("feature maps before AVG-POOL : {}".format(x))
            x = tf.layers.average_pooling2d(inputs=x, pool_size=4, strides=1, data_format=self.data_format)
            # print("feature maps after ABG-POOL : {}".format(x))

            x = tf.layers.flatten(x)
        return x

    def complete_inference(self, inputs):
        final_features = self.partial_inference(inputs)
        with tf.name_scope("ClassifierLayer"):
            logits = tf.layers.dense(inputs=final_features, units=self.num_classes)

        return logits


def grouped_convolution(inputs, groups,
                        kernel_size=3, strides=1, padding="same",
                        use_bias=False, data_format="channels_last",
                        name="grouped_conv2d"):
    axis = -1 if data_format=="channels_last" else 1
    fan_in = inputs.get_shape().as_list()[axis]
    assert fan_in % groups == 0

    filters = fan_in//groups
    list_inputs = tf.split(inputs, groups, axis=axis)
    list_outputs = []
    with tf.variable_scope(name):
        for i, partial_inputs in enumerate(list_inputs):
            partial_outputs = tf.layers.conv2d(partial_inputs, filters=filters, kernel_size=kernel_size,
                                               strides=strides, padding=padding, data_format=data_format,
                                               use_bias=use_bias, name="conv-{}".format(i))
            list_outputs.append(partial_outputs)

        outputs = tf.concat(list_outputs, axis=axis)
    return outputs