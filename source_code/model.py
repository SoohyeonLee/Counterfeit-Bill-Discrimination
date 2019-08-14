import tensorflow as tf
import math

class Model():
    def __init__(self, config):
        self.global_step = tf.get_variable('global_step', initializer=0,
                            dtype=tf.int32, trainable=False)

        self.batch_size = config.batch_size

        self.image_holder = tf.placeholder(tf.float32, [self.batch_size, 256, 256, 3])
        self.label_holder = tf.placeholder(tf.int32, [self.batch_size])
        self.keep_prob = tf.placeholder(tf.float32)

        self.starter_learning_rate = config.starter_learning_rate
        self.decay = config.decay
        self.decay_steps = config.decay_steps

    def print_activations(self, tensor):
        print(tensor.op.name, ' ', tensor.get_shape().as_list())

    def variable_with_weight_loss(self, shape, stddev, wl):
        var = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=stddev))
        if wl is not None:
            weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
            tf.add_to_collection('losses', weight_loss)
        return var

    def variable_with_xavier_conv2d(self, shape,name='xavirConv2d'):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        variable = tf.Variable(initializer(shape=shape), name=name)
        return variable

    def inference(self,classes):
        parameters = []
        
        #conv1
        with tf.name_scope('conv1') as scope:
            kernel = self.variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
            conv = tf.nn.conv2d(self.image_holder, kernel, [1, 2, 2, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)
            parameters += [kernel, biases]

            self.print_activations(conv1)
            self.activation_summary(conv1)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        self.print_activations(pool1)

        #conv2
        with tf.name_scope('conv2') as scope:
            kernel = self.variable_with_weight_loss(shape=[5, 5, 64, 48], stddev=5e-2, wl=0.0)
            conv = tf.nn.conv2d(pool1, kernel, [1, 2, 2, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[48], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope)
            parameters += [kernel, biases]

            self.print_activations(conv2)
            self.activation_summary(conv2)

        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    
        self.print_activations(pool2)

        #fc1
        with tf.name_scope('fc1') as scope:
            reshape = tf.reshape(pool2, [self.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = self.variable_with_weight_loss(shape=[dim, 1024], stddev=0.04, wl=0.04)
            biases = tf.Variable(tf.zeros([1024]), name='biases')
            fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope)

            parameters += [weights, biases]
            self.print_activations(fc1)
            self.activation_summary(fc1)

        drop1 = tf.nn.dropout(fc1, self.keep_prob, name='drop1')
        
        self.print_activations(drop1)

        #fc2
        with tf.name_scope('fc2') as scope:
            weights = self.variable_with_weight_loss(shape=[1024, 1024], stddev=0.04, wl=0.04)
            biases = tf.Variable(tf.zeros([1024]), name='biases')
            fc2 = tf.nn.relu(tf.matmul(drop1, weights) + biases, name=scope)

            parameters += [weights, biases]
            self.print_activations(fc2)
            self.activation_summary(fc2)

        drop2 = tf.nn.dropout(fc2, self.keep_prob, name='drop2')
        
        self.print_activations(drop2)

        #fc3
        with tf.name_scope('fc3') as scope:
            weights = self.variable_with_weight_loss(shape=[1024, classes], stddev=1/1024.0, wl=0.0)
            biases = tf.Variable(tf.zeros([classes]), name='biases')
        logits = tf.add(tf.matmul(drop2, weights), biases)
        self.print_activations(logits)
        self.activation_summary(logits)

        return logits, parameters

    def loss(self, logits):
        labels = tf.cast(self.label_holder, tf.int64)
        cross_entropy_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=logits, labels=labels, name='cross_entropy_perexample')

        cross_entropy = tf.reduce_mean(cross_entropy_sum, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy)

        loss_value = tf.add_n(tf.get_collection('losses'), name='total_loss')

        tf.summary.scalar('loss', loss_value)

        return loss_value

    def train_op(self, total_loss):
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,self.decay_steps, self.decay, staircase=True)
        
        train_op = (
				tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
                .minimize(total_loss, global_step=self.global_step)
        )

        return train_op

    def cal_accuracy(self, logits):
        return tf.nn.in_top_k(logits, self.label_holder, 1)

    def activation_summary(self, activation):
        name = activation.op.name
        tf.summary.histogram(name + '/activations', activation)
        tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(activation))

    def logits_summary(self, logits):
        pass
