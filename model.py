from ops import *


class DuelingDQN:

    def __init__(self, name, sess=None):
        self.name = name

        if sess is None:
            self.sess = tf.get_default_session()

        else:
            self.sess = sess

        self._view = None
        self._v = None
        self._d = None
        self.is_training = None
        self.action = None
        self._Q_pred = None
        self._Q_real = None
        self._loss = None
        self._l_rate = None
        self._train_op = None
        self.summary = None
        self.summary_writer = None
        self.global_step = None
        self.global_step_update_op = None

    def build_model(self, view_size):

        with tf.variable_scope(self.name):
            self._view = tf.placeholder(shape=[None, *view_size, 1], dtype=tf.float32)
            self._v = tf.placeholder(shape=[None, 3], dtype=tf.float32)
            self._d = tf.placeholder(shape=[None, 2], dtype=tf.float32)
            self.is_training = tf.placeholder(tf.bool)

            conv_0 = slim.conv2d(self._view, num_outputs=64, kernel_size=7, stride=2, padding="SAME", normalizer_fn=slim.batch_norm, normalizer_params={"decay": 0.9, "is_training": self.is_training})
            pool_0 = slim.max_pool2d(conv_0, kernel_size=3, stride=2, padding="SAME")
            block_0 = create_block(pool_0, 64, 3, self.is_training, skip=False)
            block_1 = create_block(block_0, 128, 3, self.is_training)
            block_2 = create_block(block_1, 128, 3, self.is_training)
            block_3 = create_block(block_2, 256, 3, self.is_training)
            block_4 = create_block(block_3, 256, 3, self.is_training)
            block_5 = create_block(block_4, 512, 3, self.is_training)
            block_6 = create_block(block_5, 512, 3, self.is_training)
            reorder = tf.reshape(block_6, [-1, np.prod(block_6.get_shape().as_list()[1:])])
            features_0 = slim.fully_connected(reorder, num_outputs=512, activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm, normalizer_params={"decay": 0.9, "is_training": self.is_training})

            fc_0_v = slim.fully_connected(self._v, num_outputs=32, activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm, normalizer_params={"decay": 0.9, "is_training": self.is_training})
            fc_1_v = slim.fully_connected(fc_0_v, num_outputs=64, activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm, normalizer_params={"decay": 0.9, "is_training": self.is_training})
            fc_2_v = slim.fully_connected(fc_1_v, num_outputs=128, activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm, normalizer_params={"decay": 0.9, "is_training": self.is_training})
            features_1 = slim.fully_connected(fc_2_v, num_outputs=256, activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm, normalizer_params={"decay": 0.9, "is_training": self.is_training})

            fc_0_d = slim.fully_connected(self._d, num_outputs=32, activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm, normalizer_params={"decay": 0.9, "is_training": self.is_training})
            fc_1_d = slim.fully_connected(fc_0_d, num_outputs=64, activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm, normalizer_params={"decay": 0.9, "is_training": self.is_training})
            fc_2_d = slim.fully_connected(fc_1_d, num_outputs=128, activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm, normalizer_params={"decay": 0.9, "is_training": self.is_training})
            features_2 = slim.fully_connected(fc_2_d, num_outputs=256, activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm, normalizer_params={"decay": 0.9, "is_training": self.is_training})

            split_0_adv, split_0_v = tf.split(features_0, num_or_size_splits=2, axis=-1)
            split_1_adv, split_1_v = tf.split(features_1, num_or_size_splits=2, axis=-1)
            split_2_adv, split_2_v = tf.split(features_2, num_or_size_splits=2, axis=-1)
            concat_adv = tf.concat([split_0_adv, split_1_adv, split_2_adv], axis=-1)
            concat_v = tf.concat([split_0_v, split_1_v, split_2_v], axis=-1)

            adv = slim.fully_connected(concat_adv, num_outputs=7, activation_fn=None)
            v = slim.fully_connected(concat_v, num_outputs=1, activation_fn=None)
            self._Q_pred = v+adv-tf.reduce_mean(adv, axis=1, keepdims=True)
            self._Q_real = tf.placeholder(shape=[None, 7], dtype=tf.float32)

            self._loss = tf.reduce_mean(tf.square(self._Q_pred-self._Q_real))

            self._l_rate = tf.placeholder(dtype=tf.float32)

            norm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(norm_update_ops):
                self._train_op = tf.train.AdamOptimizer(self._l_rate).minimize(self._loss)

    def predict(self, view, v, d):
        return self.sess.run(self._Q_pred, feed_dict={self._view: view,
                                                      self._v: v,
                                                      self._d: d,
                                                      self.is_training: False})

    def train(self, view, v, d, Q, l_rate):
        loss, _ = self.sess.run([self._loss, self._train_op], feed_dict={self._view: view,
                                                                         self._v: v,
                                                                         self._d: d,
                                                                         self.is_training: True,
                                                                         self._Q_real: Q,
                                                                         self._l_rate: l_rate})

        return loss

    def update(self, net):
        self.sess.run(get_update_ops(net, self.name))
