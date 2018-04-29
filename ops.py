import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def get_update_ops(net, target):
    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=net)
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target)

    update_ops = [target_var.assign(main_var) for main_var, target_var in zip(net_vars, target_vars)]

    return update_ops


def replay_DQN(main, target, batch, gamma=0.95, l_rate=0.01):
    view_stack = []
    v_stack = []
    Q_stack = []
    d_stack = []

    for state, action, reward, done, next_state in batch:
        Q_real = main.predict(np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0), np.expand_dims(state[2], axis=0))
        main_pred_policy = np.argmax(main.predict(np.expand_dims(next_state[0], axis=0), np.expand_dims(next_state[1], axis=0), np.expand_dims(next_state[2], axis=0)))

        if done:
            Q_real[0, action] = reward

        else:
            Q_real[0, action] = reward+gamma*np.squeeze(target.predict(np.expand_dims(next_state[0], axis=0), np.expand_dims(next_state[1], axis=0), np.expand_dims(next_state[2], axis=0)))[main_pred_policy]

        view_stack.append(state[0])
        v_stack.append(state[1])
        d_stack.append(state[2])
        Q_stack.append(Q_real)

    view_stack = np.stack(view_stack, 0)
    v_stack = np.vstack(v_stack)
    Q_stack = np.vstack(Q_stack)
    d_stack = np.vstack(d_stack)

    return main.train(view_stack, v_stack, d_stack, Q_stack, l_rate)


def create_block(x, num_outputs, kernel_size, is_training, skip=True):
    stride_0 = 2 if skip else 1
    conv_0 = slim.conv2d(x, num_outputs, kernel_size, stride=stride_0, padding="SAME", normalizer_fn=slim.batch_norm, normalizer_params={"decay": 0.9, "is_training": is_training})
    conv_1 = slim.conv2d(conv_0, num_outputs, kernel_size, activation_fn=None, stride=1, padding="SAME", normalizer_fn=slim.batch_norm, normalizer_params={"decay": 0.9, "is_training": is_training})
    merge_0 = conv_1 if skip else x+conv_1
    node = tf.nn.relu(merge_0)
    conv_2 = slim.conv2d(node, num_outputs, kernel_size, stride=1, padding="SAME", normalizer_fn=slim.batch_norm, normalizer_params={"decay": 0.9, "is_training": is_training})
    conv_3 = slim.conv2d(conv_2, num_outputs, kernel_size, activation_fn=None, stride=1, padding="SAME", normalizer_fn=slim.batch_norm, normalizer_params={"decay": 0.9, "is_training": is_training})
    merge_1 = conv_1+conv_3
    output = tf.nn.relu(merge_1)

    return output