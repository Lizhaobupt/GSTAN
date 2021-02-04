# @Time     : Jan. 12, 2019 17:45
# @Author   : Veritas YIN
# @FileName : layers.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import tensorflow as tf
# from tensorflow.contrib.layers import l2_regularizer
# regularizer = l2_regularizer(0.0001)
from utils.math_graph import *

def gconv(x, theta, Ks, c_in, c_out, ker_num):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in].
    :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.
    :param Ks: int, kernel size of graph convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, n_route, c_out].
    '''
    _, T, N, C = x.get_shape().as_list()
    Wt1 = tf.get_variable('wt1', shape=[T, N, 2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
    Wt2 = tf.get_variable('wt2', shape=[T, 2, N], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
    W = tf.matmul(Wt1, Wt2)
    print('WWWW',W.shape)
    W = tf.nn.relu(W)
    W = tf.nn.softmax(W, axis=1)
    kernel =  W 
    # x -> [batch_size, c_in, T, n_route] -> [batch_size*c_in, T, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [-1, T, N])
    # [batch_size, T, n_route]->[T, batch_size, N]
    x_tmp = tf.transpose(x_tmp, [1, 0, 2]) 
    # x_mul = [T, batch_size, n_route]->[batch_size, T, n_route]
    x_mul = tf.transpose(tf.matmul(x_tmp, kernel), [1, 0, 2])
    # x_mul = [batch_size, T, n_route]->[batch_size, c_in, T, n_route]
    x_mul = tf.reshape(x_mul, [-1, c_in, T, N])
    # x_ker -> [batch_size, T, n_route, c_in] -> [batch_size, n_route, c_in*Ks]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 2, 3, 1]), [-1, c_in])
    # x_gconv -> [n_route, batch_size, c_out] -> [batch_size, n_route, c_out]
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, N, c_out])
    return x_gconv


def layer_norm(x, scope):
    '''
    Layer normalization function.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param scope: str, variable scope.
    :return: tensor, [batch_size, time_step, n_route, channel].
    '''
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x, axes=[2, 3], keep_dims=True)

    with tf.variable_scope(scope):
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return _x


def temporal_conv_layer(x, Kt, c_in, c_out, keep_prob, act_func='tanh'):
    '''
    Temporal convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in == c_out:
        input = x
    else:
        with tf.variable_scope(f't_block_res'):
            wt = tf.get_variable(name='wt1', shape=[1, 1, c_in, c_out], dtype=tf.float32)
            tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
            input = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='SAME')

    with tf.variable_scope(f't_block_1'):
        wt1_1 = tf.get_variable(name='wt1', shape=[2, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt1_1))
        bt1_1 = tf.get_variable(name='bt1', initializer=tf.zeros([c_out]), dtype=tf.float32)
        # two
        wt1_2 = tf.get_variable(name='wt2', shape=[2, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt1_2))
        bt1_2 = tf.get_variable(name='bt2', initializer=tf.zeros([c_out]), dtype=tf.float32)

    x_conv1_1 = tf.nn.conv2d(x, wt1_1, strides=[1, 2, 1, 1], padding='VALID') + bt1_1
    x_conv1_2 = tf.nn.conv2d(x, wt1_2, strides=[1, 2, 1, 1], padding='VALID') + bt1_2

    x_conv1 = tf.concat([x_conv1_1, x_conv1_2], axis=1)
    x_conv1 = x_conv1+ input

    x_conv1 = layer_norm(x_conv1, scope='t_block1')
    with tf.variable_scope(f't_block_2'):
        wt2_1 = tf.get_variable(name='wt1', shape=[Kt, 1, c_out, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt2_1))
        bt2_1 = tf.get_variable(name='bt1', initializer=tf.zeros([c_out]), dtype=tf.float32)
        # two
        wt2_2 = tf.get_variable(name='wt2', shape=[Kt, 1, c_out, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt2_2))
        bt2_2 = tf.get_variable(name='bt2', initializer=tf.zeros([c_out]), dtype=tf.float32)
        # three
        wt2_3 = tf.get_variable(name='wt3', shape=[Kt, 1, c_out, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt2_3))
        bt2_3 = tf.get_variable(name='bt3', initializer=tf.zeros([c_out]), dtype=tf.float32)
    x_conv2_1 = tf.nn.conv2d(x_conv1, wt2_1, strides=[1, 3, 1, 1], padding='VALID') + bt2_1
    x_conv2_2 = tf.nn.conv2d(x_conv1, wt2_2, strides=[1, 3, 1, 1], padding='VALID') + bt2_2
    x_conv2_3 = tf.nn.conv2d(x_conv1, wt2_3, strides=[1, 3, 1, 1], padding='VALID') + bt2_3
    x_conv2 = tf.concat([x_conv2_1, x_conv2_2, x_conv2_3], axis=1)
    x_conv2 = x_conv2 + input
    x_conv2 = layer_norm(x_conv2, scope='t_block_2')

    with tf.variable_scope(f't_block_3'):
        wt3_1 = tf.get_variable(name='wt1', shape=[2, 1, c_out, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt1_1))
        bt3_1 = tf.get_variable(name='bt1', initializer=tf.zeros([c_out]), dtype=tf.float32)
        # two
        wt3_2 = tf.get_variable(name='wt2', shape=[2, 1, c_out, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt1_2))
        bt3_2 = tf.get_variable(name='bt2', initializer=tf.zeros([c_out]), dtype=tf.float32)
    
    x_conv3_1 = tf.nn.conv2d(x_conv2, wt3_1, strides=[1, 2, 1, 1], padding='VALID') + bt3_1
    x_conv3_2 = tf.nn.conv2d(x_conv2, wt3_2, strides=[1, 2, 1, 1], padding='VALID') + bt3_2
    x_conv = tf.concat([x_conv3_1, x_conv3_2], axis=1)

    x_conv = input + x_conv
    x_conv = layer_norm(x_conv, scope='t_block_3')
    #   x_conv = tf.concat([tf.zeros([tf.shape(x)[0], Kt-1, n, c_out]), x_conv], axis=1)
    if act_func == 'linear':
        return x_conv
    elif act_func == 'sigmoid':
        return tf.nn.sigmoid(x_conv)
    elif act_func == 'tanh':
        return tf.nn.tanh(x_conv)
    elif act_func == 'relu':
        return tf.nn.relu(x_conv)
    else:
        raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')


def spatio_conv_layer(x, Ks, c_in, c_out, keep_prob, ker_num):
    '''
    Spatial graph convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()
    
    with tf.variable_scope(f's_block_res'):
        wt = tf.get_variable(name='wt1', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        x_input = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='SAME')
            

    with tf.variable_scope(f's_block_1'):
        ws_1 = tf.get_variable(name='ws', shape=[c_in, c_out], dtype=tf.float32)
        bs_1 = tf.get_variable(name='bs', initializer=tf.zeros([c_out]), dtype=tf.float32)
        # x -> [batch_size*time_step, n_route, c_in] -> [batch_size*time_step, n_route, c_out]
        x_gconv_1 = gconv(x, ws_1, Ks, c_in, c_out, ker_num=ker_num) + bs_1
        # x_g -> [batch_size, time_step, n_route, c_out]
        x_gc_1 = tf.reshape(x_gconv_1, [-1, T, n, c_out])

    x_gc = x_gc_1 + x_input
                                           

  #  x_gc = tf.nn.dropout(x_gc, keep_prob=keep_prob)
    x_gc = layer_norm(x_gc, scope='s_block')
    return tf.nn.relu(x_gc)


def st_conv_block(x, Ks, Kt, channels, scope, keep_prob, ker_num, act_func='tanh'):
    '''
    Spatio-temporal convolutional block, which contains two temporal gated convolution layers
    and one spatial graph convolution layer in the middle.
    :param x: tensor, batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param channels: list, channel configs of a single st_conv block.
    :param scope: str, variable scope.
    :param keep_prob: placeholder, prob of dropout.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    c_si, c_t, c_oo = channels
    _, T, n, _ = x.get_shape().as_list()

    with tf.variable_scope(f'stn_block_{scope}_in'):
        x_s = temporal_conv_layer(x, Kt, c_si, c_t, keep_prob=keep_prob, act_func='relu')
        x_t = spatio_conv_layer(x, Ks, c_si, c_t,keep_prob=keep_prob, ker_num=ker_num)
        x_st = tf.concat([x_s, x_t], axis=-1)
    
    with tf.variable_scope(f'stn_block_{scope}_out'):
        x_ot = temporal_conv_layer(x_st, Kt, 2*c_t, 48, keep_prob=keep_prob, act_func='relu')
        x_os = spatio_conv_layer(x_st, Ks, 2*c_t, 48, keep_prob=keep_prob, ker_num=ker_num)
        x_ost = tf.concat([x_ot, x_os], axis=-1)

    # x_ln = layer_norm(x_o, f'layer_norm_{scope}')
    return x_ost


def fully_con_layer(x, n, channel, c_out, scope):
    '''
    Fully connected layer: maps multi-channels to one.
    :param x: tensor, [batch_size, 1, n_route, channel].
    :param n: int, number of route / size of graph.
    :param channel: channel size of input x.
    :param scope: str, variable scope.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    _, T, _, _ = x.get_shape().as_list()
    w1 = tf.get_variable(name=f'w1_{scope}', shape=[n, 2], dtype=tf.float32)
    w2 = tf.get_variable(name=f'w2{scope}', shape=[2, channel*c_out], dtype=tf.float32)
    w = tf.matmul(w1, w2)
    w = tf.reshape(w, [n, channel, c_out])
    b = tf.get_variable(name=f'b_{scope}', initializer=tf.zeros([c_out]), dtype=tf.float32)
    # batch_size, n, c 
    x = tf.reshape(x, [-1, n, channel])
    # n, batch_szie, c
    x = tf.transpose(x, [1, 0, 2])
    x_out = tf.matmul(x, w) + b
    # batch_size, n, c
    x_out = tf.transpose(x_out, [1, 0, 2])
    # batch_szie, t, n ,c_out
    x_out = tf.reshape(x_out, [-1, T, n, c_out])
    return x_out

def output_layer(x, T, scope, act_func='tanh'):
    '''
    Output layer: temporal convolution layers attach with one fully connected layer,
    which map outputs of the last st_conv block to a single-step prediction.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param T: int, kernel size of temporal convolution.
    :param scope: str, variable scope.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    _, _, n, channel = x.get_shape().as_list()

    with tf.variable_scope(f'{scope}_out'):

        x_o = temporal_conv_layer(x, T, channel, channel, keep_prob=1, act_func='tanh')

    x_fc = fully_con_layer(x_o, n, channel, 64, f'{scope}_3')
    x_fc = layer_norm(x_fc, 'out')
    x_fc_out = fully_con_layer(x_fc, n, 64, 1, f'{scope}_4')
    return x_fc_out


def variable_summaries(var, v_name):
    '''
    Attach summaries to a Tensor (for TensorBoard visualization).
    Ref: https://zhuanlan.zhihu.com/p/33178205
    :param var: tf.Variable().
    :param v_name: str, name of the variable.
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(f'mean_{v_name}', mean)

        with tf.name_scope(f'stddev_{v_name}'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(f'stddev_{v_name}', stddev)

        tf.summary.scalar(f'max_{v_name}', tf.reduce_max(var))
        tf.summary.scalar(f'min_{v_name}', tf.reduce_min(var))

        tf.summary.histogram(f'histogram_{v_name}', var)
