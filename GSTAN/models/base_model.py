# @Time     : Jan. 12, 2019 19:01
# @Author   : Veritas YIN
# @FileName : base_model.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from models.layers import *
from os.path import join as pjoin
import tensorflow as tf


def build_model(inputs, n_his, Ks, Kt, blocks, keep_prob, x_state):
    '''
    Build the base model.
    :param inputs: placeholder.
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param keep_prob: placeholder.
    '''
    x_1 = inputs[:, :n_his, :, :]
    x_2 = inputs[:, n_his:2*n_his, :, :]
    # Ko>0: kernel size of temporal convolution in the output layer.
    Ko = n_his
    # ST-Block
    
    with tf.variable_scope('x_1'):
        for i, channels in enumerate(blocks):
            x_1 = st_conv_block(x_1, Ks, Kt, channels, i, keep_prob, ker_num = 0, act_func='tanh')
            Ko -= 2 * (Ks - 1)
    with tf.variable_scope('x_2'):
        for i, channels in enumerate(blocks):
            x_2 = st_conv_block(x_2, Ks, Kt, channels, i, keep_prob, ker_num= 0, act_func='tanh')
   # Output Layer
    c_out = x_1.shape[3]
    n = x_1.shape[2]
#     print(x_1.shape)
    x = tf.concat([x_1, x_2], axis=2)
    #     print(x.shape)
    x = tf.reshape(x, [-1, 2*n_his, n, c_out])
            
    x = tf.nn.avg_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding="SAME")
#     print(x.shape)
    y = output_layer(x, 3, 'output_layer')

# x = tf.nn.conv2d(x, conv1_weights, strides=[1, 2, 1, 1], padding="SAME") + conv1_biases
    
#     y = output_layer(x, 3, 'output_layer_1')
#    if Ko > 1:
    
#    else:
#        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

#    y = tf.transpose(y, [0, 3, 2, 1])
#    y = tf.layers.dense(inputs=y, units=n_his, use_bias=False, name='output')
#    y = tf.transpose(y, [0, 3, 2, 1])
#    y = tf.concat([y1, y2, y3], axis=2)
   

    tf.add_to_collection(name='copy_loss',
                         value=tf.nn.l2_loss(inputs[:, 2*n_his-1:2*n_his, :, :] - inputs[:, 2*n_his:2*n_his + 1, :, :]))

#    train_loss = tf.reduce_mean(tf.abs(y - inputs[:, n_his:, :, :]))
    single_pred = y
#     single_pred = y
    tf.add_to_collection(name='y_pred', value=single_pred)
    y_pred = single_pred*x_state['std']
    y_truth = inputs[:, 2*n_his:, :, :] *x_state['std'] 

#    train_loss =  tf.nn.l2_loss(tf.subtract(single_pred, inputs[:, 2*n_his:, :, :]))
    train_loss =  tf.reduce_mean(tf.abs(tf.subtract(y_pred, y_truth)))
    return train_loss, single_pred


def model_save(sess, global_steps, model_name, save_path='./output/models_sw/'):
    '''
    Save the checkpoint of trained model.
    :param sess: tf.Session().
    :param global_steps: tensor, record the global step of training in epochs.
    :param model_name: str, the name of saved model.
    :param save_path: str, the path of saved model.
    :return:
    '''
    saver = tf.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name), global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')
