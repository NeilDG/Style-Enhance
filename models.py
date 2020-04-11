import tensorflow as tf
from keras.callbacks import TensorBoard
import numpy as np

def generator(input_image, ground_truth=None, scope = "", n_resnet=4, isTraining=True, use_sn=False):

    with tf.variable_scope("generator", reuse = tf.AUTO_REUSE):
        print("Generator")
        c1 = conv(input_image, 9, 64, use_sn=use_sn)
        c1 = tf.nn.relu(c1)
        temp = c1
        
        for i in range(n_resnet // 2):
            with tf.variable_scope("residual" + str(i)):
                temp = res_block(temp, 3, 64, isTraining=isTraining, use_sn=use_sn, ground_truth=ground_truth)
                
        for i in range(n_resnet // 2, n_resnet):
            with tf.variable_scope("residual" + str(i)):
                temp = res_block(temp, 3, 64, isTraining=isTraining, use_sn=use_sn, ground_truth=ground_truth)
        
        temp = conv(temp, 3, 64, use_sn=use_sn)
        temp = normalize(temp, isTraining=isTraining)
        temp = temp + c1

        with tf.variable_scope("final"):
            enhanced = conv(temp, 9, 3, use_sn=use_sn)
            enhanced = tf.nn.tanh(enhanced)
    return enhanced

def res_block(features, kernel, filter_out, isTraining=True, use_bias=False, use_sn=False, ground_truth= None):
    temp = conv(features, kernel, filter_out, use_sn=use_sn)
    temp = normalize(temp, isTraining=isTraining)
    temp = tf.nn.relu(temp)

    temp = conv(temp, kernel, filter_out, use_sn=use_sn)
    temp = normalize(temp, isTraining=isTraining)
    temp = temp + features
    
    return temp

def discriminator(image, scope = "", use_sn=False):
    with tf.variable_scope("discriminator" + scope, reuse = tf.AUTO_REUSE):
        print("Discriminator")
        with tf.variable_scope("layer1"):
            temp = conv(image, 3, 64, use_sn=use_sn)
            temp = leaky_relu(temp)

        with tf.variable_scope("layer2"):
            temp = conv(temp, 3, 64, strides=2, use_sn=use_sn)
            temp = normalize(temp, normalization_layer="BN")
            temp = leaky_relu(temp)

        with tf.variable_scope("layer3"):
            temp = conv(temp, 3, 128, use_sn=use_sn)
            temp = normalize(temp, normalization_layer="BN")
            temp = leaky_relu(temp)

        with tf.variable_scope("layer4"):
            temp = conv(temp, 3, 128, strides=2, use_sn=use_sn)
            temp = normalize(temp, normalization_layer="BN")
            temp = leaky_relu(temp)

        with tf.variable_scope("layer5"):
            temp = conv(temp, 3, 256, use_sn=use_sn)
            temp = normalize(temp, normalization_layer="BN")
            temp = leaky_relu(temp)

        with tf.variable_scope("layer6"):
            temp = conv(temp, 3, 256, strides=2, use_sn=use_sn)
            temp = normalize(temp, normalization_layer="BN")
            temp = leaky_relu(temp)

        with tf.variable_scope("layer7"):
            temp = conv(temp, 3, 512, use_sn=use_sn)
            temp = normalize(temp, normalization_layer="BN")
            temp = leaky_relu(temp)

        with tf.variable_scope("layer8"):
            temp = conv(temp, 3, 512, strides=2, use_sn=use_sn)
            temp = normalize(temp, normalization_layer="BN")
            temp = leaky_relu(temp)
            
        temp = tf.keras.layers.Flatten()(temp)
        temp = tf.layers.dense(temp, units = 1024, activation = None)
        temp = leaky_relu(temp)
        logits = tf.layers.dense(temp, units = 1, activation = None)
        probability = tf.nn.sigmoid(logits)
        
        return 1, logits, probability

############################################################################
# Convolutional layers
############################################################################

def weight_variable(shape, name='Variable', initializer=tf.random_normal_initializer(stddev=0.02)):
    #initial = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initializer(shape), name=name)

def bias_variable(shape, name='Variable'):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv(features, kernel, filter_out, strides=1, use_bias=False, use_sn=False, padding='SAME', initializer=tf.random_normal_initializer(stddev=0.02)):
    print(features.shape)
    filter_in = features.shape[3].value
    W = weight_variable([kernel, kernel, filter_in, filter_out], initializer=initializer)
    
    if(use_sn):
        W = spectral_norm(W)
    
    b = 0
    if(use_bias):
        b = bias_variable([filter_out])
    return tf.nn.conv2d(features, W, strides=[1, strides, strides, 1], padding=padding) + b

############################################################################
# Activation function
############################################################################

def leaky_relu(x, alpha = 0.2):
    return tf.maximum(alpha * x, x)

############################################################################
# Normalization
############################################################################

def instance_norm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift

def normalize(net, normalization_layer="BN", isTraining=True):
    #g_init = tf.random_normal_initializer(1., 0.02)
    if(normalization_layer == "BN"):
        if(isTraining):
            print("Batch Norm")
        return tf.keras.layers.BatchNormalization()(net, training=isTraining)
    elif(normalization_layer == "IN"):
        print("Instance Norm")
        return instance_norm(net)
    else:
        return tf.keras.layers.BatchNormalization()(net, training=isTraining)
        
    #return(net)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)


    return w_norm