import utils
import vgg
import models
import tensorflow as tf

def generator_loss(model, gen_hq, hq_image, logits_hq, logits_en, label_smoothing, batch_size):
    if(model == "hinge"):
        d_loss_real_hq = tf.reduce_mean(tf.nn.relu(1. - logits_hq))
        d_loss_en = tf.reduce_mean(tf.nn.relu(1. + logits_en))
    else:
        d_loss_real_hq = tf.reduce_mean(utils.sigmoid_cross_entropy_with_logits(logits_hq, tf.ones_like(logits_hq) * label_smoothing)) 
        d_loss_en = tf.reduce_mean(utils.sigmoid_cross_entropy_with_logits(logits_en, tf.zeros_like(logits_en)))
        
    #Generator Loss
    if(model == "hinge"):
        loss_discrim = d_loss_real_hq + d_loss_en
        loss_gen = -tf.reduce_mean(logits_en)
        gan_model = model
    elif(model == "mgan"):
        loss_discrim = d_loss_real_hq + d_loss_en
        loss_gen = -tf.reduce_mean(tf.log(tf.clip_by_value(d_loss_en,1e-10,1.0)))
        gan_model = model
    else:
        loss_discrim = d_loss_real_hq + d_loss_en
        loss_gen = tf.reduce_mean(utils.sigmoid_cross_entropy_with_logits(logits_en, tf.ones_like(logits_en)))
        gan_model = "gan"
    return loss_gen, loss_discrim, gan_model

def perceptual_loss(gen_hq_vgg, hq_vgg, VGG_LAYER, function, batch_size):
    content_size = utils._tensor_size(hq_vgg[VGG_LAYER]) * batch_size
    loss_vgg = tf.reduce_sum(tf.pow(gen_hq_vgg[VGG_LAYER] - hq_vgg[VGG_LAYER], 2)) / content_size
    
    return loss_vgg


def pixel_loss(pixel, hq_image, gen_hq, batch_size, delta=1):
    if(pixel == "L1"):
        return tf.reduce_sum(tf.math.abs(hq_image - gen_hq)) / (2 * batch_size)
    elif(pixel == "huber"):
        error = hq_image - gen_hq
        cond  = tf.math.abs(error) < delta
        loss_square = 0.5 * tf.pow(error,2)
        loss_linear = delta * (tf.math.abs(error) - 0.5 * delta)
        return tf.reduce_sum(tf.where(cond, loss_square, loss_linear)) / (2 * batch_size)
    else:
        #L2 loss
        return tf.reduce_sum(tf.pow(hq_image - gen_hq, 2))/(2 * batch_size)
    