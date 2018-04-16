import numpy as np
import tensorflow as tf
from fgm import fgm
from deepfool import deepfool
from jsma import jsma
import h5py
from cnn_copy import MnistCNN



def model(x, logits=False):
    """
    Wrapper function for net

    :param x: Tensorflow placeholder
        Input data
    :param logits: Bool
        If return logits or not
    :return y:
        Softmax predictions
    :return logits_:
        Logits
    """
    y, logits_ = net.network(x)
    if logits:
        return y, logits_
    else:
        return y

def init_adv_jsma(variable_scope):
    """
    Initiates tensors for jsma adversarials

    :param variable_scope: String
        in which scope the placeholders for adversarial generation, need to be the same as for "network"
    :return:
        Placeholders for adversarial examples generation
    """
    with tf.variable_scope(variable_scope) as scope:
        scope.reuse_variables()
        target = tf.placeholder(tf.int32, (), name='target')
        adv_epochs = tf.placeholder_with_default(20, shape=(), name='epochs')
        adv_eps = tf.placeholder_with_default(0.2, shape=(), name='eps')
        x_jsma = jsma(model, net.inputs, target, eps=adv_eps,
                          epochs=adv_epochs)
        return adv_eps, adv_epochs, x_jsma, target

def init_adv_deepfool(variable_scope):
    """
    Initiates tensors for deepfool adversarials

    :param variable_scope: String
        in which scope the placeholders for adversarial generation, need to be the same as for "network"
    :return:
        Placeholders for adversarial examples generation
    """
    with tf.variable_scope(variable_scope) as scope:
        scope.reuse_variables()
        adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
        x_adv = deepfool(model, net.inputs, epochs=adv_epochs)
        return adv_epochs, x_adv

def init_adv_fgsm(variable_scope):
    """
    Initiates tensors for fsgm adversarials

    :param variable_scope: String
        in which scope the placeholders for adversarial generation, need to be the same as for "network"
    :return:
        Placeholders for adversarial examples generation
    """
    with tf.variable_scope(variable_scope) as scope:
        scope.reuse_variables()
        fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
        fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
        x_fgsm = fgm(model, net.inputs, epochs=fgsm_epochs, eps=fgsm_eps)
        return fgsm_eps, fgsm_epochs, x_fgsm


def make_jsma(sess, x_adv, adv_eps, adv_epochs, target, X_data, epochs, eps, batch_size, n_classes=10):
    """
    Generates JSMA adverarial examples from scratch

    :param sess: Tensorflow session
    :param x_adv: Placeholder
    :param adv_eps: Placeholder
    :param adv_epochs: Placeholder
    :param target: Placeholder
        target class
    :param X_data: Numpy array
        Data to be pertubated
    :param epochs: int
        max epochs
    :param eps: float
        FGSM parameter, size of pertubation.
    :param batch_size: int
        Batch size
    :return X_adv: numpy array
        Adversarial examples
    """
    print('Making adversarials via JSMA')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    for batch in range(n_batch):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {
            net.inputs: np.zeros_like(X_data[start:end]),
            target: np.random.choice(n_classes),
            adv_epochs: epochs,
            adv_eps: eps}
        adv = sess.run(x_adv, feed_dict=feed_dict)
        X_adv[start:end] = adv

    return X_adv

def make_deepfool(sess, x_adv, adv_epochs, X_data, epochs, batch_size):
    """
    Generates deepfool adverarial examples

    :param sess: Tensorflow session
    :param x_adv: Placeholder
    :param adv_epochs: Placeholder
    :param X_data: Numpy array
            Data to be pertubated
    :param epochs: int
        max epochs
    :param batch_size: int
        Batch Size
    :return X_adv: numpy array
        Adversarial examples
    """
    print('Making adversarials via DeepFool')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    for batch in range(n_batch):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(x_adv, feed_dict={net.inputs: X_data[start:end],
                                            adv_epochs: epochs})
        X_adv[start:end] = adv

    return X_adv


def make_fgsm(sess, x_fgms, fgsm_eps, fgsm_epochs, X_data, epochs, eps, batch_size):
    """
    Generates fast gradient sign method adverarial examples

    :param sess: Tensorflow session
    :param x_fgms: Placeholder
    :param fgsm_eps: Placeholder
    :param fgsm_epochs: Placeholder
    :param X_data: Numpy array
        Data to be pertubated
    :param epochs: int
        max epochs
    :param eps: float
        FGSM parameter, size of pertubation.
    :param batch_size: int
        Batch size
    :return X_adv: numpy array
        Adversarial examples
    """
    print('Making adversarials via FGSM')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    for batch in range(n_batch):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(x_fgms, feed_dict={
            net.inputs: X_data[start:end],
            fgsm_eps: eps,
            fgsm_epochs: epochs})
        X_adv[start:end] = adv

    return X_adv


def make_adversarials(sess, net_copy, data, labels, eps=0.01, epochs=3, batch_size=128, seed=1337, filename=None, type='Deepfool', wrong=False):
    """
    Generates adverarials specified by type.

    :param sess: Tensorflow session
    :param net_copy: A copy of the network
    :param data: numpy array
        data to be pertubated
    :param labels: numpy array
        Labels to data
    :param eps: float
        Parameter, size of pertubation.
    :param epochs: int
        max epochs
    :param batch_size: int
        Batch size
    :param seed: float
        random seed
    :param filename: String
        Filename is examples is to be saved
    :param type: String
        type of adversarial examples
    :param wrong: Boolean
        Specifies if the adversarial should be miss-classified by the network to be returned
    :return X_adv: numpy array
         Adversarial examples.

    :raises Exception: If no valid type was provided
    :raises Exception: If no adversarials successfully fooled the network
    """
    np.random.seed(seed)
    if 'net' not in globals():
        global net
        net = MnistCNN(sess, save_dir='../Thesis_CNN_mnist/Mnist_save/')
    if type == 'FGSM':
        fgsm_eps, fgsm_epochs, x_fgsm = init_adv_fgsm('Predictions')
        X_adv = make_fgsm(sess, x_fgsm, fgsm_eps, fgsm_epochs, data, epochs, eps, batch_size)
    elif type == 'Deepfool':
        adv_epochs, x_adv = init_adv_deepfool('Predictions')
        X_adv = make_deepfool(sess, x_adv, adv_epochs, data, epochs, batch_size)
    elif type == 'JSMA':
        adv_eps, adv_epochs, x_adv, target = init_adv_jsma('Predictions')
        X_adv = make_jsma(sess, x_adv, adv_eps, adv_epochs, target, data, epochs, eps, batch_size, n_classes=10)
    else:
        raise Exception('No valid adversarial type, available types is "FGSM", "Deepfool" or "JSMA"')
    if wrong:
        adv_predictions, _, _ =  net_copy.predict(X_adv)
        indexes = (adv_predictions != np.argmax(labels, 1))
        X_adv = X_adv[indexes]
        if indexes.sum() == 0: raise Exception('No generated adversarials managed to fool the network')
        print(f'{len(indexes)-indexes.sum()}/{len(indexes)} adversarials was classified correctly and removed')
    if filename is not None:
        f = h5py.File(filename+'.h5', 'w')
        f.create_dataset("X_adv", data=X_adv)
        f.close()
    return X_adv


def fetch_data(filename):
    """
    Loads and return existing datafile
    :param filename: String
        The filename
    :return X_adv: numpy array
        The adverarial examples contained in filename
    """
    filename = filename + '.h5'
    try:
        print("Loading existing file '{}'.".format(filename))
        f = h5py.File(filename, 'r')
        X_adv = f['X_adv'][:]
        f.close()
        return X_adv
    except:
        print(f'The following file does not exist: {filename}')