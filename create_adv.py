import numpy as np
#import sys
from cnn import MnistCNN
#sys.path.append('../Thesis_Utilities/')
#from utilities import load_datasets
import tensorflow as tf
from fgm import fgm
import h5py
#import matplotlib.pyplot as plt


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


def init_adv(variable_scope):
    """
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


def make_fgmt(sess, x_fgms, fgsm_eps, fgsm_epochs, X_data, epochs, eps, batch_size):
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
        Adverarial examples
    """
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    for batch in range(n_batch):
        print('batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(x_fgms, feed_dict={
            net.inputs: X_data[start:end],
            fgsm_eps: eps,
            fgsm_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv


def make_adversarials(sess, net_copy, data, labels, eps=0.01, epochs=10, batch_size=128, seed=1337, filename=None):
    """

    :param sess: Tensorflow session
    :param net_copy: A copy of the network
    :param data: numpy array
        data to be pertubated
    :param labels: numpy array
        Labels to data
    :param epochs: int
        max epochs
    :param eps: float
        FGSM parameter, size of pertubation.
    :param batch_size: int
        Batch size
    :param seed: float
        random seed
    :param filename: String
        Filename is examples is to be saved
    :return X_adv: numpy array
         Adverarial examples that are classified different from their labels
    """
    np.random.seed(seed)
    global net
    net = MnistCNN(sess, save_dir='../Thesis_CNN_mnist/Mnist_save/')
    fgsm_eps, fgsm_epochs, x_fgsm = init_adv('Predictions')
    X_adv = make_fgmt(sess, x_fgsm, fgsm_eps, fgsm_epochs, data, epochs, eps, batch_size)
    adv_predictions, _, _ =  net_copy.predict(X_adv)
    indexes = (adv_predictions != np.argmax(labels, 1))
    if filename is not None:
        f = h5py.File(filename+'.h5', 'w')
        f.create_dataset("X_adv", data=X_adv[indexes])
        f.close()
    return X_adv[indexes]


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

"""

x_train, y_train, x_val, y_val, x_test, y_test = load_datasets(test_size=10000, val_size=5000, omniglot_bool=False,
                                                               name_data_set='data_omni_seed1337.h5', force=False,
                                                               create_file=True, r_seed=123)
tf.reset_default_graph()
sess = tf.Session()
net2 = MnistCNN(sess, save_dir='../Thesis_CNN_mnist/Mnist_save/')
tf.reset_default_graph()
sess = tf.Session()
s = 1337
X_adv = make_adversarials(sess, net2, x_train[0:9], y_train[0:9], eps=0.01, epochs=20, batch_size=3,
                          seed=s, filename='adv_'+str(s))

X_adv = fetch_data('adv_'+str(s))
fig1, axes1 = plt.subplots(figsize=(5, 5), nrows=3, ncols=3, sharex=True, sharey=True, squeeze=False)
k = 0
for ax_row in axes1:
    for ax in ax_row:
        ax.imshow(np.squeeze(X_adv[k,:,:,:]))
        label,_,_ = net2.predict(np.expand_dims(X_adv[k,:,:,:],0))
        ax.set_title(f'{label[0]}')
        k = k+1
plt.suptitle('Adversarial images')
plt.tight_layout()
plt.show()
"""