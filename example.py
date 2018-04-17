import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from create_adv import make_adversarials
from cnn_copy import MnistCNN
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)

# Create model
tf.reset_default_graph()
sess = tf.Session()
net2 = MnistCNN(sess, save_dir='../Thesis_CNN_mnist/Mnist_save/')
tf.reset_default_graph()
sess = tf.Session()

# Try to create 500 adversarials of each type
index = np.arange(len(mnist.train.images))
start = 0
step = 30

# JSMA adversarial
X_adv_1 = make_adversarials(sess, net2, mnist.train.images[index[start:start + step]],
                            mnist.train.labels[start:start + step], eps=0.2,
                            epochs=120,
                            batch_size=1,
                            seed=2, filename='adv_JSMA' + str(2), type='JSMA', wrong=False)
# FGSM adversarial
start = start + step
X_adv_2 = make_adversarials(sess, net2, mnist.train.images[index[start:start + step]],
                            mnist.train.labels[start:start + step], eps=0.02,
                            epochs=16,
                            batch_size=1, seed=2, filename='adv_FGSM' + str(2), type='FGSM', wrong=True)
# Deepfool adversarial
start = start + step
X_adv_3 = make_adversarials(sess, net2, mnist.train.images[index[start:start + step]],
                            mnist.train.labels[start:start + step], eps=0.2,
                            epochs=4,
                            batch_size=1, seed=2, filename='adv_Deepfool' + str(2), type='Deepfool',
                            wrong=True)

# Display
fig1, axes1 = plt.subplots(figsize=(5, 5), nrows=3, ncols=3, sharex=True, sharey=True, squeeze=False)
k = 0
for ax_row in axes1:
    for ax in ax_row:
        ax.imshow(np.squeeze(X_adv_2[k, :, :, :]), cmap='gray')
        label, _, _ = net2.predict(np.expand_dims(X_adv_2[k, :, :, :], 0))
        ax.set_title(f'{label[0]}')
        k = k + 1
plt.suptitle('Adversarial images')
plt.tight_layout()
plt.show()
