import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from cnn_copy import MnistCNN
from create_adv import make_adversarials

# Load data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)

# Create model
tf.reset_default_graph()
sess = tf.Session()
net2 = MnistCNN(sess, save_dir='../Thesis_CNN_mnist/Mnist_save/')
tf.reset_default_graph()
sess = tf.Session()
s = 1337

# Create adversarials
X_adv = make_adversarials(sess, net2, mnist.train.images[0:20], mnist.train.labels[0:20], eps=0.4, epochs=200, batch_size=1,
                          seed=s, filename='adv_'+str(s), type='JSMA')

# Display
fig1, axes1 = plt.subplots(figsize=(5, 5), nrows=3, ncols=3, sharex=True, sharey=True, squeeze=False)
k = 0
for ax_row in axes1:
    for ax in ax_row:
        ax.imshow(np.squeeze(X_adv[k,:,:,:]), cmap='gray')
        label,_,_ = net2.predict(np.expand_dims(X_adv[k,:,:,:],0))
        ax.set_title(f'{label[0]}')
        k = k+1
plt.suptitle('Adversarial images')
plt.tight_layout()
plt.show()