import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from create_adv import make_adversarials
from cnn_copy import MnistCNN
import sys

sys.path.append('../Thesis_Utilities/')
from utilities import load_datasets

seed = 42

# Load data
x_train, y_train, x_val, y_val, x_test, y_test = load_datasets(test_size=100, val_size=5000, omniglot_bool=True,
                                                               name_data_set='../data/data' + str(seed) + '.h5',
                                                               force=False,
                                                               create_file=True, r_seed=seed)
# Create model
np.random.seed(42)
tf.reset_default_graph()
sess = tf.Session()
net2 = MnistCNN(sess, save_dir='../Thesis_CNN_mnist/Mnist_save/')
tf.reset_default_graph()
sess = tf.Session()

# Try to create 500 adversarials of each type
index = np.arange(len(x_train))
start = 0
step = 30

# JSMA adversarial
X_adv_1 = make_adversarials(sess, net2, x_train[index[start:start + step]], y_train[start:start + step], eps=0.2,
                            epochs=120,
                            batch_size=1,
                            seed=seed, filename='../data/adv_JSMA' + str(seed), type='JSMA', wrong=False)
# FGSM adversarial
start = start + step
X_adv_2 = make_adversarials(sess, net2, x_train[index[start:start + step]], y_train[start:start + step], eps=0.02,
                            epochs=16,
                            batch_size=1, seed=seed, filename='../data/adv_FGSM' + str(seed), type='FGSM', wrong=True)
# Deepfool adversarial
start = start + step
X_adv_3 = make_adversarials(sess, net2, x_train[index[start:start + step]], y_train[start:start + step], eps=0.2,
                            epochs=4,
                            batch_size=1, seed=seed, filename='../data/adv_Deepfool' + str(seed), type='Deepfool',
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
