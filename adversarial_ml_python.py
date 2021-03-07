import keras
import foolbox
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50

keras.backend.set_learning_phase(0)

model = tf.keras.applications.ResNet50(weights="imagenet")
preprocessing = dict(flip_axis=-1, mean=[103.939, 116.779, 123.68])
bounds = (0, 255)
fmodel = foolbox.TensorFlowModel(model, bounds=bounds, preprocessing=preprocessing)

image, label = foolbox.utils.samples(fmodel, dataset='imagenet', batchsize=16, data_format='channels_last', bounds=(0, 1))

attack = foolbox.attacks.LinfFastGradientAttack(random_start=False)

epsilons = np.linspace(0.0, 0.005, num=20)
raw, clipped, is_adv = attack(fmodel, image, label, epsilons=0.03)

robust_accuracy = 1 - is_adv.float().mean(axis=-1)
plt.plot(epsilons, robust_accuracy.numpy())

