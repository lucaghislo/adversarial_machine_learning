import tensorflow.compat.v1  as tf
tf.disable_v2_behavior() 

images = tf.placeholder(tf.float32, (None, 224, 224, 3))
preprocessed = vgg_preprocessing(images)
logits = vgg19(preprocessed)

from foolbox.models import TensorFlowModel

model = TensorFlowModel(images, logits, bounds=(0, 255))

from foolbox.criteria import TargetClassProbability

target_class = 22
criterion = TargetClassProbability(target_class, p=0.99)

from foolbox.attacks import LBFGSAttack

attack = LBFGSAttack(model, criterion)
images, labels = foolbox.utils.samples(dataset='imagenet', batchsize=16, data_format='channels_last', bounds=(0, 255))
adversarial = attack(image, label=label)

import matplotlib.pyplot as plt

plt.subplot(1, 3, 1)
plt.imshow(image)

plt.subplot(1, 3, 2)
plt.imshow(adversarial)

plt.subplot(1, 3, 3)
plt.imshow(adversarial - image)
