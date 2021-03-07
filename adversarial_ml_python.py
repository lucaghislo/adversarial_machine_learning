import keras
import foolbox
import numpy as np
from keras.applications.resnet50 import ResNet50

keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)
image, label = foolbox.utils.imagenet_example()

np.argmax(fmodel.predictions(image[:, :, ::-1])) # Ritorna 282 (tiger cat)

attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(image[:, :, ::-1], label)

np.argmax(fmodel.predictions(adversarial[:, :, ::-1])) # Ritorna 287 (lynx, catamount)