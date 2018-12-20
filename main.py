"""Quick test for SimpleGAN.

   @author
     Victor I. Afolabi
     Artificial Intelligence Expert & Software Engineer.
     Email: javafolabi@gmail.com | victor.afolabi@zephyrtel.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: main.py
     Created on 20 December, 2018 @ 07:03 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import numpy as np
from keras.datasets import mnist

from adversarials.core import Log
from adversarials import SimpleGAN

if __name__ == '__main__':
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    Log.info('X_train.shape = {}'.format(X_train.shape))

    gan = SimpleGAN()
    gan.train(X_train, epochs=1)
