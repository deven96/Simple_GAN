"""Architecture of SimpleGAN for Adversarial package.
   @author
     Domnan Diretnan
     Artificial Intelligence Enthusiast & Python Developer
     Email: diretnandomnan@gmail.com 
     GitHub: https://github.com/deven96
   @project
     File: arch.py
   @license
     MIT License
     Copyright (c) 2018. Domnan Diretnan. All rights reserved.
"""
import os
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from adversarials.core.base import ModelBase
from adversarials.core.utils import FS, File, Log
from keras.datasets import mnist
from keras.layers import (BatchNormalization, Dense, Dropout, Flatten, Input,
                          Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

plt.switch_backend('agg')   # allows code to run without a system DISPLAY


class SimpleGAN(ModelBase):
    """Simple Generative Adversarial Network.

    Methods:
        def __init__(self, size: Union[Tuple[int], int]=28, channels: int=1, batch_size: int=32, **kwargs)

        def train(self, X_train, epochs: int=10):

        def plot_images(self, samples: int=16, step:int=0):
            # Plot generated images

    Attributes:
        G (keras.model.Model): Generator model.
        D (keras.model.Model): Discriminator model.
        model (keras.model.Model): Combined G & D model.
        shape (Tuple[int]): Input image shape.
    """

    def __init__(self, size: Union[Tuple[int], int]=28, channels: int=1, batch_size: int=32, **kwargs):
        """def __init__(size: Union[Tuple[int], int]=28, channels: int=1, batch_size: int=32, **kwargs)

        Args:
            size (int, optional): Defaults to 28. Image size. int tuple
                consisting of image width and height or int in which case
                width and height are uniformly distributed.
            channels (int, optional): Defaults to 1. Image channel.
                1 - grayscale, 3 - colored.
            batch_size (int, optional): Defaults to 32. Mini-batch size.

        Keyword Args:
            optimizer (keras.optimizer.Optimizer, optional): Defaults to Adam
                with learning rate of 1e-3, beta_1=0.5, decay = 8e-8.
            save_interval (int, optional): Defaults to 100. Interval of training on
                which to save generated images.
            save_to_dir (Union[bool, str], optional): Defaults to False. Save generated images
                to directory.
            save_model (str, optional): Defaults to ./assets/models/SimpleGAN/model.h5. 
                save generated images to directory.

        Raises:
            TypeError: Expected one of int, tuple - Got `type(size)`.
        """
        super(SimpleGAN, self).__init__(**kwargs)

        if isinstance(size, tuple):
            self.width, self.height = size
        elif isinstance(size, int):
            self.width, self.height = size, size
        else:
            raise TypeError('Expected one of int, tuple. Got {}'
                            .format(type(size)))

        self.channels = channels
        self.batch_size = batch_size

        # Extract keyword arguments.
        self.optimizer = kwargs.get('optimizer',
                                    Adam(lr=0.0002, beta_1=0.5, decay=8e-8))
        self.save_to_dir = kwargs.get ('save_to_dir', False)
        self.save_interval = kwargs.get('save_interval', 100)
        self.save_model = kwargs.get('save_model', FS.MODEL_DIR+"/SimpleGAN/model.h5")

        self._log('Compiling generator.', level='info')
        # Generator Network.
        self.G = self.__generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self._log('Compiling discriminator.', level='info')
        # Discriminator Network.
        self.D = self.__discriminator()
        self.D.compile(loss='binary_crossentropy',
                       optimizer=self.optimizer,
                       metrics=['accuracy'])

        # Stacked model.
        self._log('Combining generator & Discriminator', level='info')
        self._model = self.__stacked_generator_discriminator()
        self._model.compile(loss='binary_crossentropy',
                            optimizer=self.optimizer)

    def train(self, X_train, epochs: int=10):
        """
            Train function to be used after GAN initialization

            X_train[np.array]: full set of images to be used
        """

        half_batch = self.batch_size // 2
        if self.save_to_dir:
            File.make_dirs(self.save_to_dir, verbose=1)
        for cnt in range(epochs):
            # get legits and syntethic images to be used in training discriminator
            random_index = np.random.randint(0,
                                             len(X_train) - half_batch)
            self._log('Random Index: {}'.format(random_index))
            legit_images = X_train[random_index: random_index + half_batch]\
                .reshape(half_batch, self.width, self.height, self.channels)
            gen_noise = np.random.normal(0, 1, (half_batch, 100))
            syntetic_images = self.G.predict(gen_noise)

            # combine synthetics and legits and assign labels
            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((half_batch, 1)),
                                               np.zeros((half_batch, 1))))

            d_loss = self.D.train_on_batch(x_combined_batch,
                                           y_combined_batch)

            # train generator (discriminator training is false by default)

            noise = np.random.normal(0, 1, (self.batch_size, 100))
            y_mislabled = np.ones((self.batch_size, 1))

            g_loss = self._model.train_on_batch(noise,
                                                y_mislabled)

            self._log(('Epoch: {:,}, [Discriminator :: d_loss: {:.4f}],'
                       '[Generator :: loss: {:.4f}]')
                      .format(cnt, d_loss[0], g_loss))

            if cnt % self.save_interval == 0:
                self.plot_images(step=cnt)
        if self.save_model:
            self._model.save(self.save_model)

    def call(self, n: int=1, dim: int=100):
        """Inference method. Given a random latent sample. Generate an image.

        Args:
            samples (int, optional): Defaults to 1. Number of images to
                be generated.
            dim (int, optional): Defaults to 100. Noise dimension.

        Returns:
            np.ndarray: Array-like generated images.
        """
        noise = np.random.normal(0, 1, size=(n, dim))
        return self.G.predict(noise)

    def plot_images(self, samples=16, step=0):
        """ Plot and generate images

            samples (int, optional): Defaults to 16. Noise samples to generate.
            step (int, optional): Defaults to 0. Number of training step currently.
        """
        filename = r"{0}/generated_{1}.png".format(self.save_to_dir,step)

        # Generate images.
        images = self.call(samples)

        plt.figure(figsize=(10, 10))

        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.height, self.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()

        if self.save_to_dir:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

    def __generator(self):
        """Generator sequential model architecture.

        Summary:
            Dense -> LeakyReLU -> BatchNormalization -> Dense ->
            LeakyReLU -> BatchNormalization -> Dense -> LeakyReLU ->
            BatchNormalization -> Dense -> Reshape

        Returns:
            keras.model.Model: Generator model.
        """

        model = Sequential()
        model.add(Dense(256, input_shape=(100,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.width * self.height *
                        self.channels, activation='tanh'))
        model.add(Reshape((self.width, self.height, self.channels)))

        return model

    def __discriminator(self):
        """Discriminator sequential model architecture.

        Summary:
            Flatten -> Dense -> LeakyReLU ->
            Dense -> LeakyReLU -> Dense

        Returns:
            keras.model.Model: Generator model.
        """

        model = Sequential()
        model.add(Flatten(input_shape=self.shape))
        model.add(Dense((self.width * self.height * self.channels),
                        input_shape=self.shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        return model

    def __stacked_generator_discriminator(self):

        self.D.trainable = False

        model = Sequential()
        model.add(self.G)
        model.add(self.D)

        return model

    @property
    def shape(self):
        """Input image shape.

        Returns:
            Tuple[int]: image shape.
        """

        return self.width, self.height, self.channels

    @property
    def model(self):
        """Stacked generator-discriminator model.

        Returns:
            keras.model.Model: Combined G & D model.
        """

        return self._model

    # @property
    # def G(self):
    #     """Generator model.

    #     Returns:
    #         keras.model.Model: Generator model.
    #     """

    #     return self.__generator()

    # @property
    # def D(self):
    #     """Discriminator model.

    #     Returns:
    #         keras.model.Model: Discriminator model.
    #     """

    #     return self.__discriminator()
