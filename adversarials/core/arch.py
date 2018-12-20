""" Class implementations of different architectures of a
    Generative Adversarial Neural Network
"""
from typing import Union, Tuple

import numpy as np


from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, Reshape, Flatten, Dropout

import matplotlib.pyplot as plt
plt.switch_backend('agg')   # allows code to run without a system DISPLAY


class SimpleGAN:
    """ Generative Adversarial Network class """

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
            save_to_file (bool, optional): Defaults to False. Save generated images
                to file.
            save_interval (int, optional): Defaults to 100. Interval of training on
                which to save generated images.
        """

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
        self.save_to_file = kwargs.get('save_to_file', False)
        self.save_interval = kwargs.get('save_interval', 100)

        # Generator Network.
        self.G = self.__generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        # Discriminator Network.
        self.D = self.__discriminator()
        self.D.compile(loss='binary_crossentropy',
                       optimizer=self.optimizer, metrics=['accuracy'])

        # Stacked model.
        self._model = self.__stacked_generator_discriminator()
        self._model.compile(loss='binary_crossentropy',
                            optimizer=self.optimizer)

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
        model.add(Dense((self.width * self.height * self.channels)/2))
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

    def train(self, X_train, epochs: int=10):
        """
            Train function to be used after GAN initialization

            X_train[np.array]: full set of images to be used
        """

        for cnt in range(self.epochs):

            # get legits and syntethic images to be used in training discriminator
            random_index = np.random.randint(0, len(X_train) - self.batch/2)
            legit_images = X_train[random_index: random_index + self.batch /
                                   2].reshape(self.batch/2, self.width, self.height, self.channels)
            gen_noise = np.random.normal(0, 1, (self.batch/2, 100))
            syntetic_images = self.G.predict(gen_noise)

            # combine synthetics and legits and assign labels
            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate(
                (np.ones((self.batch/2, 1)), np.zeros((self.batch/2, 1))))

            d_loss = self.D.train_on_self.batch(
                x_combined_batch, y_combined_batch)

            # train generator (discriminator training is false by default)

            noise = np.random.normal(0, 1, (self.batch, 100))
            y_mislabled = np.ones((self.batch, 1))

            g_loss = self._model.train_on_batch(
                noise, y_mislabled)

            print('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (
                cnt, d_loss[0], g_loss))

            if cnt % self.save_interval == 0:
                self.plot_images(step=cnt)

    def plot_images(self, samples=16, step=0):
        """ Plot and generate images

            samples[int]: noise samples to generate
            step[int]: number of training step currently
        """
        filename = "./simple_gan/generated_%d.png" % step
        noise = np.random.normal(0, 1, (samples, 100))

        images = self.G.predict(noise)

        plt.figure(figsize=(10, 10))

        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.height, self.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()

        if self.savetofile:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

    @property
    def shape(self):
        return self.width, self.height, self.channels

    @property
    def model(self):
        return self._model


if __name__ == '__main__':
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    gan = SimpleGAN(epochs=1)
    gan.train(X_train)
