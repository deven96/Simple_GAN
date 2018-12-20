""" Class implementations of different architectures of a
    Generative Adversarial Neural Network 
"""

import numpy as np


from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

import matplotlib.pyplot as plt
plt.switch_backend('agg')   # allows code to run without a system DISPLAY


class SimpleGAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, width=28, height=28, channels=1, epochs=20000, 
                savetofile=False, optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8),
                batch = 32, save_interval = 100):
        """
            Initializes class
            width[int]: width
            height[int]: height
            channels[int]: number of channels in image
            epochs[int]: number of epochs to train on
            savetofile[bool]: save generated images to file
            optimizer[keras.optimizer]: valid keras optimizer to use
            batch[int]: batch to group input data
            save_interval[int]: interval of training on which to save generated image
        """

        self.width = width
        self.height = height
        self.channels = channels
        self.epochs = epochs
        self.savetofile = savetofile
        self.optimizer = optimizer
        self.batch = batch
        self.save_interval = save_interval

        self.shape = (self.width, self.height, self.channels)

        self.G = self.__generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.D = self.__discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()

        self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)


    def __generator(self):
        """ Declare generator """

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
        model.add(Dense(self.width  * self.height * self.channels, activation='tanh'))
        model.add(Reshape((self.width, self.height, self.channels)))

        return model

    def __discriminator(self):
        """ Declare discriminator """

        model = Sequential()
        model.add(Flatten(input_shape=self.shape))
        model.add(Dense((self.width * self.height * self.channels), input_shape=self.shape))
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

    def train(self, X_train):
        """
            Train function to be used after GAN initialization

            X_train[np.array]: full set of images to be used
        """

        for cnt in range(self.epochs):

            ## get legits and syntethic images to be used in training discriminator
            random_index = np.random.randint(0, len(X_train) - self.batch/2)
            legit_images = X_train[random_index : random_index + self.batch/2].reshape(self.batch/2, self.width, self.height, self.channels)
            gen_noise = np.random.normal(0, 1, (self.batch/2, 100))
            syntetic_images = self.G.predict(gen_noise)
            
            #combine synthetics and legits and assign labels
            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((self.batch/2, 1)), np.zeros((self.batch/2, 1))))

            d_loss = self.D.train_on_self.batch(x_combined_batch, y_combined_batch)


            # train generator (discriminator training is false by default)

            noise = np.random.normal(0, 1, (self.batch, 100))
            y_mislabled = np.ones((self.batch, 1))

            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))

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


if __name__ == '__main__':
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)


    gan = SimpleGAN(epochs=1)
    gan.train(X_train)