# Simple GAN

This is my attempt to make a wrapper class for a GAN in keras which can be used to abstract the whole architecture process.

## Overview

![alt text](assets/mnist_gan.png "GAN network using the MNIST dataset")

## Flow Chart

Setting up a Generative Adversarial Network involves having a discriminator and a generator working in tandem, with the ultimate goal being that the generator can come up with samples that are indistinguishable from valid samples by the discriminator.

![alt text](assets/flow.jpg "High level flowchart")

## References

- [Understanding Generative Adversarial Networks](https://towardsdatascience.com/understanding-generative-adversarial-networks-4dafc963f2ef) - Noaki Shibuya
- [Github Keras Gan](https://github.com/osh/KerasGAN)
- [Simple gan](https://github.com/daymos/simple_keras_GAN/blob/master/gan.py)
