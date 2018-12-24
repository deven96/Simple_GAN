# Simple GAN

This is my attempt to make a wrapper class for a GAN in keras which can be used to abstract the whole architecture process.

[![Build Status](https://travis-ci.org/deven96/Simple_GAN.svg?branch=master)](https://travis-ci.com/deven96/Simple_GAN)[![PyPI version](https://badge.fury.io/py/Adversarials.svg)](https://badge.fury.io/py/Adversarials)![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=deven96_Simple_GAN&metric=alert_status)

- [Simple GAN](#simple-gan)
  - [Overview](#overview)
  - [Flow Chart](#flow-chart)
  - [Installation](#installation)
  - [Example](#example)
  - [Documentation](#documentation)
  - [Credits](#credits)
  - [Contribution](#contribution)
  - [License (MIT)](#license-mit)
  - [Todo](#todo)

## Overview

![alt text](assets/mnist_gan.png "GAN network using the MNIST dataset")

## Flow Chart

Setting up a Generative Adversarial Network involves having a discriminator and a generator working in tandem, with the ultimate goal being that the generator can come up with samples that are indistinguishable from valid samples by the discriminator.

![alt text](assets/flow.jpg "High level flowchart")

## Installation

```bash
    pip install adversarials
```

## Example 

```python
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

    gan = SimpleGAN(save_to_dir="./assets/images",
    save_interval=20)
    gan.train(X_train, epochs=40)
```

## Documentation

[Github Pages](https://deven96.github.io/Simple_GAN)

## Credits

- [Understanding Generative Adversarial Networks](https://towardsdatascience.com/understanding-generative-adversarial-networks-4dafc963f2ef) - Noaki Shibuya
- [Github Keras Gan](https://github.com/osh/KerasGAN)
- [Simple gan](https://github.com/daymos/simple_keras_GAN/blob/master/gan.py)

## Contribution

You are very welcome to modify and use them in your own projects.

Please keep a link to the [original repository](https://github.com/deven96/Simple_GAN). If you have made a fork with substantial modifications that you feel may be useful, then please [open a new issue on GitHub](https://github.com/deven96/Simple_GAN/issues) with a link and short description.

## License (MIT)

This project is opened under the [MIT 2.0 License](https://github.com/deven96/Simple_GAN/blob/master/LICENSE) which allows very broad use for both academic and commercial purposes.

A few of the images used for demonstration purposes may be under copyright. These images are included under the "fair usage" laws.

## Todo

- Add view training(discriminator and generator) simultaneously using tensorboard
- Provision for Parallel data processing and multithreading
- Saving models to Protobuff files
- Using TfGraphDef and other things that could speed up training and inference