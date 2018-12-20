"""Adversarial core package.

   @author
     Victor I. Afolabi
     Artificial Intelligence Expert & Software Engineer.
     Email: javafolabi@gmail.com | victor.afolabi@zephyrtel.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: __init__.py
     Created on 20 December, 2018 @ 06:56 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

from adversarials.core.config import Config
from adversarials.core.base import ModelBase
from adversarials.core.arch import SimpleGAN
from adversarials.core.utils import Log, File
from adversarials.core.consts import FS, LOGGER, SETUP

__all__ = [
    'SimpleGAN',
    'Log', 'File',
    'ModelBase', 'Config',
    'FS', 'LOGGER', 'SETUP',
]
