"""Constants for Adversarial package.

   @author
     Victor I. Afolabi
     Artificial Intelligence Expert & Software Engineer.
     Email: javafolabi@gmail.com | victor.afolabi@zephyrtel.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: consts.py
     Created on 20 December, 2018 @ 07:03 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import os.path
from adversarials.core import Config

__all__ = [
  'FS', 'SETUP', 'LOGGER',
]

################################################################################################
# +--------------------------------------------------------------------------------------------+
# | FS: File System.
# +--------------------------------------------------------------------------------------------+
################################################################################################
class FS:
    # Project name & absolute directory.
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    APP_NAME = os.path.basename(PROJECT_DIR)

    ASSET_DIR = os.path.join(PROJECT_DIR, "assets")
    CACHE_DIR = os.path.join(ASSET_DIR, "cache")

    MODEL_DIR = os.path.join(CACHE_DIR, "models")


################################################################################################
# +--------------------------------------------------------------------------------------------+
# | Setup configuration constants.
# +--------------------------------------------------------------------------------------------+
################################################################################################
class SETUP:
    # Global setup configuration.
    __global = Config.from_cfg(os.path.join(FS.PROJECT_DIR, "adversarials/config/",
                                            "setup/global.cfg"))
    # Build mode/type.
    MODE = __global['config']['MODE']


################################################################################################
# +--------------------------------------------------------------------------------------------+
# | Logger: Logging configuration paths.
# +--------------------------------------------------------------------------------------------+
################################################################################################
class LOGGER:
    # Root Logger:
    ROOT = os.path.join(FS.PROJECT_DIR, 'adversarials/config/logger',
                        f'{SETUP.MODE}.cfg')
