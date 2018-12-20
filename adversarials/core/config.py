"""Configuration file for Adversarial package.

   @author
     Victor I. Afolabi
     Artificial Intelligence Expert & Software Engineer.
     Email: javafolabi@gmail.com | victor.afolabi@zephyrtel.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: config.py.py
     Created on 20 December, 2018 @ 07:07 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
# Built-in libraries.
import os
import json
import pickle
import configparser

from abc import ABCMeta
from typing import Callable, Any

# Third party libraries.
import yaml
from easydict import EasyDict

# In order to use LibYAML bindings, which is much faster than pure Python.
# Download and install [LibYAML](https://pyyaml.org/wiki/LibYAML).
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# Exported classes & functions.
__all__ = [
    'Config',
]


################################################################################################
# +--------------------------------------------------------------------------------------------+
# | Config: Configuration avatar class to convert save & load config files.
# +--------------------------------------------------------------------------------------------+
################################################################################################
class Config(metaclass=ABCMeta):
    @staticmethod
    def from_yaml(file: str):
        """Load configuration from a YAML file.

        Args:
            file (str): A `.yml` or `.yaml` filename.

        Raises:
            AssertionError: File is not a YAML file.
            FileNotFoundError: `file` was not found.

        Returns:
            easydict.EasyDict: config dictionary object.
        """

        assert (file.endswith('yaml') or
                file.endswith('yml')), 'File is not a YAML file.'

        if not os.path.isfile(file):
            raise FileNotFoundError('{} was not found'.format(file))

        with open(file, mode="r") as f:
            cfg = EasyDict(yaml.load(f, Loader=Loader))

        return cfg

    @staticmethod
    def from_cfg(file: str, ext: str = 'cfg'):
        """Load configuration from an cfg file.

        Args:
            file (str): An cfg filename.
            ext (str, optional): Defaults to 'cfg'. Config file extension.

        Raises:
            AssertionError: File is not an `${ext}` file.
            FileNotFoundError: `file` was not found.

        Returns:
            easydict.EasyDict: config dictionary object.
        """

        assert file.endswith(ext), f'File is not a/an `{ext}` file.'

        if not os.path.isfile(file):
            raise FileNotFoundError('{} was not found'.format(file))

        cfg = configparser.ConfigParser(dict_type=EasyDict)
        cfg.read(file)

        return cfg

    @staticmethod
    def from_json(file: str):
        """Load configuration from a json file.

        Args:
            file (str): A JSON filename.

        Raises:
            AssertionError: File is not a JSON file.
            FileNotFoundError: `file` was not found.

        Returns:
            easydict.EasyDict: config dictionary object.
        """

        assert file.endswith('json'), 'File is not a `JSON` file.'

        if not os.path.isfile(file):
            raise FileNotFoundError('{} was not found'.format(file))

        with open(file, mode='r') as f:
            cfg = EasyDict(json.load(f))

        return cfg

    @staticmethod
    def to_yaml(cfg: EasyDict, file: str, **kwargs):
        """Save configuration object into a YAML file.

        Args:
            cfg (EasyDict): Configuration: as a dictionary instance.
            file (str): Path to write the configuration to.

        Keyword Args:
            Passed into `dumper`.

        Raises:
            AssertionError: `dumper` must be callable.
        """
        # Use LibYAML (which is much faster than pure Python) dumper.
        kwargs.setdefault('Dumper', Dumper)

        # Write to a YAML file.
        Config._to_file(cfg=cfg, file=file, dumper=yaml.dump, **kwargs)

    @staticmethod
    def to_json(cfg: EasyDict, file: str, **kwargs):
        """Save configuration object into a JSON file.

        Args:
            cfg (EasyDict): Configuration: as dictionary instance.
            file (str): Path to write the configuration to.

        Keyword Args:
            Passed into `dumper`.

        Raises:
            AssertionError: `dumper` must be callable.
        """

        # Write to a JSON file.
        Config._to_file(cfg=cfg, file=file, dumper=json.dump, **kwargs)

    @staticmethod
    def to_cfg(cfg: EasyDict, file: str, **kwargs):
        """Save configuration object into a cfg or ini file.

        Args:
            cfg (Any): Configuration: as dictionary instance.
            file (str): Path to write the configuration to.

        Keyword Args:
            Passed into `dumper`.
        """
        print(cfg, file, **kwargs)
        return NotImplemented

    @staticmethod
    def to_pickle(cfg: Any, file: str, **kwargs):
        """Save configuration object into a pickle file.

        Args:
            cfg (Any): Configuration: as dictionary instance.
            file (str): Path to write the configuration to.

        Keyword Args:
            Passed into `dumper`.

        Raises:
            AssertionError: `dumper` must be callable.
        """
        Config._to_file(cfg=cfg, file=file, dumper=pickle.dump, **kwargs)

    @staticmethod
    def _to_file(cfg: Any, file: str, dumper: Callable, **kwargs):
        """Save configuration object into a file as allowed by `dumper`.

        Args:
            cfg (Any): Configuration: as dictionary instance.
            file (str): Path to write the configuration to.
            dumper (Callable): Function/callable handler to save object to disk.

        Keyword Args:
            Passed into `dumper`.

        Raises:
            AssertionError: `dumper` must be callable.
        """

        assert callable(dumper), "`dumper` must be callable."

        # Create directory if it doesn't exist.
        # if director(y|ies) doesn't already exist.
        if not os.path.isdir(file):
            # Create director(y|ies).
            os.makedirs(file)

        # Write configuration to file.
        with open(file, mode="wb", encoding="utf-8") as f:
            dumper(cfg, f, **kwargs)
