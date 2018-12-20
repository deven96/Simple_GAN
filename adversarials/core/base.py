"""Base classes for Adversarial package.

   @author
     Victor I. Afolabi
     Artificial Intelligence Expert & Software Engineer.
     Email: javafolabi@gmail.com | victor.afolabi@zephyrtel.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: base.py
     Created on 20 December, 2018 @ 06:56 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import os.path

from abc import ABCMeta, abstractmethod
from adversarials.core import File, FS

class _Base(object):
    def __init__(self, *args, **kwargs):
        # Verbosity level: 0 or 1.
        self._verbose = kwargs.setdefault('verbose', 1)
        self._name = kwargs.get('name', self.__class__.__name__)

    def __repr__(self):
        """Object representation of Sub-classes."""
        # args = self._get_args()
        kwargs = self._get_kwargs()

        # Format arguments.
        # fmt = ", ".join(map(repr, args))
        fmt = ""

        # Format keyword arguments.
        for k, v in kwargs:
            # Don't include these in print-out.
            if k in ('captions', 'filename', 'ids'):
                continue
            fmt += ", {}={!r}".format(k, v)

        # Object representation of Class.
        return '{}({})'.format(self.__class__.__name__, fmt.lstrip(', '))

    def __str__(self):
        """String representation of Sub-classes."""
        return "{}()".format(self.__class__.__name__,
                             ", ".join(map(str, self._get_args())))

    def __format__(self, format_spec):
        if format_spec == "!r":
            return self.__repr__()
        return self.__str__()

    def _log(self, *args, level: str = 'log', **kwargs):
        """Logging method helper based on verbosity."""
        # No logging if verbose is not 'on'.
        if not kwargs.pop('verbose', self._verbose):
            return

        # Validate log levels.
        _levels = ('log', 'debug', 'info', 'warn', 'error', 'critical')
        if level.lower() not in _levels:
            raise ValueError("`level` must be one of {}".format(_levels))

        # Call the appropriate log level, eg: LogUtils.info(*args, **kwargs)
        eval(f'LogUtils.{level.lower()}(*args, **kwargs)')

    # noinspection PyMethodMayBeStatic
    def _get_args(self):
        # names = ('data_dir', 'sub_dirs', 'results_dir')
        # return [getattr(self, f'_{name}') for name in names]
        return []

    def _get_kwargs(self):
        # names = ('overwrite', 'sub_dirs', 'verbose', 'version')
        # return [(name, getattr(self, f'_{name}')) for name in names]
        return sorted([(k.lstrip('_'), getattr(self, f'{k}'))
                       for k in self.__dict__.keys()])

    @property
    def name(self):
        return self._name

    @property
    def verbose(self):
        return self._verbose


class ModelBase(_Base, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super(ModelBase, self).__init__(*args, **kwargs)

        # Extract Keyword arguments.
        self._cache_dir = kwargs.get('cache_dir',
                                     os.path.join(FS.ModelSavedDir.base, self._name))
        # Create cache directory if it doesn't exist.
        File.make_dirs(self._cache_dir, verbose=self._verbose)

    def __call__(self, *args, **kwargs):
        return self.call()

    # noinspection PyUnusedLocal
    @abstractmethod
    def call(self, *args, **kwargs):
        return NotImplemented

    @staticmethod
    def int_shape(x):
        """Returns the shape of tensor or variable as tuple of int or None entries.

        Args:
            x (Union[tf.Tensor, tf.Variable]): Tensor or variable. hasattr(x, 'shape')

        Returns:
            tuple: A tuple of integers (or None entries).
        """
        try:
            shape = x.shape
            if not isinstance(shape, tuple):
                shape = tuple(shape.as_list())
            return shape
        except ValueError:
            return None

    @property
    def cache_dir(self):
        return self._cache_dir
