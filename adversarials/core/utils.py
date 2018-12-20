"""Utility file for Adversarial package.

   @author
     Victor I. Afolabi
     Artificial Intelligence Expert & Software Engineer.
     Email: javafolabi@gmail.com | victor.afolabi@zephyrtel.com
     GitHub: https://github.com/victor-iyiola

   @project
     File: utils.py
     Created on 20 December, 2018 @ 07:00 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import os
import sys
import logging

from abc import ABCMeta
from typing import Iterable
from logging.config import fileConfig

from adversarials.core import FS, LOGGER

__all__ = [
    'File', 'Log',
]

################################################################################################
# +--------------------------------------------------------------------------------------------+
# | File: File utility class for working with directories & files.
# +--------------------------------------------------------------------------------------------+
################################################################################################


class File(metaclass=ABCMeta):
    @staticmethod
    def make_dirs(path: str, verbose: int = 0):
        """Create Directory if it doesn't exist.

        Args:
            path (str): Directory/directories to be created.
            verbose (bool, optional): Defaults to 0. 0 turns of logging,
                while 1 gives feedback on creation of director(y|ies).

        Example:
            ```python
            >>> path = os.path.join("path/to", "be/created/")
            >>> File.make_dirs(path, verbose=1)
            INFO  |  "path/to/be/created/" has been created.
            ```
        """

        # if director(y|ies) doesn't already exist.
        if not os.path.isdir(path):
            # Create director(y|ies).
            os.makedirs(path)

            if verbose:
                # Feedback based on verbosity.
                Log.info('"{}" has been created.'.format(path))

    @staticmethod
    def get_dirs(path: str, exclude: Iterable[str] = None, optimize: bool = False):
        """Retrieve all directories in a given path.

        Args:
            path (str): Base directory of directories to retrieve.
            exclude (Iterable[str], optional): Defaults to None. List of paths to
                remove from results.
            optimize (bool, optional): Defaults to False. Return an generator object,
                to prevent loading all directories in memory, otherwise: return results
                as a normal list.

        Raises:
            FileNotFoundError: `path` was not found.

        Returns:
            Union[Generator[str], List[str]]: Generator expression if optimization is turned on,
                otherwise list of directories in given path.
        """
        # Return only list of directories.
        return File.listdir(path, exclude=exclude, dirs_only=True, optimize=optimize)

    @staticmethod
    def get_files(path: str, exclude: Iterable[str] = None, optimize: bool = False):
        """Retrieve all files in a given path.

        Args:
            path (str): Base directory of files to retrieve.
            exclude (Iterable[str], optional): Defaults to None. List of paths to
                remove from results.
            optimize (bool, optional): Defaults to False. Return an generator object,
                to prevent loading all directories in memory, otherwise: return results
                as a normal list.

        Raises:
            FileNotFoundError: `path` was not found.

        Returns:
            Union[Generator[str], List[str]]: Generator expression if optimization is turned on,
                otherwise list of files in given path.
        """
        # Return only list of directories.
        return File.listdir(path, exclude=exclude, files_only=True, optimize=optimize)

    @staticmethod
    def listdir(path: str, exclude: Iterable[str] = None,
                dirs_only: bool = False, files_only: bool = False,
                optimize: bool = False):
        """Retrieve files/directories in a given path.

        Args:
            path (str): Base directory of path to retrieve.
            exclude (Iterable[str], optional): Defaults to None. List of paths to
                remove from results.
            dirs_only (bool, optional): Defaults to False. Return only directories in `path`.
            files_only (bool, optional): Defaults to False. Return only files in `path`.
            optimize (bool, optional): Defaults to False. Return an generator object,
                to prevent loading all directories in memory, otherwise: return results
                as a normal list.

        Raises:
            FileNotFoundError: `path` was not found.

        Returns:
            Union[Generator[str], List[str]]: Generator expression if optimization is turned on,
                otherwise list of directories in given path.
        """
        if not os.path.isdir(path):
            raise FileNotFoundError('"{}" was not found!'.format(path))

        # Get all files in `path`.
        if files_only:
            paths = (os.path.join(path, p) for p in os.listdir(path)
                     if os.path.isfile(os.path.join(path, p)))
        else:
            # Get all directories in `path`.
            if dirs_only:
                paths = (os.path.join(path, p) for p in os.listdir(path)
                         if os.path.isdir(os.path.join(path, p)))
            else:
                # Get both files and directories.
                paths = (os.path.join(path, p) for p in os.listdir(path))

        # Exclude paths from results.
        if exclude is not None:
            # Remove excluded paths.
            paths = filter(lambda p: os.path.basename(p) not in exclude, paths)

        # Convert generator expression to list.
        if not optimize:
            paths = list(paths)

        return paths


################################################################################################
# +--------------------------------------------------------------------------------------------+
# | Log: For logging and printing download progress, etc...
# +--------------------------------------------------------------------------------------------+
################################################################################################
class Log(metaclass=ABCMeta):
    # File logger configuration.
    fileConfig(LOGGER.ROOT)
    _logger = logging.getLogger()

    # Log Level.
    level = _logger.level

    @staticmethod
    def setLevel(level: int):
        Log._logger.setLevel(level=level)

    @staticmethod
    def debug(*args, **kwargs):
        sep = kwargs.pop('sep', " ")
        Log._logger.debug(sep.join(map(repr, args)), **kwargs)

    @staticmethod
    def info(*args, **kwargs):
        sep = kwargs.pop('sep', " ")
        Log._logger.info(sep.join(map(repr, args)), **kwargs)

    @staticmethod
    def warn(*args, **kwargs):
        sep = kwargs.pop('sep', " ")
        Log._logger.warning(sep.join(map(repr, args)), **kwargs)

    @staticmethod
    def error(*args, **kwargs):
        sep = kwargs.pop('sep', " ")
        Log._logger.error(sep.join(map(repr, args)), **kwargs)

    @staticmethod
    def critical(*args, **kwargs):
        sep = kwargs.pop('sep', " ")
        Log._logger.critical(sep.join(map(repr, args)), **kwargs)

    @staticmethod
    def log(*args, **kwargs):
        """Logging method avatar based on verbosity.

        Args:
            *args

        Keyword Args:
            verbose (int, optional): Defaults to 1.
            level (int, optional): Defaults to ``Log.level``.
            sep (str, optional): Defaults to " ".

        Returns:
            None
        """

        # No logging if verbose is not 'on'.
        if not kwargs.pop('verbose', 1):
            return

        # Handle for callbacks & log level.
        sep = kwargs.pop('sep', " ")

        Log._logger.log(
            Log.level, sep.join(map(repr, args)), **kwargs)

    @staticmethod
    def progress(count: int, max_count: int):
        """Prints task progress *(in %)*.

        Args:
            count {int}: Current progress so far.
            max_count {int}: Total progress length.
        """

        # Percentage completion.
        pct_complete = count / max_count

        # Status-message. Note the \r which means the line should
        # overwrite itself.
        msg = "\r- Progress: {0:.02%}".format(pct_complete)

        # Print it.
        # Log.log(msg)
        sys.stdout.write(msg)
        sys.stdout.flush()

    @staticmethod
    def report_hook(block_no: int, read_size: bytes, file_size: bytes):
        """Calculates download progress given the block number, read size,
        and the total file size of the URL target.

        Args:
            block_no {int}: Current download state.
            read_size {bytes}: Current downloaded size.
            file_size {bytes}: Total file size.

        Returns:
            None.
        """
        # Calculates download progress given the block number, a read size,
        #  and the total file size of the URL target.
        pct_complete = float(block_no * read_size) / float(file_size)

        msg = "\r\t -Download progress {:.02%}".format(pct_complete)
        # Log.log(msg)
        sys.stdout.stdwrite(msg)
        sys.stdout.flush()
