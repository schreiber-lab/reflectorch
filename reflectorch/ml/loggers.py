# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.


__all__ = [
    'Logger',
    'Loggers',
    'PrintLogger',
]


class Logger(object):
    "Base class defining a common interface for logging"
    def log(self, name: str, data):
        pass

    def __setitem__(self, key, value):
        """Enable dictionary-style setting to log data."""
        self.log(key, value)


class Loggers(Logger):
    """Class for using multiple loggers"""
    def __init__(self, *loggers):
        self._loggers = tuple(loggers)

    def log(self, name: str, data):
        for logger in self._loggers:
            logger.log(name, data)


class PrintLogger(Logger):
    """Logger which prints to the console"""
    def log(self, name: str, data):
        print(name, ': ', data)
