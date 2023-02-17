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
    def log(self, name: str, data):
        pass

    def __setitem__(self, key, value):
        self.log(key, value)


class Loggers(Logger):
    def __init__(self, *loggers):
        self._loggers = tuple(loggers)

    def log(self, name: str, data):
        for logger in self._loggers:
            logger.log(name, data)


class PrintLogger(Logger):
    def log(self, name: str, data):
        print(name, ': ', data)
