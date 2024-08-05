from typing import Any

__all__ = [
    "ProcessData",
    "ProcessPipeline",
]


class ProcessData(object):
    def __add__(self, other):
        if isinstance(other, ProcessData):
            return ProcessPipeline(self, other)

    def apply(self, args: Any, context: dict = None):
        return args

    def __call__(self, args: Any, context: dict = None):
        return self.apply(args, context)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class ProcessPipeline(ProcessData):
    def __init__(self, *processes):
        self._processes = list(processes)

    def apply(self, args: Any, context: dict = None):
        for process in self._processes:
            args = process(args, context)
        return args

    def __add__(self, other):
        if isinstance(other, ProcessPipeline):
            return ProcessPipeline(*self._processes, *other._processes)
        elif isinstance(other, ProcessData):
            return ProcessPipeline(*self._processes, other)

    def __repr__(self):
        processes = ", ".join(repr(p) for p in self._processes)
        return f'ProcessPipeline({processes})'
