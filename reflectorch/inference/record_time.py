from time import perf_counter
from contextlib import contextmanager
from functools import wraps


class EvaluateTime(list):
    @contextmanager
    def __call__(self, name: str, *args, **kwargs):
        start = perf_counter()
        yield
        self.action(perf_counter() - start, name, *args, **kwargs)

    @staticmethod
    def action(delta_time, name, *args, **kwargs):
        print(f"Time for {name} = {delta_time:.2f} sec")

    def __repr__(self):
        return f'EvaluateTime(total={sum(self)}, num_records={len(self)})'


def print_time(name: str or callable):
    if isinstance(name, str):
        return _print_time_context(name)
    else:
        return _print_time_wrap(name)


def _print_time_wrap(func, name: str = None):
    name = name or func.__name__

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        with _print_time_context(name):
            return func(*args, **kwargs)

    return wrapped_func


@contextmanager
def _print_time_context(name: str):
    start = perf_counter()
    yield
    print(f"Time for {name} = {(perf_counter() - start):.2f} sec")
