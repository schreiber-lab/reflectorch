from time import perf_counter
from contextlib import contextmanager


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


@contextmanager
def print_time(name: str):
    start = perf_counter()
    yield
    print(f"Time for {name} = {(perf_counter() - start):.2f} sec")
