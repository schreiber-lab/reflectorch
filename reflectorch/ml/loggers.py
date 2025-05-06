from torch.utils.tensorboard import SummaryWriter

__all__ = [
    'Logger',
    'Loggers',
    'PrintLogger',
    'TensorBoardLogger',
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

class TensorBoardLogger(Logger):
    def __init__(self, log_dir: str):
        """
        Args:
            log_dir (str): Directory where TensorBoard logs will be written
        """
        super().__init__()
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 1
        
    def log(self, name: str, data):
        """Log scalar data to TensorBoard
        
        Args:
            name (str): Name/tag for the data
            data: Scalar value to log
        """
        if hasattr(data, 'item'):
            data = data.item()
        self.writer.add_scalar(name, data, self.step)
        self.step += 1