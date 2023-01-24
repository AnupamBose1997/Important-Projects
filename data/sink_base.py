from abc import ABC, abstractmethod
import logging


class Sink(ABC):

    def __init__(self, output_collector):
        self._logger = logging.getLogger(__name__)
        self._out_coll = output_collector

    @abstractmethod
    def write(self):
        """Abstract method implemented to write to database
        """
        pass
