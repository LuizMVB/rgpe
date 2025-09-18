from abc import ABC, abstractmethod


class Command(ABC):
    @abstractmethod
    def execute(self, *args) -> None:
        pass

    @classmethod
    def options(cls):
        pass
