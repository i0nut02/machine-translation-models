from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod

    @abstractmethod
    def train(self):
        """
        Trains the model on the dataset.
        """
        pass

    @abstractmethod
    def translate(self, source_sentence):
        """
        Translates a source sentence to the target language.
        """
        pass