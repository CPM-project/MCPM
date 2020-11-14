from MulensModel import Coordinates
from MulensModel import MulensData

from MCPM.pixellensingmodel import PixelLensingModel


class PixelLensingEvent(object):
    """
    Class that emulates MulensModel.Event for pixel lensing case
    """
    def __init__(self, datasets, model):
        if not isinstance(model, PixelLensingModel):
            raise TypeError('wrong input')
        self._model = model
        if isinstance(datasets, (list, tuple, MulensData)) or datasets is None:
            self._set_datasets(datasets)
        else:
            raise TypeError('incorrect argument datasets')

    def _set_datasets(self, new_value):
        """
        sets the value of self._datasets
        """
        self._datasets = new_value
        self._model.set_datasets(self._datasets)

    @property
    def datasets(self):
        """
        *list* of :py:class:`~MulensModel.mulensdata.MulensData`
        """
        if len(self._datasets) == 0:
            raise ValueError('No datasets were linked to the model')
        return self._datasets

    @property
    def model(self):
        """a PixelLensingModel instance"""
        return self._model
