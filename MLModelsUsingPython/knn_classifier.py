import logging

from octaipipe.models.model_base import Model
from skl2onnx import convert_sklearn
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier(Model):
    def __init__(self, **kwargs):
        '''kNN Classification class

        `Link to Sklearn docs <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`__
        '''
        super().__init__(**kwargs)
        self._solutiontype = 'classification'
        self._logger = logging.getLogger(__name__)

    def _build_new(self):
        self._logger.info('Instantiating kNN classification model')
        self.estimator = KNeighborsClassifier()

    def _to_onnx(self):
        '''
        Method to convert model to onnx for saving. The skl2onnx function for
        conversion is used and then the filename is changed in order to save
        the model as onnx.
        '''
        self.onnx_estimator = convert_sklearn(self.estimator,
                                              initial_types=self.data_schema)
