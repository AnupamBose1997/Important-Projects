import logging

from octaipipe.models.model_base import Model
from skl2onnx import convert_sklearn
from sklearn.linear_model import LogisticRegression


class LogisticRegressor(Model):
    def __init__(self, **kwargs):
        '''Logistic regression class for classification

        `Sklearn implementation can be found here <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`__
        '''
        super().__init__(**kwargs)
        self._solutiontype = 'classification'
        self._logger = logging.getLogger(__name__)

    def _build_new(self):
        self._logger.info('Instantiating logistic regression model')
        self.estimator = LogisticRegression()

    def _to_onnx(self):
        '''
        Method to convert model to onnx for saving. The skl2onnx function for
        conversion is used and then the filename is changed in order to save
        the model as onnx.
        '''
        self.onnx_estimator = convert_sklearn(self.estimator,
                                              initial_types=self.data_schema)
