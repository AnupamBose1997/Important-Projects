import logging

from octaipipe.models.model_base import Model
from skl2onnx import convert_sklearn
from sklearn.linear_model import BayesianRidge


class BayesRidgeRegression(Model):
    '''
    Bayesian Ridge regression class.
    `Link to sklearn docs <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html>`__
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._solutiontype = 'regression'
        self._logger = logging.getLogger(__name__)

    def _build_new(self):
        '''
        Builds a Bayes Ridge Regressor estimator with sklearns BayesianRidge
        class.

        Returns:
            None.

        '''
        self.estimator = BayesianRidge()
        self._logger.info(f'{self.estimator.__class__.__name__} model built')

    def _to_onnx(self):
        '''
        Method to convert model to onnx for saving. The skl2onnx function for
        conversion is used and then the filename is changed in order to save
        the model as onnx.
        '''
        self.onnx_estimator = convert_sklearn(self.estimator,
                                              initial_types=self.data_schema)
