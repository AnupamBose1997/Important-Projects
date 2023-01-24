import logging

import lightgbm as lgb
from mlprodict.onnx_conv import guess_schema_from_data
from octaipipe.models.model_base import Model
from onnxmltools import convert_lightgbm


class LightGBMRegressor(Model):
    '''
    LightGBM regression class.

    `Link to LGBM docs <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html>`__
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._solutiontype = 'regression'
        self._logger = logging.getLogger(__name__)

    def _build_new(self):
        self._logger.info("Instantiating LightGBM model")
        self.estimator = lgb.LGBMRegressor()

    def _to_onnx(self):
        '''
        Method to convert model to onnx for saving
        '''
        self.onnx_estimator = convert_lightgbm(self.estimator,
                                               initial_types=self.data_schema)

    def train(self, input_data, target_data):
        """
        Trains or re-trains regressor.
        Uses __sklearn_is_fitted__() to check if estimator has already
        been trained.

        Args:
            input_data: Input features to train model with
            target_data: Target data to train model with

        Returns:
            None.
        """

        self._logger.info(f"Training {self._name} model")

        self.data_schema = guess_schema_from_data(input_data)

        if self.estimator.__sklearn_is_fitted__():
            # check data/model compatibility
            # to-do

            # Train model, initializing with self.estimator
            self.estimator = self.estimator.fit(
                X=input_data, y=target_data, init_model=self.estimator
            )
        else:
            self.estimator = self.estimator.fit(X=input_data, y=target_data)

        # assign new version and model id to the newly fitted model
        self._version = self._assign_version()
        self._id = self._assign_id()
