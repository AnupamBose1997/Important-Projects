import logging

import pandas as pd
from octaipipe.models.model_base import Model
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ExponentialSmoother(Model):
    '''Exponential smoothing time series model

    `Link to statsmodel docs <https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html>`__
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._solutiontype = 'timeseries'
        self._logger = logging.getLogger(__name__)

    def _build_new(self, **kwargs):
        '''Builds new model from statsmodel class'''
        if 'endog' in kwargs:
            self._logger.info('Instantiating ExponentialSmoothing model')
            self.estimator = ExponentialSmoothing(**kwargs)
        else:
            self._logger.info('Instantiating placeholder model')
            self.estimator = 'placeholder'

    def train(self, input_data, target_data, **kwargs):
        """Model training. Custom implementation as Prophet needs specific col
        names. If no date_col is given, assumes standard influx query is used,
        which sets '_time' as index. Resets index to get '_time' as col and
        sets date_col to replace that.

        Args:
            input_data (pd dataframe (for now)): data to use for training the
            model
            target_data: output data for the model training
        """
        self._logger.info(f'Training {self._name} model')

        if self.estimator == 'placeholder':
            self.estimator = ExponentialSmoothing(endog=target_data, **kwargs)

        self.estimator = self.estimator.fit()
        # assign new version and model id to the newly fitted model
        self._version = self._assign_version()
        self._id = self._assign_id()

    def predict(self,
                start,
                stop,
                onnx_pred: bool,
                **kwargs):
        """Model evaluation (prediction).

        Args:
            start: (int, str, datetime): When to start prediction
            stop: (int, str, datetime): When to stop prediction

        Returns:
            pd.Series: Series with predictions
        """
        if onnx_pred is True:
            raise ValueError('Model does not have ONNX implementation')
        else:
            y_pred = self.estimator.predict(start, stop)

        return pd.Series(data=y_pred, name='y_pred')
