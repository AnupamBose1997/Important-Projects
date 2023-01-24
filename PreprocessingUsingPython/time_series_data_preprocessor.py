
import logging
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesDataPreprocessor:
    '''
    Class to apply standard preprocessing techniques to time-series data.

    Args:
        steps (list): a list of preprocessing steps to take in order.
        initial_RUL (int): initial RUL for building output target set with
                           linear degradation model. That is, the system is
                           considered healthy during an initial period, in
                           which remaining useful life is constant
                           (initial_RUL) until a point at which the system
                           begins to degrade linearly until failure.

    '''

    def __init__(self, steps, degradation_model, initial_RUL, **kwargs):
        self._logger = logging.getLogger(__name__)
        self._steps = steps
        self._logger.info(f'Preprocessing steps: {self._steps}')
        self._initial_RUL = initial_RUL
        self._degradation_model = degradation_model

    def _function_mapping(self, name):
        '''
        Maps preprocessing names to class methods.

        Args:
            name (string): a preprocessing technique.

        Returns:
            mapping[name] (method): the method corresponding to name.

        '''
        mapping = {
            'interpolate_missing_data': self._interpolate_missing,
            'remove_constant_columns': self._remove_constant_columns,
            'normalise': self._normalise,
            'make_target': self._make_RUL_target,
            'make_test_targets': self._make_RUL_test_targets
            }
        return mapping[name]

    def _interpolate_missing(self, df):
        '''
        Interpolates missing values in data.

        The technique used to interpolate depends on the index type of the
        DataFrame. If the index is timestamps, the interpolation uses method
        'time' which takes the time between values into account. Otherwise,
        linear interpolation is used which assumes equally spaced values.

        Args:
            df (DataFrame): DataFrame that may have missing data.

        Returns:
            DataFrame: DataFrame with missing data interpolated.

        '''
        if df.index.inferred_type == 'datetime64':
            method = 'time'
        else:
            method = 'linear'
        return df.interpolate(method=method, axis=0, limit_direction='both')

    def _remove_constant_columns(self, df):
        '''
        Removes columns that are constant through time based on standard
        deviation.

        Computes the standard deviation of each column and round to 6 d.p.
        as some constant columns have tiny but non-zero standard devs.

        If column is not numeric, set to 1.0 to retain.

        Args:
            df (DataFrame): the original data.

        Returns:
            df (DataFrame): the data with constant columns removed.

        '''
        std_devs = [
            df[col].std().round(6)
            if is_numeric_dtype(df[col]) else 1.0
            for col in df.columns
            ]
        cols_to_remove = list(df.columns.to_numpy()[std_devs == 0.])
        df.drop(cols_to_remove, axis=1, inplace=True)
        return df

    def _normalise(self, df):
        '''
        Min-max scale the columns of dataset.

        Scales each column to the range [0,1] using sklearns MinMaxScaler,
        and converts returned ndarray into a DataFrame.

        Args:
            df (DataFrame): data to be scaled.

        Returns:
            scaled_df (DataFrame): scaled data.

        '''
        self.scaler = MinMaxScaler()
        self.scaler.fit(df)
        scaled_df = pd.DataFrame(
            data=self.scaler.fit_transform(df),
            columns=df.columns,
            index=df.index
            )
        return scaled_df

    def _make_pw_linear_degradation_targets(self, df, remaining_life=0):
        '''
        Makes RUL targets with piece-wise linear degradation model using list
        comprehension.

        Args:
            df (pd.DataFrame): data used for RUL estimation.
            remaining_life (int): the actual remaining life for test data
                that ends before failure.

        Returns:
            targets (list): contains the RUL at each timestep.

        '''
        # compute start of failure based on degradation model
        failure_time = len(df) + remaining_life
        start_of_failure = failure_time - self._initial_RUL

        targets = [self._initial_RUL
                   if 0 <= timestep <= start_of_failure
                   else failure_time - timestep
                   for timestep in range(failure_time)]
        return targets

    def _make_linear_degradation_targets(self, df, remaining_life=0):
        '''
        Makes RUL targets with linear degradation model.

        Args:
            df (pd.DataFrame): data used for RUL estimation.
            remaining_life (int): the actual remaining life for test data
                that ends before failure.
        '''
        targets = list(range(len(df)+remaining_life, 0, -1))
        return targets

    def _make_RUL_target(self, df):
        '''
        Generates a target column for RUL based on piece-wise linear
        degradation model.

        For more info, see class description docstring.

        Args:
            df (DataFrame): dataframe to be used for RUL estimation.

        Returns:
            df (DataFrame): dataset with RUL as a column.

        '''
        if self._degradation_model == 'linear':
            targets = self._make_linear_degradation_targets(df)

        elif self._degradation_model == 'pw_linear':
            targets = self._make_pw_linear_degradation_targets(df)

        else:
            raise KeyError(
                f'Degradation model {self._degradation_model} not valid.'
                )

        # add as a column of dataframe
        df['RUL'] = targets
        return df

    def _make_RUL_test_targets(self, df, y_test):
        '''
        Generates RUL targets for a test set containing a full input test set,
        but only one output value.

        Currently this is specific to the NASA turbofan dataset, where the test
        data ends prior to failure, and the goal is to predict RUL at the time
        of the final row.

        To visualise our RUL predictions over time, we also want the true RUL
        values across the whole input test data. To get these, we assume the
        same piece-wise linear degradation model as when generating RUL for
        the training set. Knowing the initial RUL and the truth RUL at a
        single point we can work backwards to generate the other RUL values.

        See the RUL_evaluation_methods notebook for a visual example.

        Args:
            df (pd.DataFrame): testing input data.
            y_test (pd.Series): a single RUL test output.

        Returns:
            y_test_all (pd.Series): the RUL outputs corresponding to all
                                    inputs.

        '''
        if self._degradation_model == 'linear':
            targets = self._make_linear_degradation_targets(
                df,
                remaining_life=y_test
                )

        elif self._degradation_model == 'pw_linear':
            targets = self._make_pw_linear_degradation_targets(
                df,
                remaining_life=y_test
                )

        # make targets the right length
        targets = targets[0:len(df)]

        # convert to pd.Series, using timestamps index from df
        y_test_all = pd.Series(data=targets, index=df.index, name='y_test')

        return y_test_all

    def preprocess(self, df):
        '''
        Carry out all preprocessing steps on data.

        Args:
            df (DataFrame): raw data.

        Returns:
            df (DataFrame): preprocessed data.

        '''

        for step in self._steps:
            self._logger.debug(f'On preprocessing step {step}')
            df = self._function_mapping(step)(df)

        self._logger.info(f'Data preprocessed, new shape: {df.shape}')
        self._logger.debug(f'Preview: {df.head()}, {df.tail()}')

        return df
