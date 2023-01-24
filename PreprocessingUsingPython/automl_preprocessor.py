import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


class AutoMLPreprocessor:
    def __init__(self, specs: dict):
        self._logger = logging.getLogger(__name__)
        self._specs = specs
        self._cycle_field = self._specs['cycle_field']
        self._target = self._specs.pop('target_label')

        # Order functions to custom order and add train_test_split
        self._specs['steps'].append('train_test_split')
        if (
            self._specs['solutiontype'] == 'classification' and
            'make_target' in self._specs['steps']
        ):
            self._specs['steps'].append('rul_class')
        self._specs['steps'] = sorted(self._specs['steps'],
                                      key=lambda x: self.function_order[x])

    def preprocess(self, data):
        '''Carries out all the preprocessing steps listed in the steps field
        in configs. This goes through steps in order, so the first step will
        be carried out first.

        Args:
            data (pd.DataFrame): data to preprocess.

        Returns:
            data_copy (pd.DataFrame): preprocessed copy of data frame
        '''
        data_copy = data.copy()
        steps = self._specs['steps']

        for step in steps:
            self._logger.info(f'AutoML preprocessing at step: {step}')
            data_copy = self._function_mapping(step)(data_copy)

        return data_copy

    def _add_remaining_useful_life(self, data):
        '''Receives DataFrame and computes RUL target column. If cycle_field is
        given as string, computes RUL grouped by this field. If solutiontype
        is classification, makes RUL degradation classes rather than continuous
        variable.

        Args:
            data (pd.DataFrame): Input data to compute target for

        Returns:
            data_copy (pd.DataFrane): data with 'RUL' in PW linear fashion
        '''
        self._logger.info('Making RUL target')
        data_copy = data.copy()

        def calc_rul(df):
            return list(range(len(df), 0, -1))

        if self._cycle_field is not None:
            rul = data_copy.groupby(self._cycle_field).\
                transform(calc_rul).iloc[:, 0].values
        else:
            rul = calc_rul(data_copy)

        data_copy['RUL'] = rul

        if self._specs['initial_RUL'] is not None:
            upper = self._specs['initial_RUL']
        else:
            upper = self._find_rul_clip(data_copy)

        data_copy['RUL'] = data_copy['RUL'].clip(
            upper=upper
        )

        return data_copy

    def _find_rul_clip(self, data):
        '''Method that takes data frame and determined where the clipping level
        for the RUL target should be. This uses the average between the
        shortest cycle and half of the mean cycle length. This decision is
        based on work by Heimes (2008).

        Args:
            data (pd.DataFrame): data frame with RUL output

        Returns:
            upper (int): clipping level of RUL for data frame
        '''
        if self._cycle_field is not None:
            min_len = data.groupby(self._cycle_field)['RUL'].count().min()
            avg_half = data.groupby(self._cycle_field)['RUL'].count().mean()/2
            upper = int(np.mean([min_len, avg_half]))
        else:
            upper = int(data.shape[0]*0.75)

        # Save if needed for classification threshold
        self._specs['initial_RUL'] = upper

        return upper

    def _add_degradation_class(self, data):
        '''Adds categorical RUL target with 2 or 3 classes, specifying
        boundaries based on the thresholds from self._specs['rul_class_thresh']

        Args:
            data (pd.DataFrame): data with RUL output variable

        Returns:
            data_copy (pd.DataFrame): same data frame with RUL class target
        '''
        self._logger.info('Making categorical RUL target')
        data_copy = data.copy()
        n_class = int(self._specs['rul_classes'])
        # Ensure number of thresholds is rul_classes-1
        self._logger.info('Turning RUL target into categorical variable')
        if n_class == 3:
            thresholds = [self.RUL_clip/2, self.RUL_clip]
        elif n_class == 2:
            thresholds = [self.RUL_clip]
        else:
            raise ValueError(f'rul_classes cannot be {n_class}. Takes 2 or 3.')

        # Sort to have lowest threshold first and set classes
        thresholds.sort()
        data_copy['RUL_classes'] = 0
        for i, t in enumerate(thresholds):
            if i+1 < len(thresholds):
                data_copy.loc[data_copy['RUL'].between(t, thresholds[i+1]),
                              'RUL_classes'] = i+1
            else:
                data_copy['RUL_classes'] = \
                    data_copy['RUL_classes'].mask(data_copy['RUL'] > t, i+1)

        # Get rid of unnecessary column
        data_copy['RUL'] = data_copy['RUL_classes']
        data_copy.drop('RUL_classes', axis=1, inplace=True)

        return data_copy

    def _train_test_split(self, data):
        '''Splits the data into a train and test set. This method is always
        called in a preprocessing step. If self._cycle_field is not None, the
        split is based on cycles, otherwise on rows.

        Args:
            data (pd.DataFrame): data to split

        Returns:
            data_copy (pd.DataFrame): same data with column for train/test
        '''
        data_copy = data.copy()
        data_copy['subset'] = 'test'
        if self._cycle_field is not None:
            split_idx = data_copy[self._cycle_field].unique()
            split_idx = np.random.choice(split_idx,
                                         size=round(len(split_idx)*0.7),
                                         replace=False)
            split_idx = data_copy[self._cycle_field].isin(split_idx)
        else:
            split_idx = data_copy.sample(frac=0.7).index

        data_copy.loc[split_idx, 'subset'] = 'train'

        return data_copy

    def _remove_outliers(self, data):
        '''Removes outliers for all columns in a data frame an replaces them
        with np.nan.

        Args:
            data (pd.DataFrame): data to remove outliers from

        Returns:
            data_copy (pd.DataFrame): data with outliers set to np.nan
        '''
        self._logger.info('Removing outliers from data')
        data_copy = data.copy()
        if self._cycle_field is not None:
            data_copy = data_copy.groupby(self._cycle_field). \
                transform(self._is_outlier)
            data_copy[self._cycle_field] = data[self._cycle_field]
        else:
            train = data_copy.copy().loc[data_copy['subset'] == 'train', :]
            test = data_copy.copy().loc[data_copy['subset'] == 'test', :]

            train = train.transform(self._is_outlier)
            test = test.transform(self._is_outlier)

            data_copy = pd.concat([train, test], axis=0)
            data_copy.sort_index(inplace=True)

        # add back old target
        data_copy[self._target] = data[self._target]

        return data_copy

    def _is_outlier(self, ser):
        '''Mask series with numpy.nan for values outside of IQR.

        Args:
            ser (pd.Series): series to mask
        Returns:
            ser_copy (pd.Series): series with outliers as np.nan
        '''
        # get the percentile values and range
        lower_p = np.nanpercentile(ser, 25, axis=0)
        upper_p = np.nanpercentile(ser, 75, axis=0)
        iqr = upper_p - lower_p

        # compute the outlier cutoff
        cut_off = iqr * 1.5

        # compute the lower and upper range with 1.5x of the bounds
        lower, upper = lower_p - cut_off, upper_p + cut_off

        # set the mask for the series
        mask = ser.between(lower, upper)

        return ser.where(mask, other=np.nan)

    def _impute_missing(self, data):
        '''Impute missing values given method specified

        Args:
            data (pd.DataFrame): data with nans to impute

        Returns:
            data_imputed (pd.DataFrame): copy of data with no missing values
        '''
        self._logger.info('Imputing missing values using forward fill')

        data_copy = data.copy()

        def _impute_cycle(ser):
            return ser.fillna(method='ffill')

        if self._cycle_field is not None:
            data_copy = data_copy.groupby(self._cycle_field). \
                transform(_impute_cycle)
            data_copy[self._cycle_field] = data[self._cycle_field]
        else:
            train = data_copy.copy().loc[data_copy['subset'] == 'train', :]
            test = data_copy.copy().loc[data_copy['subset'] == 'test', :]

            train = train.transform(_impute_cycle)
            test = test.transform(_impute_cycle)

            data_copy = pd.concat([train, test], axis=0)
            data_copy.sort_index(inplace=True)

        return data_copy

    def _normalise(self, data):
        '''Scales features across cycles given the self._specs['scaler']. Does
        not scale RUL or cycle_field if they exist in data.

        Args:
            data (pd.DataFrame): data to scale

        Returns:
            data_copy (pd.DataFrame): copy of data with scaled features
        '''
        self._logger.info('Normalizing data using RobustScaler')
        data_copy = data.copy()
        data_copy.drop(self._target, axis=1, inplace=True)

        if self._cycle_field in data_copy.columns:
            data_copy.drop(self._cycle_field, axis=1, inplace=True)

        scaler = RobustScaler(quantile_range=(25, 75))

        train = data_copy.copy().loc[data_copy['subset'] == 'train', :]
        test = data_copy.copy().loc[data_copy['subset'] == 'test', :]
        train.drop('subset', axis=1, inplace=True)
        test.drop('subset', axis=1, inplace=True)

        train_scaled = scaler.fit_transform(train)
        test_scaled = scaler.transform(test)

        train = pd.DataFrame(train_scaled,
                             index=train.index,
                             columns=scaler.get_feature_names_out())
        test = pd.DataFrame(test_scaled,
                            index=test.index,
                            columns=scaler.get_feature_names_out())

        data_copy = pd.concat([train, test], axis=0)
        data_copy.sort_index(inplace=True)

        if self._cycle_field is not None:
            data_copy[[self._cycle_field, self._target, 'subset']] = \
                data[[self._cycle_field, self._target, 'subset']]
        else:
            data_copy[[self._target, 'subset']] = \
                data[[self._target, 'subset']]

        return data_copy

    def _trim_data(self, data):
        '''This function selects a maximum number of time steps to consider
        within any given cycle. The purpose of this function is to trim rows
        mapping to the maximum RUL variable (in the training set) aiming to
        reduce skewness in target variable distribution used during training.
        Trimming is only done for the training set.

        Args:
            data (pd.DataFrame): data with RUL column

        Returns:
            data_copy (pd.DataFrame): data trimmed from too many values per
                cycle.
        '''
        self._logger.info(f'Trimming data from {self._specs["initial_RUL"]*2}')

        train = data.copy().loc[data['subset'] == 'train', :]
        test = data.copy().loc[data['subset'] == 'test', :]
        max_RUL = self._specs['initial_RUL']*2
        train = train.drop(
            (train.loc[train[self._target] > max_RUL].index)
        )

        data_copy = pd.concat([train, test], axis=0)
        data_copy.sort_index(inplace=True)

        return data_copy

    def _function_mapping(self, name) -> dict:
        '''Maps preprocessing names to class methods.

        Args:
            name (str): preprocessing step

        Returns:
            mapping[name] (method): Corresponding method in class
        '''
        return self.function_map[name]

    @property
    def function_map(self):
        '''Function name to method in class'''
        return {
            'make_target': self._add_remaining_useful_life,
            'remove_outliers': self._remove_outliers,
            'train_test_split': self._train_test_split,
            'impute_missing_values': self._impute_missing,
            'normalise': self._normalise,
            'trim_data': self._trim_data,
            'rul_class': self._add_degradation_class,
        }

    @property
    def function_order(self):
        '''Set order of the functions in preprocessing, so that for example
        remove outliers does not happen after imputing missing values.'''
        return {
            'make_target': 0,
            'remove_outliers': 1,
            'train_test_split': 2,
            'impute_missing_values': 3,
            'normalise': 4,
            'trim_data': 5,
            'rul_class': 6,
        }
