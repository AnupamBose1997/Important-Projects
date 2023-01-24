import logging
import warnings


class TimeSeriesDataSplitter:
    '''
    Class to split time-series data into training, validation and test sets
    for model development.

    Becuase the data is time-series, the order is upheld.

    Args:
        train_size (float): proportion of dataset used for training.
        val_size (float): proportion of dataset used for validation.
        test_size (float): proportion of dataset used for testing.

    '''

    def __init__(self, train_size, val_size, test_size):
        self._logger = logging.getLogger(__name__)
        self._train_size = train_size
        self._val_size = val_size
        self._test_size = test_size

        total_size = self._train_size + self._val_size + self._test_size
        if total_size < 1:
            warnings.warn(
                f''' Total size of subsets is {total_size}, not using all
                available data. '''
                )

        if total_size > 1:
            raise ValueError(
                f''' Total size of subsets is {total_size}, cannot be greater
                than 1. '''
                )

    def split_data(self, df):
        '''
        Splits time-series data into 3 three sets.

        Args:
            df (DataFrame): dataset to be split up.

        Returns:
            train (DataFrame): training set.
            val (DataFrame): validation set.
            test (DataFrame): test set.

        '''
        # convert from proportion to number of rows
        train_size_samples = round(self._train_size * len(df))
        val_size_samples = round(self._val_size * len(df))

        # find the final index of the train set
        train_end_index = train_size_samples

        # find the start and end indices of the validation set
        val_start_index = train_end_index
        val_end_index = val_start_index + val_size_samples

        # find the start index of the testing set
        test_start_index = val_end_index

        # perform the split based on the indices
        train = df.iloc[0:train_end_index, :]
        self._logger.info(f'Train set preview: {train.head()}')

        val = df.iloc[val_start_index:val_end_index, :]
        self._logger.debug(f'Validation set preview: {val.head()}')

        test = df.iloc[test_start_index:, :]
        self._logger.debug(f'Test set preview: {test.head()}')

        self._logger.info(
            f'''df length: {len(df)}, combined length of train, val, test:
                {len(train)+len(val)+len(test)}'''
                )

        self._logger.debug(
            f'''Train length: {len(train)},
            train proportion: {len(train)/len(df):.2f}'''
            )

        self._logger.debug(
            f'''Val length: {len(val)},
            val proportion: {len(val)/len(df):.2f}'''
            )

        self._logger.debug(
            f'''Test length: {len(test)},
            test proportion: {len(test)/len(df):.2f}'''
            )

        return train, val, test
