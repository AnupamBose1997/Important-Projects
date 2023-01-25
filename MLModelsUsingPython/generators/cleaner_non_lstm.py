import numpy as np
import pandas as pd
import logging
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataCleaner:
    """
    Object for cleaning features of a single cycle

    Args:
        sensor_features (list): list of sensor features to be cleaned.
        cleaning_steps (dict): dictionary with specification for cleaning.
                               Steps not taken are set to None,
                               others specify parameters for step.
    """

    def __init__(
        self,
        sensor_features: list,
        cleaning_steps: dict,
    ):

        self._logger = logging.getLogger(__name__)
        self._logger.info("Cleaning for generator starting")

        self._sensor_features = sensor_features
        self._cleaning_steps = cleaning_steps

    def clean_data(self, df):

        """
        Runs cleaning steps on data frame

        Args:
            df: data to be cleaned

        Returns:
            df (pd.DataFrame): cleaned data with RUL.
                               If some columns of df are all missing,
                               returns None.

        """

        # Reset index and make time_cycles for RUL creation and sorting
        df = df.reset_index(drop=True)
        df["time_cycles"] = df.index

        # Sort df by timestamp and time_cycles to get chronological data
        df = df.sort_values(by=["_time", "time_cycles"])

        # Always add RUL
        df = self._add_remaining_useful_life(df,
                                             self._cleaning_steps["RUL_clip"])

        # Run cleaning steps

        # Outliers
        if self._cleaning_steps["outliers"]["run_step"]:
            df = self._handle_outliers(df, self._sensor_features)

        # Return None if some columns have all missing to skip cycle
        if self._check_if_all_nan(df,
                                  self._cleaning_steps["nan_reject"]):
            return None

        # Imputation
        if self._cleaning_steps["imputation"] is not None:
            df = self._impute_groups(
                df,
                self._sensor_features,
                self._cleaning_steps["imputation"],
            )

        # Scaling
        if self._cleaning_steps["scaling"] is not None:
            df = self._scale_features(
                df,
                self._sensor_features,
                self._cleaning_steps["scaling"],
            )

        # Skipping rows
        if self._cleaning_steps["skip_rows"] is not None:
            df = self._skip_n_rows(df, self._cleaning_steps["skip_rows"])

        # Lagging
        if self._cleaning_steps["lagging"] is not None:
            df = self._lag_features(
                df,
                self._sensor_features,
                self._cleaning_steps["lagging"],
            )

        self._logger.info("Cleaning for generator finished")

        return df

    def _add_remaining_useful_life(self, df, max_RUL: int = 0):
        """
        Add remaining useful life (RUL) and clipped RUL columns
        to input dataframe.

        Args:
            df: data frame with feature data and time_cycles column
                with stroke count.
            upper (int): clipping level for RUL. Defaults to 0,
                         i.e. no clipping.

        Returns:
            df (pd.DataFrame): data frame with RUL and RUL_clipped variables.
        """

        # Make max_cycle
        df["max_cycle"] = df["time_cycles"].max()

        # Add new columns to the dataset
        df["RUL"] = df["max_cycle"] - df["time_cycles"] + 1

        # drop max_cycle as it's no longer needed
        df = df.drop("max_cycle", axis=1)

        clip_RUL = True if max_RUL > 0 else False

        # Clip RUL if True
        if clip_RUL:
            df["RUL"].clip(upper=max_RUL, inplace=True)

        return df

    def _handle_outliers(self, df, sensor_features: list):
        """
        Finds outliers and makes them NaN

        Args:
            df: data frame with features to clean
            sensor_features (list): features to handle outliers for

        Returns:
            df (pd.DataFrame): data with outliers as NaN
        """

        # Set outliers to NaN
        df[sensor_features] = df[sensor_features].transform(
            self._is_outlier,
            **self._cleaning_steps["outliers"]["outlier_range"],
        )

        # assign temporary series that flags row with column
        # containing NaNs - NaN, 1 - otherwise
        df["subcycle"] = np.where(df.notna().all(axis=1), 1, np.nan)

        # update the temporary series to trim the beginning and
        # ending which are outside range
        df["subcycle"] = df["subcycle"].transform(self._trim_series)

        # drop rows which have subcycle = nan
        df.dropna(subset=["subcycle"], inplace=True)

        # drop the temporary column
        df.drop(["subcycle"], axis=1, inplace=True)

        return df

    def _is_outlier(
        self,
        ser,
        lower_p: int = 25,
        upper_p: int = 75,
    ):
        """
        Mask series with numpy.nan for values outside of percentile range.

        Args:
            ser: series to mask
            lower_p (int, optional): lower percentile in range
            upper_p (int, optional): upper percentile in range

        Returns:
            series with True for rows within percentile range, otherwise np.nan
        """

        # get the percentile values and range
        lower_p = np.percentile(ser, lower_p, axis=0)
        upper_p = np.percentile(ser, upper_p, axis=0)
        iqr = upper_p - lower_p

        # compute the outlier cutoff
        cut_off = iqr * 1.5

        # compute the lower and upper range with 1.5x of the bounds
        lower, upper = lower_p - cut_off, upper_p + cut_off

        # set the mask for the series
        mask = ser.between(lower, upper)

        return ser.where(mask, other=np.nan)

    def _first_and_last_valid_index(self, ser):
        """
        Return first and last valid index where pandas.series is not NaN
        """

        # first_valid_index() returns None if no missing values
        # Return first index if first_valid_index() is None
        # Same for last_valid_index()
        first_valid_ind = (
            ser.index.get_loc(ser.first_valid_index())
            if ser.first_valid_index() is not None
            else ser.index[0]
        )
        last_valid_ind = (
            ser.index.get_loc(ser.last_valid_index())
            if ser.last_valid_index() is not None
            else ser.index[-1]
        )

        return first_valid_ind, last_valid_ind

    def _trim_series(self, ser):
        """
        Delete rows that are not in the list of valid indices

        Args:
            ser: series with data

        Returns:
            ser: series where all values after first_valid_ind
                 and before last_valid_ind are 1
        """

        # Make copy
        ser = ser.copy()

        # get the first and last valid index which is not numpy.nan
        first_valid_ind, last_valid_ind = self._first_and_last_valid_index(ser)

        # set the values to 1 if within the valid range, and numpy.nan others
        ser.iloc[first_valid_ind:last_valid_ind + 1] = 1

        return ser

    def _check_if_all_nan(self, df, rejection_rate: float = 1.0):
        """
        Checks if any columns have all missing values.

        Args:
            df: data frame
            rejection_rate (float, optional): rate of NaNs taht we
                                              reject data at

        Returns:
            True if all missing in some column, otherwise False.
        """

        # Get number of strokes not NaN for data to be OK
        strokes_to_reject = df.shape[0] * rejection_rate

        cols_with_only_nans = np.where(
            df.isnull().sum() >= strokes_to_reject, True, False
        )

        if sum(cols_with_only_nans) > 1:
            return True

        return False

    def _impute_groups(
        self, df, sensor_features: list, impute_opt: str = "SimpleImputer"
    ):

        """
        Method to impute missing to mean.

        Args:
            df: data frame with features.
            sensor_features (list): list of features to impute values for.

        Returns:
            df: data frame with no missing values.
        """

        # set the imputer
        if impute_opt == "KNNImputer":
            imputer = KNNImputer(n_neighbors=2)
        elif impute_opt == "SimpleImputer":
            imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

        # Apply imputation for each feature
        df[sensor_features] = df[sensor_features].transform(
            lambda x: imputer.fit_transform(x.values.reshape(-1, 1)).ravel()
        )

        return df

    def _scale_features(
        self,
        df,
        sensor_features: list,
        scaling_method: str = "MinMax",
    ):
        """
        Scales data frame given specified method.

        Args:
            df: data frame with cycle data.
            sensor_features (list): list of features to scale.
            scaling_method (str, optional): method to use when scaling.
                                            Defaults to MinMax.

        Returns:
            df (pd.DataFrame): data frame with scaled features
        """

        # Pick scaling method
        if scaling_method == "MinMax":
            scaler = MinMaxScaler()
        elif scaling_method == "Standard":
            scaler = StandardScaler()

        # Scale features
        df[sensor_features] = scaler.fit_transform(df[sensor_features])

        return df

    def _skip_n_rows(
        self,
        df,
        num_skip: int = 5,
    ):
        """
        Samples every n rows from df

        Args:
            df: data frame
            num_skips (int, optional): how many rows to skip

        Returns:
            df (pd.DataFrame): subsample of original data
        """

        return df[::num_skip]

    def _lag_features(
        self,
        df,
        sensor_features: list,
        num_lags: int = 2,
    ):
        """
        Lags features in data frame

        Args:
            df: data frame with cycle data.
            sensor_features (list): list of features to scale.
            num_lags (int, optional): number of timesteps to go back when
                                      lagging features. Defaults to 2.

        Returns:
            df (pd.DataFrame): data frame with lagged features
        """

        # Make a copy of the dataframe
        df_copy = df[sensor_features]

        # Get list of shifted dataframes
        shifted_dfs = []
        for i in range(1, num_lags + 1):
            shifted_df = df_copy.shift(i)
            shifted_df.columns = [f"{n} (t-{i})" for n in df_copy.columns]
            shifted_dfs.append(shifted_df)

        # Concatenate all shifted dfs horizontally
        new_df = pd.concat(shifted_dfs, axis=1)

        # Join the original df to the new df
        df = pd.concat([df, new_df], axis=1)

        # Drop rows with nans created from shifting and reset the index
        df = df.dropna(inplace=False).reset_index(inplace=False, drop=True)

        return df
