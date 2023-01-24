import logging
import pandas as pd
from itertools import product
from sklearn.base import clone
import time


class GridSearch:
    '''
    Class to perform a grid search to optimise a model's hyparparameters.

    Args:
        model (sklearn estimator): the model to be optimised, with default
                                   estimator parameters.
        grid (dict): dictionary containing keys of parameter names, with
                     corresponding values as a list over which the search is
                     carried out.
        metric (function): the metric used to evaluate each set of
                           hyperparameters.

    '''

    def __init__(self, model, grid, metric):
        self._logger = logging.getLogger(__name__)
        self._base_model = model
        self._grid = grid
        # get every hyperparameter combination as a list of dicts
        self._param_combos = [dict(zip(self._grid.keys(), v))
                              for v in product(*self._grid.values())]
        self._logger.info(
            'Total number of hyperparameter combinations: '
            f'{len(self._param_combos)}'
            )
        self._metric = metric
        self._metric_name = self._metric.__name__

    def _get_best_params(self):
        '''
        Gets the set of hyperparameters corresponding to the best validation
        score.

        This is the set of parameters in the top row of results_sorted df.

        Returns:
            dict: best set of hyperparameters.

        '''
        # get the top row from results_sorted
        # astype(object) is needed to stop data types changing
        row = self.results_sorted.astype(object).iloc[0, :].copy()
        # drop the evaluation score
        row_params_only = row.drop(self._metric_name)
        return row_params_only.to_dict()

    def _join_train_and_val(self, x_train, x_val, y_train, y_val):
        '''
        Rejoins training and validation data.

        Used to train a final model using all available data.

        Args:
            x_train (pd.DataFrame): input training data.
            x_val (pd.DataFrame): input validation data.
            y_train (pd.Series): output training data.
            y_val (pd.Series): output validation data.

        Returns:
            x (pd.DataFrame): input data.
            y (pd.Series): output target data.

        '''
        x = pd.concat([x_train, x_val], axis=0)
        y = pd.concat([y_train, y_val], axis=0)
        return x, y

    def run_grid_search(self, x_train, x_val, y_train, y_val):
        '''
        Performs a grid search for hyperparameter optimisation.

        For each hyperparameter set, the base_model is built with that set of
        parameters and trained on the training data. Each model is then
        evaluated on the validation set. A DataFrame is then built, with each
        row containing the parameters and the validation score. It is then
        sorted in ascending order.

        todo: also allow sorting in descending order if score is to be
        maximised.

        Args:
            x_train (pd.DataFrame): input training data.
            x_val (pd.DataFrame): input validation data.
            y_train (pd.Series): output training data.
            y_val (pd.Series): output validation data.

        Returns:
            None.

        '''
        self._logger.info('Starting grid search')
        results = []
        start_time = time.time()
        for count, params in enumerate(self._param_combos):
            self._logger.debug(f'Current set of parameters: {params}')
            self._logger.debug(
                f'Progress: {count} / {len(self._param_combos)}'
            )
            # fit the model with current set of parameters
            # clone of base_model taken to keep base_model unchanged
            temp_model = clone(self._base_model).set_params(**params)
            # fit the model with the training data
            temp_model.fit(x_train, y_train)
            # make predictions on validation data and evaluate
            y_val_predictions = temp_model.predict(x_val)
            score = self._metric(y_val, y_val_predictions)
            results.append(list(params.values())+[score])
        # compute how long grid search took
        end_time = time.time()
        total_time_s = int(end_time - start_time)
        total_time_h = round(total_time_s/3600, 4)
        self._logger.info(f'Grid search took {total_time_h} hours to complete')
        # build DataFrame containing results
        columns = list(params.keys())+[self._metric_name]
        results = pd.DataFrame(data=results, columns=columns)
        # sort in ascending order, make instance attribute
        self.results_sorted = results.sort_values(
            self._metric_name,
            ascending=True
            )
        self._logger.info(
            f'Grid search finished, results: {self.results_sorted}'
            )

    def fit_with_best_params(self, x_train, x_val, y_train, y_val):
        '''
        Fits the base_model with the optimal hyperparameters found in the grid
        search.

        The training and validation data are joined to make use of all data.

        Args:
            x_train (pd.DataFrame): input training data.
            x_val (pd.DataFrame): input validation data.
            y_train (pd.Series): output training data.
            y_val (pd.Series): output validation data.

        Returns:
            best_model (sklearn estimator): model with optimised hyperparams
                                            fitted on all the data.

        '''
        self._logger.info(
            'Fitting the model with best hyperparams on all data'
            )
        best_params = self._get_best_params()
        x, y = self._join_train_and_val(x_train, x_val, y_train, y_val)
        # instantiate model with the optimal hyperparameters
        best_model = clone(self._base_model).set_params(**best_params)
        # fit on all the data
        best_model.fit(x, y)
        return best_model
