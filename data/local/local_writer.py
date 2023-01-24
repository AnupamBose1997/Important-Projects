from data.sink_base import Sink
import shutil
import pandas as pd
import json
import os


class LocalDataWriter(Sink):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def write(self,
              data,
              file_path: str,
              write_config: dict = None,
              metadata=None,
              **kwargs):
        '''Write to local filepath. file_path is used to define where data
        should be written to and what method to use (from extension). Use as:

        .. highlight:: python
        .. code-block:: python

            self.write(data=df, file_path='path/to/file.csv')
            ...

        Args:
            data: data to be written. Type depends on writing method.
            metadata: metadata from data loader.
            file_path (str): defines where data should be written and extension
                defines method. Currently accepts json, csv and xls/xlsx. If
                json, then data is assumed to be dictionary, otherwise data is
                assumed to be pd.DataFrame.
            write_config (dict): any key-value arguments to give to write
                functions. Defaults to None.

        Returns:
            (True) if successfully written
        '''
        if write_config is None:
            write_config = {'index': False}

        if file_path.endswith('.csv'):
            assert isinstance(data, pd.DataFrame), \
                'For csv writing, data must be pd.DataFrame'
            self._logger.info(f'Writing data to csv file: {file_path}')
            data.to_csv(path_or_buf=file_path, **write_config)
            return True
        elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            assert isinstance(data, pd.DataFrame), \
                'For xls/xlsx writing, data must be pd.DataFrame'
            self._logger.info(f'Writing data to csv file: {file_path}')
            data.to_excel(excel_writer=file_path, **write_config)
            return True
        elif file_path.endswith('.json'):
            assert isinstance(data, dict), \
                'For json writing, data must be dictionary'
            self._logger.info(f'Writing data to json file: {file_path}')
            with open(file_path, 'w') as _f:
                json.dump(data, _f)
            return True
        elif file_path.endswith('.zip'):
            assert os.path.isdir(data), 'For writing to zip, data must be dir'
            self._logger.info(f'Writing data to zip file: {file_path}')
            shutil.make_archive(file_path.rstrip('.zip'), 'zip', data)
            return True
        else:
            msg = f'{file_path} does not have valid extension. \
Valid extensions are: ".csv", ".xls/.xlsx", ".json", or ".zip".'
            raise ValueError(msg)
