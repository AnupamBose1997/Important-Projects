from data.source_base import Source
import pandas as pd


class LocalDataLoader(Source):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metadata = {}

    def load(self,
             query: dict,
             query_type: str = None):
        '''Method for loading data from local file. query_config is given to
        read function as dict.

        query (dict): Parameters for read
        query_type (str): How to read data, csv or excel

        Returns:
            data (pd.DataFrame)
        '''
        self._out_coll.add_output({
            'source_query': str(query)
        })
        if query_type == 'csv':
            data = pd.read_csv(**query)
        elif query_type == 'excel':
            data = pd.read_excel(**query)
        else:
            msg = f'{query_type} not valid query type. Valid are "csv" and "excel"'
            raise ValueError(msg)

        return data

    def load_from_configs(self,
                          query_config: dict = None,
                          query_template: str = None,
                          query_template_path: str = None,
                          query_type: str = None):
        '''Method to load data from configs in PipelineStep. In this
        implementation it only uses the query_config and query type to call the
        load method. Use as:

        .. highlight:: python
        .. code-block:: python

            load_from_configs(query_config={
                    'filepath_or_buffer': 'path/to/file.csv'
                },
                query_type='csv')
            ...

        Args:
            query_config (dict): Arguments to pass to load method
            query_template (str): Ignored argument for PipelineStep
            query_template_path (str): Ignored argument for PipelineStep
            query_type (str): How to get data, e.g. csv or excel

        Returns:
            data (pd.DataFrame)
        '''
        data = self.load(query_config, query_type)

        if data is None or len(data) == 0:
            self._logger.warn(
                "WARNING: no results for the given query, returning None")
            return None

        return data

    def _compose_query(self):
        '''Placeholder for Source base'''
        pass
