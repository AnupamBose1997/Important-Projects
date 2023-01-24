from datasinksourcelibrary.source_base import Source

import pandas as pd
import requests
import json


class ApiDataLoader(Source):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metadata = {}

    def load(self, query: str, query_type: str):
        """Method for fetching data from api endpoint.

        :param query: Parameters to read the api endpoint
        :type query: str
        :param query_type: Parameter to choose to load the data as json or DataFrame
        :type query_type: str
        :return: json or DataFrame
        :rtype: dict or DataFrame
        """
        self._out_coll.add_output({
            'source_query': str(query)
        })
        self.url = query
        self.response = requests.get(self.url, timeout=5)
        try:
            self.response = requests.get(self.url, timeout=5)
            self.response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print(errh)
        except requests.exceptions.ConnectionError as errc:
            print(errc)
        except requests.exceptions.Timeout as errt:
            print(errt)
        except requests.exceptions.RequestException as err:
            print(err)
        if query_type == 'api_json':
            if self.response.status_code == 200:
                return self.response.json()
        elif query_type == 'api_df':
            data_list = []
            if self.response.status_code == 200:
                for i in self.response.json().items():
                    # tuples for ease of parsing
                    data_list.append([i[0], i[1]])
                df_api = pd.DataFrame(data_list, columns=['key', 'value'])
                # flatten the list
                df_api = df_api.explode('value')
                # convert it to json and orient to records
                # json string to python dictionary and normalize it
                return pd.json_normalize(json.loads(df_api.to_json(orient="records")))

    def load_from_configs(self,
                          query_config: dict = None,
                          query_template_path: str = None,
                          query_type: str = None):
        """Method to load data from configs in PipelineStep. In this
        implementation it only uses the query_template_path and query type (api_json or api_df) to call the
        load method. Use as:
        load_from_configs(query_config={'query_template_path': 'path/to/file.txt'},query_type='api_json'or'api_df')
        :param query_config: Ignored argument for PipelineStep
        :type query_config: dict
        :param query_template: Ignored argument for PipelineStep, defaults to None
        :type query_template: str, optional
        :param query_template_path: path of the template, defaults to None
        :type query_template_path: str, optional
        :param query_type: How to get data, e.g. json or DataFrame, defaults to None
        :type query_type: str, optional
        :return: data DataFrame or json depending on the query_type entered by user
        :rtype: json or DataFrame
        """
        with open(query_template_path, 'r') as f:
            query = f.read()
        data = self.load(query, query_type)

        if data is None or len(data) == 0:
            self._logger.warn(
                "WARNING: no results for the given query, returning None")
            return None

        return data

    def _compose_query(self):
        """Placeholder for Source base
        """
        pass
