import logging

import influxdb_client
import pandas as pd
from datasinksourcelibrary.influxdb.utils import dataframe_to_list_of_dicts


class InfluxFlatConverter:
    """This class converts data from the format in which influx returns the data
    (list of influx tables) to a pandas dataframe with columns as fields and
    tags and time and encoder angle as index.
    Accepted ``query_type`` values: 'influx'
    """

    def __init__(self, tags_to_keep=[]):
        self._logger = logging.getLogger(__name__)
        self._logger.info(
            f'Influx flat converter initialized, keeping tags: {tags_to_keep}'
        )
        self._tags_to_keep = tags_to_keep

    def convert_for_analysis(self, input_data: list):
        assert type(input_data) == list, ValueError(
            'InfluxFlatConverter accepts list of influxtables as input data')
        if len(input_data) == 0:
            self._logger.info('No data has been loaded')
            return None
        assert type(input_data[0]) == \
            influxdb_client.client.flux_table.FluxTable, ValueError(
                f'Converter accepts list of influx tables,\
                    not {type(input_data[0])}'
        )
        if type(input_data[0]) == pd.core.frame.DataFrame:
            converted_data = self._list_of_dataframes_to_dataframe(input_data)
        else:
            converted_data = self._list_of_influx_tables_to_dataframe(
                input_data
            )
        self._logger.info(f'Converted Data \n {converted_data.head()}')
        return converted_data

    def convert_for_writing(input_data, measurement: str, tags: dict):
        """Public interface to converting the data form pandas dataframe
        into the list of dicts understood by influx. First
        implementation is for dataframe with time as index and fields as
        columns, measurement and tags passed separately.
        Args:
            input_data ([type]): data (values) to be written to influx
            measurement (str): measurement name
            tags (dict): dictionary of tags to add to the values
        """
        assert type(input_data) == pd.core.frame.DataFrame, ValueError(
            'InfluxFlatConverter only accepts pandas dataframe to \
                convert for writing'
        )
        converted_data = dataframe_to_list_of_dicts(input_data,
                                                    measurement,
                                                    tags)

        return converted_data

    def _list_of_influx_tables_to_dataframe(self, table_list: list):

        self._logger.info(
            'Converting the list of Influx tables to dataframe for analysis'
        )

        self._logger.info('Converting each table to dataframe')
        # extract data from influx records to dataframes
        dfs = [
            pd.DataFrame([rec.values for rec in table.records])
            for table in table_list
        ]
        # if there is no data at given query, return None
        if len(dfs) == 0:
            return None

        full_df = self._list_of_dataframes_to_dataframe(dfs)

        return full_df

    def _list_of_dataframes_to_dataframe(self, dfs: list):

        full_df = pd.concat(dfs, axis=0)
        full_df.reset_index(inplace=True)
        full_df = full_df.pivot(
            columns=['_field'],
            values=['_value'],
            index=['_time'] + self._tags_to_keep
        )
        full_df.columns = full_df.columns.droplevel(0)
        full_df.sort_index(level=['_time'], inplace=True)
        for tag in self._tags_to_keep:
            full_df[tag] = full_df.index.get_level_values(tag).astype('int')
            full_df.index = full_df.index.droplevel(tag)

        full_df.index.rename('time', inplace=True)
        # sort DataFrame columns before returning
        col_names = list(full_df.columns).copy()
        col_names.sort(key=str)
        full_df = full_df[col_names]

        return full_df
