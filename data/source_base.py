from abc import ABC, abstractmethod
import logging


class Source(ABC):
    """Base class for all source objects that specific db classes can inherit
    from.
    """

    def __init__(self, output_collector):
        self._logger = logging.getLogger(__name__)
        self._out_coll = output_collector
        self.metadata = {}

    def load_from_query(self,
                        query: str,
                        query_type: str = None):
        """Load data from query

        Args:
            query_type (str): How data should be queried from database. Default
                              is None.
            query (str): Query as a single string.

        Returns:
            data: Output from load query. Format depends on client and query.
                Returns None is load returns None or length of output is 0.
        """
        # save the query
        self._out_coll.add_output({
            'source_query': query
        })
        # Load data
        data = self.load(query, query_type)

        # Check if data is returned by query, else return None
        if data is None or len(data) == 0:
            self._logger.warn(
                "WARNING: no results for the given query, returning None")
            return None

        return data

    def load_from_configs(self,
                          query_config: dict,
                          query_template: str = None,
                          query_template_path: str = None,
                          query_type: str = None):
        """Loads the data based on the query tempalte and query config. Should
        be provided with either query template or a path to a file where the
        tempalte is stored. If both ``query_template`` and
        ``query_template_path are provided, ``query_template`` takes
        precedence.

        Args:
            query_config (dict): dictionary of values to fill query template
            query_template (str, optional): query template string. Must be
            provided is ``query_template_path`` is None. Defaults to None.
            query_template_path (str, optional): path to a text file containing
            the query template. Must be provided is query_template is None.
            Defaults to None.
            query_type (str, optional): Type of query you want to run, read
            guides associated with the data store you are using for further
            information. Defaults to None.

        Returns:
            (depends on ``query_type``): the data returned by the data store
            connector
        """

        self._logger.info('Starting load_from_configs')

        if query_template is None:
            redundant_path = False
            if query_template_path is not None:
                # read the template
                with open(query_template_path, 'r') as f:
                    query_template = f.read()
            else:
                raise ValueError('Loading from config requries query'
                                 'template path or query template itself'
                                 '. Neither was provided.')
        else:
            if query_template_path is not None:
                redundant_path = True
                self._logger.warning('Both query template and query template '
                                     'and query template path were provided.'
                                     'Query template takes pecedence')
            else:
                redundant_path = False

        try:
            # composing the query from the template and config
            self._logger.info('Composing the query from template and config')
            query = self._compose_query(
                query_template,
                query_config
            )

            # load raw data from the database
            self._logger.info('Using the query to load data')
            data = self.load_from_query(query, query_type)
            self._logger.info('Data loaded')
        except Exception as e:
            if redundant_path:
                self._logger.warning('Loading with query template failed with '
                                     f'a message {e}.'
                                     ' Trying with template path')
                return self.load_from_configs(
                    query_config=query_config,
                    query_template_path=query_template_path,
                    query_template=None,
                    query_type=query_type
                )
            else:
                raise ValueError('Loading with configs failed with message '
                                 f'{e} Please check '
                                 'your query template and config values')

        return data

    @abstractmethod
    def load(self,
             query: str,
             query_type: str = None):
        """Abstract method for loading data from database
        """
        pass

    @abstractmethod
    def _compose_query(self,
                       query_template: str,
                       query_config: str):
        """Abstract method for composing query from template and configs
        """
        pass
