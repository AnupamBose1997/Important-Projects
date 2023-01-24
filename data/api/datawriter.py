from data.sink_base import Sink

import requests
import pandas as pd


class ApiDataWriter(Sink):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def write(self, data, url: str):
        """rite data using API endpoint. url is used to define where data should be posted to.
        Data can be of type DataFrame or json
        :param data: data to be written
        :type data: DataFrame or json
        :param url: API endpoint where the data should be posted to
        :type url: str
        :return: (<Response [201]>) if successfully posted
        :rtype: str
        """
        if isinstance(data, pd.DataFrame):
            data = data.to_json(orient='columns')
        elif isinstance(data, dict):
            data = data
        try:
            r = requests.post(url, data=data)
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print(errh)
        except requests.exceptions.ConnectionError as errc:
            print(errc)
        except requests.exceptions.Timeout as errt:
            print(errt)
        except requests.exceptions.RequestException as err:
            print(err)
        return requests.post(url, data=data)
