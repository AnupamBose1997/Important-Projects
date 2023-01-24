import logging

from octaipipe.models.model_base import Model
from skl2onnx import convert_sklearn
from sklearn.cluster import KMeans


class KMeansClustering(Model):
    '''
    Kmeans model class

    `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`__
    '''
    def __init__(self, n_clusters, **kwargs):
        super().__init__(**kwargs)
        self._solutiontype = 'clustering'
        self._logger = logging.getLogger(__name__)
        self.n_clusters = n_clusters

    def _build_new(self):
        self._logger.info('Instantiating linear regression model')
        self.estimator = KMeans(n_clusters=self.n_clusters)

    def _to_onnx(self):
        '''
        Method to convert model to onnx for saving. The skl2onnx function for
        conversion is used and then the filename is changed in order to save
        the model as onnx.
        '''
        self.onnx_estimator = convert_sklearn(self.estimator,
                                              initial_types=self.data_schema)
