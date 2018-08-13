import pickle

import requests

from ppdl.client.parameter_manager import ParameterManager


class HttpParameterManager(ParameterManager):

    def __init__(self, get_url, post_url):
        self.get_url = get_url
        self.post_url = post_url

    def upload_deltas(self, client_id, deltas):
        url = "%s/%s" % (self.post_url, client_id)
        requests.post(url, pickle.dumps(deltas))

    def download_params(self, client_id):
        url = "%s/%s" % (self.get_url, client_id)
        response = requests.get(url)
        result = pickle.load(response.content)
        return result['indices'], result['weights']
