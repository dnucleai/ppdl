import pickle

import torch
import requests

weights = torch.normal(mean=torch.arange(1, 51), std=0.1)

print("Old weights", weights)

get_url = "http://localhost:5000/fetch_weights/"
post_url = "http://localhost:5000/store_weights/"

requests.post(post_url, pickle.dumps(weights))

r = requests.get(get_url)

new_weights = pickle.loads(r.content)

print("New weights", new_weights)

requests.post(post_url, pickle.dumps(weights))

r = requests.get(get_url)

new_weights = pickle.loads(r.content)

print("New weights", new_weights)


