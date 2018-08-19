import pickle

from flask import Flask, request, Response
from ppdl.server.parameter_server import ParameterServer
from ppdl.storage.memory_weight_storage import MemoryWeightStorage

SIZE = 21840
NUM_CLIENTS = 2
THETA = 1

app = Flask(__name__)

storage = MemoryWeightStorage()
server = ParameterServer(storage, NUM_CLIENTS, THETA)
server.initialize_weights(SIZE)


@app.route("/register/<client_id>", methods=['POST'])
def register_client(client_id):
    print("Registering", client_id)
    server.register_client(client_id)
    return Response(status=200)


@app.route("/store_weights/<client_id>", methods=['POST'])
def store_weights(client_id):
    print("Storing weights", client_id)
    deltas = pickle.loads(request.get_data())
    print("Deltas", deltas)
    server.update_weights(client_id, deltas)
    return Response(status=200)


@app.route("/fetch_weights/<client_id>", methods=['GET'])
def fetch_weights(client_id):
    print("Fetching weights", client_id)
    result = server.fetch_weights()
    print(result)
    return pickle.dumps(result)


if __name__ == '__main__':
    app.run(debug=True)
