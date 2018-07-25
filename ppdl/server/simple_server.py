import pickle

from flask import Flask, request, Response
from ppdl.server.parameter_server import  ParameterServer
from ppdl.storage.memory_weight_storage import MemoryWeightStorage

SIZE = 50

app = Flask(__name__)

storage = MemoryWeightStorage()
server = ParameterServer(storage)
server.initialize_weights(SIZE)


@app.route("/store_weights/", methods = ['POST'])
def store_weights():
    deltas = pickle.loads(request.get_data())
    server.update_weights(deltas)
    return Response(status=200)


@app.route("/fetch_weights/", methods = ['GET'])
def fetch_weights():
    return pickle.dumps(server.fetch_weights())


if __name__ == '__main__':
    app.run(debug=True)
