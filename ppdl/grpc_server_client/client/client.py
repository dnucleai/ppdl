import grpc
import nucleai_pb2 as pb
import nucleai_pb2_grpc as pb_grpc
import time


class Client():

    def __init__(self, client_id, url=None):
        self.channel = grpc.insecure_channel(url or "localhost:1453")
        self.stub = pb_grpc.LearningStub(self.channel)
        self.id = client_id

    def _get_client_id(self):
        return pb.ClientId(txt=self.id)

    # deltas: {index: value} of deltas
    def upload(self, cycle_id, deltas):
        self.stub.Upload(pb.UploadRequest(
            cycleId=pb.CycleId(num=cycle_id),
            clientId=self._get_client_id(),
            deltas=pb.Parameters(parameters=[pb.IndexedValue(index=idx, value=val) for idx, val in deltas.items()]),
            ))

    # returns (cycle number, ttl, {index: value} of parameters)
    def download(self):
        ret = self.stub.Download(pb.DownloadRequest(clientId=self._get_client_id()))
        return (ret.cycleId.num, ret.waitTime.secondsFromNow, {iv.index: iv.value for iv in ret.parameters.parameters})


if __name__ == "__main__":
    client = Client("client_" + str(time.time()))
    cycle_id, params = client.download()
    client.upload(cycle_id, {1: 1e6, 2: 2e6})

