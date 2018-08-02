import grpc
import nucleai_pb2 as pb
import nucleai_pb2_grpc as pb_grpc
import time


class Client():

    def __init__(self):
        self.channel = grpc.insecure_channel("localhost:1453")
        self.stub = pb_grpc.LearningStub(self.channel)
        self.id = "client_" + str(time.time())

    def get_client_id(self):
        return pb.ClientId(txt=self.id)

    def upload(self, cycle_id, deltas):
        return self.stub.Upload(pb.UploadRequest(
            cycleId=pb.CycleId(num=cycle_id),
            clientId=self.get_client_id(),
            deltas=deltas,
            ))

    def download(self):
        return self.stub.Download(pb.DownloadRequest(clientId=self.get_client_id()))


if __name__ == "__main__":
    client = Client()
    dl = client.download()
    print(dl)
    print(client.upload(
        dl.cycleId.num,
        pb.Parameters(parameters=[
            pb.IndexedValue(index=1, value=1e6),
            pb.IndexedValue(index=2, value=1e6),
            ]
            )))

