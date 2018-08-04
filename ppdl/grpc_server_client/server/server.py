import threading
import time
import os
import grpc
import traceback
import nucleai_pb2 as pb
import nucleai_pb2_grpc as pb_grpc
from logger import singleton as log
from concurrent import futures
import random
from database import Database
from collections import deque, defaultdict


ADMIN_USER = "admin"

PHASE_NONE = 0
PHASE_TRAIN = 1
PHASE_UPDATE = 2

class Learner:

    class Exception(Exception):
        pass

    # private methods are run inside a separate thread
    # public methods interact with that thread safely

    # I use Python native arrays here
    #   TODO optimize later if needed 

    def __init__(self, initial_parameters_f, starting_cycle_time=15):
        self.initial_parameters_f = initial_parameters_f
        self.dropout_ratio = 0.25 # fraction of params selected for download and upload

        self.db = Database()
        self.job_id = None
        self.cycle_id = None
        self.cycle_time = starting_cycle_time
        self.clock = None
        self.phase = PHASE_NONE
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.daemon = True # so the thread halts if the parent halts

        # TODO for now we always create a new job here
        self._create_job()

    def _create_job(self):
        self.job_id = self.db.execute("""
        INSERT INTO ppdl.job (id) VALUES (DEFAULT) RETURNING id;
        """, returning=True)
        log.debug("created job with id {}".format(self.job_id))
        self.db.commit()

    def _create_cycle(self):
        assert self.job_id is not None
        self.cycle_id = self.db.execute("""
        INSERT INTO ppdl.cycle(job_id, starts_at, finishes_at) (
            SELECT %s, now(), now() + interval '{}' sec
        ) RETURNING id
        """.format(self.cycle_time), (self.job_id,), returning=True)
        log.debug("created cycle with id {}".format(self.cycle_id))
        self.db.commit()

    def _add_parameters(self, parameters, user_id):
        # create the upload row
        upload_id = self.db.execute("""
        INSERT INTO ppdl.parameters_upload (cycle_id, user_id) (
            SELECT %s, %s
        ) RETURNING id
        """, (self.cycle_id, user_id), returning=True)
        # create the parameter rows
        dimensions, values = zip(*parameters.items())
        self.db.execute("""
        INSERT INTO ppdl.parameter(upload_id, dimension, value) (
            SELECT %s, d, v 
            FROM unnest((%s)::int[], (%s)::float8[]) params(d, v)
        )
        """, (upload_id, list(dimensions), list(values)))
        self.db.commit()

    def _get_last_parameters(self):
        # on the fly summation; TODO maybe cache
        rows = self.db.query("""
        SELECT p.dimension, sum(p.value)
        FROM ppdl.parameter p, ppdl.parameters_upload u, ppdl.cycle c
        WHERE p.upload_id = u.id
        AND u.cycle_id < %s
        AND c.id = u.cycle_id
        AND c.job_id = %s
        GROUP BY p.dimension
        """, (self.cycle_id, self.job_id))
        self.db.commit()
        assert rows
        return {r[0]: r[1] for r in rows}

    def _train_phase(self):
        self.phase = PHASE_TRAIN

        while self.clock > 0:
            time.sleep(1)
            self.clock -= 1

    def _update_phase(self):
        self.phase = PHASE_UPDATE
        # nothing else to do here yet...	

    def _run_loop(self):
        # set initial parameters in first cycle
        self._create_cycle()
        parameters = self.initial_parameters_f()
        self._add_parameters(parameters, ADMIN_USER)

        while True:
            log.debug("Starting cycle {}".format(self.cycle_id))
            self._create_cycle()
            self.clock = self.cycle_time
            log.info("Entering download/train phase")
            self._train_phase()
            log.info("Entering update phase")
            self._update_phase() 

    def start(self):
        self.thread.start()

    def download(self, request):
        log.debug("Client {} downloading".format(request.clientId.txt))
        if self.phase != PHASE_TRAIN: # TODO maybe only allow downloads if enough time remaining
            raise self.Exception("cannot download except in the training phase")
        # download a random subset of the parameters
        parameters_l = self._get_last_parameters()
        parameters = [pb.IndexedValue(index=i, value=val) for i, val in random.sample(list(parameters_l.items()), int(len(parameters_l) * self.dropout_ratio))]
        log.debug("all parameters = {}, parameters being downloaded = {}".format(parameters_l, parameters))
        parameters = pb.Parameters(parameters=parameters)
        return pb.DownloadResponse(
                cycleId=pb.CycleId(num=self.cycle_id),
                parameters=parameters,
                )

    def upload(self, request):
        log.debug("Client {} uploading".format(request.clientId.txt))
        # TODO don't let a client upload twice per cycle
        if self.phase != PHASE_TRAIN:
            raise self.Exception("cannot upload except in the training phase")
        if self.cycle_id != request.cycleId.num:
            raise self.Exception("upload is for cycle {}, but server is on cycle {}"
                    .format(request.cycleId.num, self.cycle_id))
        parameters = {p.index: p.value for p in request.deltas.parameters}
        self._add_parameters(parameters, request.clientId.txt)
        return pb.UploadResponse()


global_learner = None

class LearningServicer(pb_grpc.LearningServicer):

    def __init__(self):
        pass

    def _try(self, f, context):
        try:
            ret = f()
            return ret
        except Exception as e:
            if isinstance(e, Learner.Exception):
                # bad input, not our fault; send back the exception for the client to see
                log.warn("Invalid input from client, exception: {}".format(e))
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(repr(e))
                return pb.Void()
            else:
                # our fault; log the exception, but don't send it to the client
                log.error("unexpected exception: {}".format(e))
                context.set_code(grpc.StatusCode.ABORTED)
                context.set_details(repr("internal error"))
                return pb.Void()


    # service endpoints, which shall just pass the request on to the parameter server

    def Download(self, request, context):
        def f():
            return global_learner.download(request)
        return self._try(f, context)

    def Upload(self, request, context):
        def f():
            return global_learner.upload(request)
        return self._try(f, context)


def run():

    # start the parameter server
    global global_learner
    initial_parameters_f = (lambda: {i: random.random() for i in range(1, 11)}) # TODO temp
    global_learner = Learner(initial_parameters_f=initial_parameters_f)
    global_learner.start()

    # start the gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_grpc.add_LearningServicer_to_server(LearningServicer(), server)
    server.add_insecure_port('[::]:{}'.format(os.getenv("PORT") or 1453))
    server.start()
    try:
        while True: # otherwise the script just exits
            print("tick")
            time.sleep(600)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    run()
