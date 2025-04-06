import os
import grpc
from concurrent import futures
import time

from proto.schema_pb2_grpc import (
    MicroApiServicer,
    add_MicroApiServicer_to_server,
)
from proto.schema_pb2 import GetResponse, GetRequest


class Service(MicroApiServicer):
    def Get(self, req: GetRequest, context):
        print(f"Received message: {req.message}")
        res = GetResponse(message=f"{req.message} -> Hello from python")
        return res


def run_server():
    port = os.getenv("PORT", "3000")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_MicroApiServicer_to_server(Service(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Server started on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    run_server()
