// package: micro
// file: schema.proto

/* tslint:disable */
/* eslint-disable */

import * as grpc from "@grpc/grpc-js";
import * as schema_pb from "./schema_pb";

interface IMicroApiService extends grpc.ServiceDefinition<grpc.UntypedServiceImplementation> {
    get: IMicroApiService_IGet;
}

interface IMicroApiService_IGet extends grpc.MethodDefinition<schema_pb.GetRequest, schema_pb.GetResponse> {
    path: "/micro.MicroApi/Get";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<schema_pb.GetRequest>;
    requestDeserialize: grpc.deserialize<schema_pb.GetRequest>;
    responseSerialize: grpc.serialize<schema_pb.GetResponse>;
    responseDeserialize: grpc.deserialize<schema_pb.GetResponse>;
}

export const MicroApiService: IMicroApiService;

export interface IMicroApiServer extends grpc.UntypedServiceImplementation {
    get: grpc.handleUnaryCall<schema_pb.GetRequest, schema_pb.GetResponse>;
}

export interface IMicroApiClient {
    get(request: schema_pb.GetRequest, callback: (error: grpc.ServiceError | null, response: schema_pb.GetResponse) => void): grpc.ClientUnaryCall;
    get(request: schema_pb.GetRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: schema_pb.GetResponse) => void): grpc.ClientUnaryCall;
    get(request: schema_pb.GetRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: schema_pb.GetResponse) => void): grpc.ClientUnaryCall;
}

export class MicroApiClient extends grpc.Client implements IMicroApiClient {
    constructor(address: string, credentials: grpc.ChannelCredentials, options?: Partial<grpc.ClientOptions>);
    public get(request: schema_pb.GetRequest, callback: (error: grpc.ServiceError | null, response: schema_pb.GetResponse) => void): grpc.ClientUnaryCall;
    public get(request: schema_pb.GetRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: schema_pb.GetResponse) => void): grpc.ClientUnaryCall;
    public get(request: schema_pb.GetRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: schema_pb.GetResponse) => void): grpc.ClientUnaryCall;
}
