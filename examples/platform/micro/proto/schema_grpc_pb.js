// GENERATED CODE -- DO NOT EDIT!

'use strict';
var grpc = require('@grpc/grpc-js');
var schema_pb = require('./schema_pb.js');

function serialize_micro_GetRequest(arg) {
  if (!(arg instanceof schema_pb.GetRequest)) {
    throw new Error('Expected argument of type micro.GetRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_micro_GetRequest(buffer_arg) {
  return schema_pb.GetRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_micro_GetResponse(arg) {
  if (!(arg instanceof schema_pb.GetResponse)) {
    throw new Error('Expected argument of type micro.GetResponse');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_micro_GetResponse(buffer_arg) {
  return schema_pb.GetResponse.deserializeBinary(new Uint8Array(buffer_arg));
}


var MicroApiService = exports.MicroApiService = {
  get: {
    path: '/micro.MicroApi/Get',
    requestStream: false,
    responseStream: false,
    requestType: schema_pb.GetRequest,
    responseType: schema_pb.GetResponse,
    requestSerialize: serialize_micro_GetRequest,
    requestDeserialize: deserialize_micro_GetRequest,
    responseSerialize: serialize_micro_GetResponse,
    responseDeserialize: deserialize_micro_GetResponse,
  },
};

exports.MicroApiClient = grpc.makeGenericClientConstructor(MicroApiService);
