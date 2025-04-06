import { promisify } from "util";

import * as grpc from "@grpc/grpc-js";
import { Controller, Post, Body } from "@nestjs/common";

import * as microApi from "../micro/proto/schema_pb";

import { Config } from "./config";

import * as api from "~/platform/common/schema";
import { MicroApiClient } from "~/platform/micro/proto/schema_grpc_pb";

@Controller("/")
export class ApiController {
  private readonly microClient: MicroApiClient;

  constructor(private readonly config: Config) {
    this.microClient = new MicroApiClient(
      this.config.microServiceUrl,
      grpc.credentials.createInsecure(),
    );
  }

  @Post("/ping")
  public async ping(@Body() body: api.PingRequest): Promise<api.PingResponse> {
    let subMessage = "";
    {
      const req = new microApi.GetRequest();
      req.setMessage(body.message);
      const res = (await promisify(this.microClient.get.bind(this.microClient))(
        req,
      )) as microApi.GetResponse;
      subMessage = res.getMessage();
    }
    return {
      message: `grpc: ${subMessage}`,
    };
  }
}
