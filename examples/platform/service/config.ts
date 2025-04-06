import { Injectable } from "@nestjs/common";
import { ConfigService as NestConfigService } from "@nestjs/config";
import { config } from "dotenv";
import Joi from "joi";

export const ENV_SCHEMA = Joi.object({
  PORT: Joi.number().required(),
  MICRO_SERVICE_URL: Joi.string().required(),
});

@Injectable()
export class Config {
  constructor(private readonly env: NestConfigService) {
    config();
  }

  public get port(): number {
    return this.env.getOrThrow("PORT", { infer: true });
  }

  public get microServiceUrl(): string {
    return this.env.getOrThrow("MICRO_SERVICE_URL");
  }
}
