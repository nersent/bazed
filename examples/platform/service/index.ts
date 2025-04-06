import { resolve } from "path";

import { Module, ValidationPipe } from "@nestjs/common";
import { ConfigModule } from "@nestjs/config";
import { NestFactory } from "@nestjs/core";
import {
  FastifyAdapter,
  NestFastifyApplication,
} from "@nestjs/platform-fastify";

import { ApiController } from "./api_controller";
import { Config, ENV_SCHEMA } from "./config";

import { mbToBytes } from "~/base/js/bytes";
import { createLogger } from "~/base/nest/logger";

@Module({
  imports: [
    ConfigModule.forRoot({
      envFilePath: [resolve(".env.test"), resolve(".env")],
      cache: true,
      validationSchema: ENV_SCHEMA,
    }),
  ],
  controllers: [ApiController],
  providers: [Config],
})
export class AppModule {}

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type
export const runServer = async () => {
  const adapter = new FastifyAdapter({ bodyLimit: mbToBytes(256) });

  const app = await NestFactory.create<NestFastifyApplication>(
    AppModule,
    adapter,
    {
      rawBody: true,
      logger: createLogger({
        path: resolve(process.env["OUT_PATH"] ?? "", "service.log"),
      }),
    },
  );

  const config = app.get(Config);
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      transform: true,
      transformOptions: { enableImplicitConversion: true },
    }),
  );
  app.enableCors({
    origin: true,
    credentials: true,
  });

  let restAddress: string | undefined;
  await app.listen(config.port, "0.0.0.0", (err, address) => {
    if (err) throw err;
    console.log(`Rest API Listening on ${address}`);
    restAddress = address;
  });
  restAddress = restAddress!;

  return { app, restAddress };
};
