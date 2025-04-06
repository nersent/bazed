import { IsString } from "class-validator";

export class PingRequest {
  @IsString()
  message!: string;
}

export interface PingResponse {
  message: string;
}
