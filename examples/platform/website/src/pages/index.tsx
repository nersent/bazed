import axios, { AxiosResponse } from "axios";
import { useCallback } from "react";

import { Button } from "~/base/ui/button";
import * as api from "~/platform/common/schema";

const API_URL = process.env["API_URL"];
console.log(API_URL);

export default function Home() {
  const onClick = useCallback(async () => {
    const res = await axios.post<
      api.PingResponse,
      AxiosResponse<api.PingResponse>,
      api.PingRequest
    >(`${API_URL}/ping`, {
      message: `ping ${new Date().toString()}`,
    });
    alert(res.data.message);
  }, []);

  return (
    <div>
      <Button onClick={onClick}>Click me</Button>
    </div>
  );
}
