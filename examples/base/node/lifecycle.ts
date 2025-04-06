export const runForever = (): Promise<void> => {
  return new Promise<void>((resolve) => {
    process.on("SIGINT", () => {
      console.log("SIGINT");
      resolve();
    });
    process.on("SIGTERM", () => {
      console.log("SIGTERM");
      resolve();
    });
  });
};

export const run = async (delegate: () => Promise<any>): Promise<any> => {
  process.on("exit", () => {
    console.log("Process exited");
  });

  process.on("unhandledRejection", (reason, promise) => {
    console.trace("Unhandled Rejection at:", promise, "reason:", reason);
    process.exit(1);
  });

  process.on("uncaughtException", (error) => {
    console.trace("Uncaught Exception:", error);
    process.exit(1);
  });

  process.on("SIGINT", () => {
    console.log("SIGINT");
    process.exit(0);
  });

  await delegate();
};
