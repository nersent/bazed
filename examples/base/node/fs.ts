import { mkdir as _mkdir, stat } from "node:fs/promises";

export const exists = async (path: string): Promise<boolean> => {
  try {
    await stat(path);
  } catch (err) {
    return false;
  }
  return true;
};
