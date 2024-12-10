import { electronAPI } from '@electron-toolkit/preload';

declare global {
  interface Window {
    electron: typeof electronAPI;
    api: {
      runPython: (args: string[]) => Promise<string>;
      resolveFilePath: (filename: string) => Promise<string>;
    };
  }
}
