import { contextBridge, ipcRenderer } from 'electron';
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

// Python APIs
const api = {
  runPython: (args: string[]) => ipcRenderer.invoke('run-python', args),
  resolveFilePath: (filename: string) => ipcRenderer.invoke('resolve-file-path', filename),
};

// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI);
    contextBridge.exposeInMainWorld('api', api);
  } catch (error) {
    console.error(error);
  }
} else {
  window.electron = electronAPI;
  window.api = api;
}
