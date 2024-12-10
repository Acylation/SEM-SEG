import { join } from 'path';

const { app } = require('electron');

export function resolveFilePath(filename: string) {
  return join(app.getAppPath(), filename);
}
