import { spawn, execFile } from 'child_process';
import { join } from 'path';
import { is } from '@electron-toolkit/utils';

const PYTHON_PATH = 'python'; // Dev only, local environment dependent
const SCRIPT_PATH = join(__dirname, '../../resources/backend/app.py');
const EXE_PATH = join(__dirname, '../../resources/backend/app.exe');

export function runPython(args: string[]): Promise<string> {
  if (is.dev) {
    return runScript(args);
  } else {
    return runExecutable(args);
  }
}

function runScript(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const subprocess = spawn(PYTHON_PATH, [SCRIPT_PATH, ...args], {
      shell: true, // Ensures compatibility across environments
    });

    let output = '';
    subprocess.stdout.on('data', (data) => {
      output += data.toString();
    });

    subprocess.stderr.on('data', (data) => {
      console.error('Python stderr:', data.toString());
    });

    subprocess.on('close', (code) => {
      if (code === 0) resolve(output);
      else reject(new Error(`Python script exited with code ${code}, path: ${SCRIPT_PATH}`));
    });
  });
}

function runExecutable(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(EXE_PATH, args, (error, stdout, stderr) => {
      if (error) {
        console.error('Executable error:', stderr);
        return reject(error);
      }
      resolve(stdout);
    });
  });
}
