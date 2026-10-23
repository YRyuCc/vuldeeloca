import * as cp from "child_process";
import * as vscode from "vscode";

export function resolvePython(): string {
  const configured = vscode.workspace
    .getConfiguration("vuldee")
    .get<string>("pythonPath", "");
  if (configured.trim()) {
    return configured.trim();
  }
  return process.platform === "win32" ? "python" : "python3";
}

export interface StreamHandlers {
  onStdout: (chunk: string) => void;
  onStderr: (chunk: string) => void;
  onExit: (code: number | null) => void;
}

export function spawnVuldee(args: string[], handlers: StreamHandlers): cp.ChildProcess {
  const python = resolvePython();
  const child = cp.spawn(python, ["-u", "-m", "vuldee.cli", ...args]);
  child.stdout.on("data", (d: Buffer) => handlers.onStdout(d.toString()));
  child.stderr.on("data", (d: Buffer) => handlers.onStderr(d.toString()));
  child.on("close", (code) => handlers.onExit(code));
  return child;
}

export interface Finding {
  line: number;
  type: string;
  name: string;
  score: number;
}

export interface ScanResult {
  file: string;
  findings: Finding[];
}

function run(args: string[], timeoutMs = 120000): Promise<string> {
  const python = resolvePython();
  return new Promise<string>((resolve, reject) => {
    cp.execFile(
      python,
      ["-m", "vuldee.cli", ...args],
      { timeout: timeoutMs, maxBuffer: 64 * 1024 * 1024 },
      (error, stdout, stderr) => {
        if (error) {
          const detail = stderr?.trim() || error.message;
          reject(new Error(`VulDee backend failed: ${detail}`));
          return;
        }
        resolve(stdout);
      }
    );
  });
}

function extractJson(stdout: string): unknown {
  const start = stdout.indexOf("{");
  const end = stdout.lastIndexOf("}");
  if (start === -1 || end === -1) {
    throw new Error("VulDee backend did not return JSON: " + stdout.trim());
  }
  return JSON.parse(stdout.slice(start, end + 1));
}

export async function scanFile(
  filePath: string,
  modelDir: string
): Promise<ScanResult> {
  const stdout = await run(["scan", "--file", filePath, "--model-dir", modelDir]);
  return extractJson(stdout) as ScanResult;
}

export async function trainModel(
  corpus: string,
  w2v: string,
  modelDir: string
): Promise<string> {
  return run([
    "train",
    "--corpus",
    corpus,
    "--w2v",
    w2v,
    "--out-dir",
    modelDir,
  ]);
}

export async function runDemo(modelDir: string): Promise<string> {
  return run(["demo", "--out-dir", modelDir]);
}
