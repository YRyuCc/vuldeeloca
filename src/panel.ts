import * as cp from "child_process";
import * as vscode from "vscode";
import { resolvePython, spawnVuldee } from "./bridge";

interface VulDeeConfig {
  pythonPath: string;
  modelDir: string;
  corpusPath: string;
  w2vPath: string;
  epochs: number;
  maxlen: number;
  vectorDim: number;
  scanOnSave: boolean;
}

function readConfig(): VulDeeConfig {
  const cfg = vscode.workspace.getConfiguration("vuldee");
  return {
    pythonPath: cfg.get("pythonPath", ""),
    modelDir: cfg.get("modelDir", ""),
    corpusPath: cfg.get("corpusPath", ""),
    w2vPath: cfg.get("w2vPath", ""),
    epochs: cfg.get("epochs", 4),
    maxlen: cfg.get("maxlen", 900),
    vectorDim: cfg.get("vectorDim", 30),
    scanOnSave: cfg.get("scanOnSave", false),
  };
}

async function writeConfig(config: VulDeeConfig): Promise<void> {
  const cfg = vscode.workspace.getConfiguration("vuldee");
  await Promise.all([
    cfg.update("pythonPath", config.pythonPath, vscode.ConfigurationTarget.Global),
    cfg.update("modelDir", config.modelDir, vscode.ConfigurationTarget.Global),
    cfg.update("corpusPath", config.corpusPath, vscode.ConfigurationTarget.Global),
    cfg.update("w2vPath", config.w2vPath, vscode.ConfigurationTarget.Global),
    cfg.update("epochs", config.epochs, vscode.ConfigurationTarget.Global),
    cfg.update("maxlen", config.maxlen, vscode.ConfigurationTarget.Global),
    cfg.update("vectorDim", config.vectorDim, vscode.ConfigurationTarget.Global),
    cfg.update("scanOnSave", config.scanOnSave, vscode.ConfigurationTarget.Global),
  ]);
}

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export class VulDeePanel {
  private readonly panel: vscode.WebviewPanel;
  private child: cp.ChildProcess | undefined;

  constructor(extensionUri: vscode.Uri) {
    this.panel = vscode.window.createWebviewPanel(
      "vuldee.panel",
      "VulDee 训练与配置",
      vscode.ViewColumn.One,
      { enableScripts: true }
    );
    this.panel.webview.html = this.getHtml();
    this.panel.webview.onDidReceiveMessage(
      (msg) => this.onMessage(msg),
      undefined
    );
    this.panel.onDidDispose(() => this.child?.kill());
  }

  private getHtml(): string {
    const cfg = readConfig();
    return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); padding: 16px; color: var(--vscode-foreground); }
  .field { margin-bottom: 10px; }
  label { display: block; font-size: 12px; margin-bottom: 3px; opacity: .85; }
  input, select { width: 100%; box-sizing: border-box; padding: 5px 8px; background: var(--vscode-input-background); color: var(--vscode-input-foreground); border: 1px solid var(--vscode-input-border); }
  .row { display: flex; gap: 10px; }
  .row .field { flex: 1; }
  button { padding: 6px 14px; margin-right: 8px; margin-bottom: 10px; background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; cursor: pointer; }
  button.secondary { background: var(--vscode-button-secondaryBackground); }
  #log { background: var(--vscode-textCodeBlock-background); font-family: var(--vscode-editor-font-family); font-size: 12px; padding: 10px; height: 260px; overflow-y: auto; white-space: pre-wrap; }
  #bar { height: 4px; background: var(--vscode-progressBar-background); width: 0%; transition: width .2s; margin-bottom: 8px; }
</style>
</head>
<body>
  <h3>VulDee 训练与配置</h3>
  <div class="row">
    <div class="field"><label>Python 解释器（已安装 vuldee）</label><input id="pythonPath" value="${esc(cfg.pythonPath)}" placeholder="留空则用 PATH 里的 python"></div>
    <div class="field"><label>模型目录</label><input id="modelDir" value="${esc(cfg.modelDir)}" placeholder="locator.pt / word2vec.model 所在目录"></div>
  </div>
  <div class="row">
    <div class="field"><label>语料 pkl（train 用）</label><input id="corpusPath" value="${esc(cfg.corpusPath)}"></div>
    <div class="field"><label>word2vec 模型（train 用）</label><input id="w2vPath" value="${esc(cfg.w2vPath)}"></div>
  </div>
  <div class="row">
    <div class="field"><label>epochs</label><input id="epochs" type="number" value="${cfg.epochs}"></div>
    <div class="field"><label>maxlen</label><input id="maxlen" type="number" value="${cfg.maxlen}"></div>
    <div class="field"><label>vector dim</label><input id="vectorDim" type="number" value="${cfg.vectorDim}"></div>
    <div class="field"><label>保存时扫描</label><input id="scanOnSave" type="checkbox" ${cfg.scanOnSave ? "checked" : ""}></div>
  </div>
  <div>
    <button id="save">保存配置</button>
    <button id="demo" class="secondary">生成演示模型</button>
    <button id="train" class="secondary">训练真实语料</button>
    <button id="pick" class="secondary">选择文件并扫描</button>
    <button id="scan" class="secondary">扫描当前文件</button>
  </div>
  <div id="bar"></div>
  <div id="log">就绪。Python: ${esc(resolvePython())}</div>
<script>
const vscode = acquireVsCodeApi();
const $ = (id) => document.getElementById(id);
const log = (text) => { const el = $('log'); el.textContent += text; el.scrollTop = el.scrollHeight; };
const fields = () => ({
  type: 'save',
  pythonPath: $('pythonPath').value,
  modelDir: $('modelDir').value,
  corpusPath: $('corpusPath').value,
  w2vPath: $('w2vPath').value,
  epochs: parseInt($('epochs').value || '4', 10),
  maxlen: parseInt($('maxlen').value || '900', 10),
  vectorDim: parseInt($('vectorDim').value || '30', 10),
  scanOnSave: $('scanOnSave').checked
});
$('save').onclick = () => vscode.postMessage(fields());
$('demo').onclick = () => vscode.postMessage({ type: 'demo' });
$('train').onclick = () => vscode.postMessage({ type: 'train' });
$('pick').onclick = () => vscode.postMessage({ type: 'pickAndScan' });
$('scan').onclick = () => vscode.postMessage({ type: 'scan' });
window.addEventListener('message', (event) => {
  const m = event.data;
  if (m.type === 'log') log(m.text);
  else if (m.type === 'done') log('\\n' + m.text + '\\n');
  else if (m.type === 'progress') { $('bar').style.width = (m.percent || 0) + '%'; }
});
</script>
</body>
</html>`;
  }

  private appendLog(text: string): void {
    this.panel.webview.postMessage({ type: "log", text });
  }

  private finish(text: string): void {
    this.panel.webview.postMessage({ type: "progress", percent: 100 });
    this.panel.webview.postMessage({ type: "done", text });
    this.child = undefined;
  }

  private run(args: string[]): void {
    this.child?.kill();
    this.panel.webview.postMessage({ type: "progress", percent: 10 });
    this.child = spawnVuldee(args, {
      onStdout: (chunk) => this.appendLog(chunk),
      onStderr: (chunk) => this.appendLog(chunk),
      onExit: (code) => {
        this.finish(code === 0 ? "完成 ✔" : `结束（退出码 ${code}）`);
      },
    });
  }

  private async onMessage(msg: any): Promise<void> {
    if (msg.type === "save") {
      writeConfig(msg as VulDeeConfig).then(() =>
        this.finish("配置已保存 ✔")
      );
      return;
    }
    const modelDir = msg.modelDir || readConfig().modelDir;
    if (!modelDir) {
      this.finish("请先填写模型目录（modelDir）");
      return;
    }
    if (msg.type === "demo") {
      const epochs = msg.epochs ?? readConfig().epochs;
      this.run(["demo", "--out-dir", modelDir, "--epochs", String(epochs)]);
    } else if (msg.type === "train") {
      const corpus = msg.corpusPath || readConfig().corpusPath;
      const w2v = msg.w2vPath || readConfig().w2vPath;
      const epochs = msg.epochs ?? readConfig().epochs;
      const maxlen = msg.maxlen ?? readConfig().maxlen;
      if (!corpus || !w2v) {
        this.finish("训练需要填写语料 pkl 和 word2vec 模型路径");
        return;
      }
      this.run([
        "train", "--corpus", corpus, "--w2v", w2v,
        "--out-dir", modelDir, "--epochs", String(epochs), "--maxlen", String(maxlen),
      ]);
    } else if (msg.type === "scan") {
      vscode.commands.executeCommand("vuldee.scanFile");
    } else if (msg.type === "pickAndScan") {
      const uris = await vscode.window.showOpenDialog({
        canSelectMany: false,
        title: "选择要扫描的 C/C++ 文件",
        filters: { "C/C++ 源码": ["c", "h", "cpp", "cc", "cxx", "hpp"] },
      });
      if (uris && uris.length > 0) {
        const document = await vscode.workspace.openTextDocument(uris[0]);
        await vscode.window.showTextDocument(document);
        vscode.commands.executeCommand("vuldee.scanFile");
      }
    }
  }
}
