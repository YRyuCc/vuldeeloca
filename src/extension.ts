import * as vscode from "vscode";
import { applyFindings, clearDiagnostics } from "./diagnostics";
import { runDemo, scanFile, trainModel } from "./bridge";
import { VulDeePanel } from "./panel";

const SUPPORTED_LANGUAGES = new Set(["c", "cpp"]);

function isSupported(document: vscode.TextDocument): boolean {
  return SUPPORTED_LANGUAGES.has(document.languageId);
}

function findTargetEditor(): vscode.TextEditor | undefined {
  const active = vscode.window.activeTextEditor;
  if (active) {
    return active;
  }
  // When the webview panel has focus there is no "active" editor, so fall
  // back to the first visible C/C++ editor.
  return vscode.window.visibleTextEditors.find((e) => isSupported(e.document));
}

function configuredModelDir(): string {
  const dir = vscode.workspace.getConfiguration("vuldee").get<string>("modelDir", "");
  if (!dir.trim()) {
    throw new Error(
      "VulDee: no model directory configured. Set `vuldee.modelDir` (run `VulDee: Generate demo data...` first)."
    );
  }
  return dir.trim();
}

export function activate(context: vscode.ExtensionContext): void {
  const diagnostics = vscode.languages.createDiagnosticCollection("vuldee");
  context.subscriptions.push(diagnostics);

  const scan = async (document: vscode.TextDocument): Promise<void> => {
    if (!isSupported(document)) {
      return;
    }
    if (document.isDirty) {
      await document.save();
    }
    try {
      const result = await scanFile(document.uri.fsPath, configuredModelDir());
      applyFindings(diagnostics, document, result);
      const count = result.findings.length;
      vscode.window.setStatusBarMessage(
        `VulDee: ${count} potential vulnerable line${count === 1 ? "" : "s"} found.`,
        5000
      );
    } catch (err) {
      clearDiagnostics(diagnostics, document.uri);
      vscode.window.showErrorMessage(err instanceof Error ? err.message : String(err));
    }
  };

  context.subscriptions.push(
    vscode.commands.registerCommand("vuldee.scanFile", async () => {
      const editor = findTargetEditor();
      if (!editor) {
        vscode.window.showWarningMessage("VulDee: 没有打开可扫描的 C/C++ 文件。");
        return;
      }
      if (!isSupported(editor.document)) {
        vscode.window.showWarningMessage("VulDee: 只支持 C/C++ 文件。");
        return;
      }
      await scan(editor.document);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("vuldee.openPanel", () => {
      new VulDeePanel(context.extensionUri);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("vuldee.demo", async () => {
      const modelDir = configuredModelDir();
      await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: "VulDee: training demo model..." },
        async () => {
          try {
            await runDemo(modelDir);
            vscode.window.showInformationMessage(
              `VulDee: demo model saved to ${modelDir}`
            );
          } catch (err) {
            vscode.window.showErrorMessage(err instanceof Error ? err.message : String(err));
          }
        }
      );
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("vuldee.trainModel", async () => {
      const corpus = await vscode.window.showInputBox({
        prompt: "Path to the corpus pkl file",
      });
      const w2v = await vscode.window.showInputBox({
        prompt: "Path to the trained word2vec model",
      });
      if (!corpus || !w2v) {
        return;
      }
      await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: "VulDee: training model..." },
        async () => {
          try {
            await trainModel(corpus, w2v, configuredModelDir());
            vscode.window.showInformationMessage("VulDee: training finished.");
          } catch (err) {
            vscode.window.showErrorMessage(err instanceof Error ? err.message : String(err));
          }
        }
      );
    })
  );

  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument(async (document) => {
      const scanOnSave = vscode.workspace
        .getConfiguration("vuldee")
        .get<boolean>("scanOnSave", false);
      if (scanOnSave) {
        await scan(document);
      }
    })
  );
}

export function deactivate(): void {}
