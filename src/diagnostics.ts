import * as vscode from "vscode";
import { ScanResult } from "./bridge";

export function applyFindings(
  collection: vscode.DiagnosticCollection,
  document: vscode.TextDocument,
  result: ScanResult
): void {
  const diagnostics: vscode.Diagnostic[] = result.findings.map((finding) => {
    const line = Math.max(0, finding.line - 1);
    const range = new vscode.Range(line, 0, line, 0);
    const name = finding.name ? ` (${finding.name})` : "";
    const message = `Possible ${finding.type} vulnerability${name} — score ${finding.score}`;
    const diagnostic = new vscode.Diagnostic(
      range,
      message,
      vscode.DiagnosticSeverity.Warning
    );
    diagnostic.source = "VulDee";
    return diagnostic;
  });
  collection.set(document.uri, diagnostics);
}

export function clearDiagnostics(
  collection: vscode.DiagnosticCollection,
  uri: vscode.Uri
): void {
  collection.delete(uri);
}
