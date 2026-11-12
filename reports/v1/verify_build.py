"""Check the actual XeLaTeX log and PDF, retaining a machine-readable result."""
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parent
log = (root / "report.log").read_text(encoding="utf-8", errors="replace")
patterns = {
    "errors": r"(?m)^!|(?:LaTeX|Package \S+|Class \S+) Error:|Undefined control sequence|Emergency stop|Fatal error|(?m:^.*\.(?:tex|sty):\d+:)",
    "missing_characters": r"Missing character:",
    "unresolved_references": r"(?:Reference|Citation) .+ undefined|There were undefined|Rerun to get|Label\(s\) may have changed|Please rerun",
    "overfull_boxes": r"Overfull \\[hv]box",
    "warnings": r"(?m)^.*Warning:.*$",
    "underfull_boxes": r"Underfull \\[hv]box",
}
findings = {name: re.findall(pattern, log) for name, pattern in patterns.items()}
pdf = root / "report.pdf"
output = re.search(r"Output written on report\.pdf \((\d+) pages", log)
success = pdf.is_file() and output is not None and not any(findings.values())
summary = {
    "success": success,
    "engine": "XeLaTeX in WSL",
    "pages": int(output.group(1)) if output else None,
    "pdf_bytes": pdf.stat().st_size if pdf.exists() else 0,
    "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest() if pdf.exists() else None,
    "findings": findings,
}
(root / "build_verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
sys.exit(0 if success else 1)
