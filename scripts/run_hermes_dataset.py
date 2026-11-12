"""Run the installed dataset plugin through the configured Hermes model."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]

def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hermes")
    parser.add_argument("--max-turns", type=int, default=240)
    parser.add_argument("--resume", help="Hermes conversation ID, not dataset job ID")
    args = parser.parse_args()
    executable = args.hermes or shutil.which("hermes")
    if not executable:
        fallback = Path.home()/".local/bin/hermes"
        executable = str(fallback) if fallback.is_file() else None
    if not executable:
        parser.error("Supply --hermes or add Hermes CLI to PATH")
    os.umask(0o077)
    prompt = args.prompt.expanduser().resolve(strict=True)
    out = args.output.expanduser().absolute()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    manual = ROOT/"skills/isaac-image-dataset/SKILL.md"
    effective_prompt = out/"prompt-with-manual.md"
    effective_prompt.write_text(manual.read_text()+"\n\n## User request\n\n"+prompt.read_text())
    command = [executable, "chat", "--oneshot", "--format", "stream-json",
               "-t", "isaac-image-dataset",
               "--max-turns", str(args.max_turns), "--source", "tool", "--in", str(out),
               "--query-file", str(effective_prompt)]
    if args.resume:
        command += ["--resume", args.resume, "--no-restore-cwd"]
    metadata = {"started_utc": now(), "model": "host-configured", "natural_language_model_invoked": True,
                "command": command, "cli_returncode": None}
    (out/"execution.json").write_text(json.dumps(metadata, indent=2))
    with (out/"private-stream.jsonl").open("x") as stdout, (out/"private-stderr.log").open("x") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr)
    metadata.update(finished_utc=now(), cli_returncode=result.returncode)
    (out/"execution.json").write_text(json.dumps(metadata, indent=2))
    events = []
    for line in (out/"private-stream.jsonl").read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") in {"tool_use", "tool_result"}:
            events.append(event)
    (out/"tool-events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
