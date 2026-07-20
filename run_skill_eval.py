#!/usr/bin/env python3
"""
Skill Quality Lab Runner — SKILL.md kalite ve göreve uygunluk kontrolü.

Kullanım:
    # Sadece statik lint (LLM'siz, anında):
    python run_skill_eval.py --skill path/to/SKILL.md

    # Lint + task-fit judge + birleşik skor:
    python run_skill_eval.py --skill path/to/SKILL.md \
        --task "Haftalık satış CSV'sinden bölge raporu üret" --model demo-model
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from api.services.skill_eval_service import SkillEvalService


def main() -> int:
    parser = argparse.ArgumentParser(description="SKILL.md quality + task-fit evaluation")
    parser.add_argument("--skill", required=True, help="Path to a SKILL.md file")
    parser.add_argument("--task", default=None, help="Task description (enables the fit judge)")
    parser.add_argument("--model", default=None, help="Judge model key from config/models.yaml")
    parser.add_argument("--output", default=None, help="Output JSON path (default: reports/skill_eval_<ts>.json)")
    parser.add_argument("--no-save", action="store_true", help="Do not persist a report")
    args = parser.parse_args()

    if args.task and not args.model:
        raise SystemExit("--task requires --model (judge model key)")

    skill_text = Path(args.skill).read_text(encoding="utf-8")
    service = SkillEvalService()

    if args.task:
        report = service.full(skill_text, args.task, args.model, save=not args.no_save and not args.output)
        lint = report["lint"]
        fit = report["fit"]
        print(f"\nSkill    : {args.skill} (name: {lint['summary']['name']})")
        print(f"Lint     : {lint['score']}/100 "
              f"({lint['summary']['errors']} error, {lint['summary']['warnings']} warning, "
              f"{lint['summary']['security_flags']} security)")
        if fit:
            print(f"Task fit : {fit['overall']} — {fit['verdict']}")
            for name, criterion in fit["criteria"].items():
                print(f"  - {name}: {criterion['score']}")
            for gap in fit["gaps"]:
                print(f"  gap: {gap}")
        else:
            print("Task fit : UNUSABLE (judge parse failed) — combined is lint-only")
        print(f"Combined : {report['combined_score']} ({report['combined_basis']})")
    else:
        report = service.lint(skill_text)
        print(f"\nSkill : {args.skill} (name: {report['summary']['name']})")
        print(f"Lint  : {report['score']}/100")
        for check in report["checks"]:
            if not check["passed"]:
                print(f"  [{check['severity']}] {check['id']}: {check['message']}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved: {args.output}")
    elif not args.no_save and args.task and report.get("report_path"):
        print(f"\nReport saved: {report['report_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
