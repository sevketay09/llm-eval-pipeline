"""
Skill Lint — static quality analysis for agent SKILL.md files.

Zero-dependency (stdlib + yaml, already a project dep). No LLM calls.

Checks are grouped in four areas, following the common skill-eval practice
(frontmatter/format spec per anthropics/skills skill-creator; description
banding and naming per static skill benchmarkers; security red-flag scan
per marketplace skill advisors):
  - format:   frontmatter presence/validity, name, description
  - body:     non-empty body, size/token estimate, dead sections
  - structure: progressive disclosure (references to bundled files)
  - security: pipe-to-shell, destructive commands, secret access, sudo

Public API:
    lint_skill(text)      -> {"score": int 0-100, "checks": [...], "summary": {...}}
    lint_skill_file(path) -> same, reads the file (utf-8)

Each check: {"id", "area", "severity": "error"|"warning"|"info",
             "passed": bool, "message": str}
Score: 100 minus 25 per failed error, 10 per failed warning (floor 0).
Info checks never affect the score.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Limits from the Agent Skills spec (anthropics/skills skill-creator).
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
# Description shorter than this rarely carries enough signal for triggering.
MIN_DESCRIPTION_LENGTH = 20
# Body budget: ~4 chars/token heuristic; beyond this the skill bloats context.
MAX_BODY_TOKEN_ESTIMATE = 5000

ERROR_PENALTY = 25
WARNING_PENALTY = 10

_KEBAB_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
# Markdown links or bare relative paths pointing at bundled resource files.
_BUNDLED_REF_RE = re.compile(
    r"(\]\(\.?/?[\w./-]+\)|\b[\w-]+/[\w./-]+\.(?:md|py|sh|js|ts|json|yaml|yml|txt)\b)"
)

_SECURITY_PATTERNS: List[Tuple[str, str, str]] = [
    # (check_id, regex, message)
    (
        "sec_pipe_to_shell",
        r"(curl|wget)[^\n|]*\|\s*(sudo\s+)?(ba)?sh\b",
        "Downloads and pipes remote content directly into a shell",
    ),
    (
        "sec_destructive_rm",
        r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\s+[/~]",
        "Recursive force-delete targeting root or home paths",
    ),
    (
        "sec_base64_exec",
        r"base64\s+(-d|--decode)[^\n]*\|\s*(ba)?sh\b",
        "Decodes base64 content and executes it",
    ),
    (
        "sec_sudo",
        r"\bsudo\s+\S",
        "Instructs the agent to run commands with sudo",
    ),
    (
        "sec_secret_access",
        r"(cat|source|less|head|tail)\s+[^\n]*(\.env\b|id_rsa\b|credentials\b|\.aws/|\.ssh/)",
        "Reads secret/credential files",
    ),
    (
        "sec_env_exfiltration",
        r"(printenv|env)\b[^\n]*\|\s*curl\b",
        "Pipes environment variables to a network call",
    ),
]

_SECURITY_SEVERITY = {
    "sec_pipe_to_shell": "error",
    "sec_destructive_rm": "error",
    "sec_base64_exec": "error",
    "sec_sudo": "warning",
    "sec_secret_access": "error",
    "sec_env_exfiltration": "error",
}


def _check(check_id: str, area: str, severity: str, passed: bool, message: str) -> Dict[str, Any]:
    return {
        "id": check_id,
        "area": area,
        "severity": severity,
        "passed": bool(passed),
        "message": message,
    }


def split_frontmatter(text: str) -> Tuple[Optional[str], str]:
    """Return (frontmatter_str or None, body). Frontmatter = leading --- block."""
    if not text.lstrip().startswith("---"):
        return None, text
    stripped = text.lstrip()
    parts = stripped.split("---", 2)
    # parts[0] == "" before the first ---; need a closing --- to count.
    if len(parts) < 3:
        return None, text
    return parts[1], parts[2]


def _parse_frontmatter(raw: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Return (dict or None, error message or None)."""
    if raw is None:
        return None, "missing"
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return None, f"invalid YAML: {exc}"
    if not isinstance(data, dict):
        return None, "frontmatter is not a mapping"
    return data, None


def estimate_tokens(text: str) -> int:
    """Cheap ~4-chars-per-token heuristic; enough for a size band check."""
    return max(0, round(len(text) / 4))


def _find_dead_sections(body: str) -> List[str]:
    """Headings with no content before the next heading (or EOF)."""
    dead = []
    matches = list(_HEADING_RE.finditer(body))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        if not body[start:end].strip():
            dead.append(match.group(2).strip())
    return dead


def _format_checks(frontmatter: Optional[Dict[str, Any]], fm_error: Optional[str]) -> List[Dict[str, Any]]:
    checks = []
    fm_ok = frontmatter is not None
    if fm_error == "missing":
        msg = "SKILL.md must start with a `---` YAML frontmatter block"
    elif fm_error:
        msg = f"Frontmatter could not be parsed ({fm_error})"
    else:
        msg = "Frontmatter present and valid YAML"
    checks.append(_check("frontmatter_valid", "format", "error", fm_ok, msg))

    name = (frontmatter or {}).get("name")
    name_str = str(name).strip() if name is not None else ""
    checks.append(
        _check(
            "name_present",
            "format",
            "error",
            bool(name_str),
            "Frontmatter `name` present" if name_str else "Frontmatter `name` is required",
        )
    )
    if name_str:
        checks.append(
            _check(
                "name_kebab_case",
                "format",
                "warning",
                bool(_KEBAB_RE.match(name_str)),
                f"`name` should be kebab-case (got: {name_str!r})"
                if not _KEBAB_RE.match(name_str)
                else "`name` is kebab-case",
            )
        )
        checks.append(
            _check(
                "name_length",
                "format",
                "warning",
                len(name_str) <= MAX_NAME_LENGTH,
                f"`name` length {len(name_str)} (max {MAX_NAME_LENGTH})",
            )
        )

    description = (frontmatter or {}).get("description")
    desc_str = str(description).strip() if description is not None else ""
    checks.append(
        _check(
            "description_present",
            "format",
            "error",
            bool(desc_str),
            "Frontmatter `description` present"
            if desc_str
            else "Frontmatter `description` is required — it drives skill triggering",
        )
    )
    if desc_str:
        checks.append(
            _check(
                "description_not_too_short",
                "format",
                "warning",
                len(desc_str) >= MIN_DESCRIPTION_LENGTH,
                f"`description` length {len(desc_str)} — too short to trigger reliably (min {MIN_DESCRIPTION_LENGTH})"
                if len(desc_str) < MIN_DESCRIPTION_LENGTH
                else f"`description` length {len(desc_str)} OK",
            )
        )
        checks.append(
            _check(
                "description_not_too_long",
                "format",
                "warning",
                len(desc_str) <= MAX_DESCRIPTION_LENGTH,
                f"`description` length {len(desc_str)} exceeds spec max {MAX_DESCRIPTION_LENGTH}"
                if len(desc_str) > MAX_DESCRIPTION_LENGTH
                else "`description` within spec length",
            )
        )
    return checks


def _body_checks(body: str) -> List[Dict[str, Any]]:
    checks = []
    body_str = body.strip()
    checks.append(
        _check(
            "body_present",
            "body",
            "error",
            bool(body_str),
            "Skill body present" if body_str else "Skill body is empty — no instructions to load",
        )
    )
    tokens = estimate_tokens(body_str)
    checks.append(
        _check(
            "body_size",
            "body",
            "warning",
            tokens <= MAX_BODY_TOKEN_ESTIMATE,
            f"Body ≈{tokens} tokens (budget {MAX_BODY_TOKEN_ESTIMATE}); consider moving detail to bundled files"
            if tokens > MAX_BODY_TOKEN_ESTIMATE
            else f"Body ≈{tokens} tokens, within budget",
        )
    )
    dead = _find_dead_sections(body)
    checks.append(
        _check(
            "no_dead_sections",
            "body",
            "warning",
            not dead,
            f"Empty sections: {', '.join(dead)}" if dead else "No empty sections",
        )
    )
    return checks


def _structure_checks(body: str) -> List[Dict[str, Any]]:
    has_refs = bool(_BUNDLED_REF_RE.search(body))
    tokens = estimate_tokens(body)
    # Small skills don't need progressive disclosure; large ones should split.
    passed = has_refs or tokens <= MAX_BODY_TOKEN_ESTIMATE
    return [
        _check(
            "progressive_disclosure",
            "structure",
            "info",
            passed,
            "References bundled files (progressive disclosure)"
            if has_refs
            else "No bundled-file references — fine for small skills, split if the body grows",
        )
    ]


def _security_checks(text: str) -> List[Dict[str, Any]]:
    checks = []
    for check_id, pattern, message in _SECURITY_PATTERNS:
        hit = re.search(pattern, text, re.IGNORECASE)
        checks.append(
            _check(
                check_id,
                "security",
                _SECURITY_SEVERITY[check_id],
                hit is None,
                f"{message}: `{hit.group(0).strip()}`" if hit else f"OK — no pattern: {message.lower()}",
            )
        )
    return checks


def _score(checks: List[Dict[str, Any]]) -> int:
    score = 100
    for check in checks:
        if check["passed"]:
            continue
        if check["severity"] == "error":
            score -= ERROR_PENALTY
        elif check["severity"] == "warning":
            score -= WARNING_PENALTY
    return max(0, score)


def lint_skill(text: str) -> Dict[str, Any]:
    """Run all static checks against SKILL.md content."""
    raw_fm, body = split_frontmatter(text or "")
    frontmatter, fm_error = _parse_frontmatter(raw_fm)

    checks: List[Dict[str, Any]] = []
    checks.extend(_format_checks(frontmatter, fm_error))
    checks.extend(_body_checks(body))
    checks.extend(_structure_checks(body))
    checks.extend(_security_checks(text or ""))

    failed = [c for c in checks if not c["passed"]]
    return {
        "score": _score(checks),
        "checks": checks,
        "summary": {
            "total_checks": len(checks),
            "failed": len(failed),
            "errors": sum(1 for c in failed if c["severity"] == "error"),
            "warnings": sum(1 for c in failed if c["severity"] == "warning"),
            "security_flags": sum(1 for c in failed if c["area"] == "security"),
            "name": (frontmatter or {}).get("name"),
            "body_tokens_estimate": estimate_tokens(body.strip()),
        },
    }


def lint_skill_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return lint_skill(f.read())
