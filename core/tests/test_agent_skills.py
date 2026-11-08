"""The agent skills and AGENTS.md must name things that exist.

Every skill is .claude/skills/<name>/SKILL.md with `name` and `description`
frontmatter, listed in AGENTS.md. Every repo path, preset, sweep, config-group
override and `python -m` module a skill mentions must resolve, so the playbooks
fail here instead of misleading an agent after a rename.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILLS = sorted((ROOT / ".claude" / "skills").glob("*/SKILL.md"))
DOCS = [*SKILLS, ROOT / "AGENTS.md", ROOT / "configs" / "README.md"]
TOP = ("core/", "configs/", "docs/", "scripts/", "experiments/", "artifacts/", ".claude/")
PLACEHOLDER = re.compile(r"[<>*{}$]|\.\.\.")


def frontmatter(path):
    text = path.read_text()
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert match, f"{path} has no frontmatter"
    fields = dict(line.split(": ", 1) for line in match.group(1).splitlines())
    return fields, text[match.end():]


def test_skills_exist_and_are_indexed():
    assert len(SKILLS) >= 13
    agents = (ROOT / "AGENTS.md").read_text()
    listed = set(re.findall(r"\(\.claude/skills/([\w-]+)/SKILL\.md\)", agents))
    assert listed == {path.parent.name for path in SKILLS}
    for path in SKILLS:
        fields, body = frontmatter(path)
        assert fields.get("name") == path.parent.name
        assert 0 < len(fields.get("description", "")) <= 1024
        assert "## Done when" in body, f"{path.parent.name} needs a 'Done when' check"


def mentions(path):
    """(kind, value) pairs a doc refers to."""
    text = path.read_text()
    for token in re.findall(r"`([^`\n]+)`", text):
        for word in token.split():
            yield "token", word.strip(".,;:()'\"")
    for target in re.findall(r"\]\(([^)#\s]+)", text):
        if not target.startswith(("http://", "https://", "mailto:")):
            yield "link", target
    for path in re.findall(r"(?<![\w/.$-])(?:core|configs|docs|scripts|experiments|artifacts|\.claude)/[\w./@<>*{}-]*", text):
        yield "path", path


def resolve_path(word):
    word = re.sub(r":\d+$", "", word).rstrip("/.")
    if not word.startswith(TOP) or PLACEHOLDER.search(word) or "=" in word:
        return None
    return ROOT / word


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_named_paths_presets_overrides_and_modules_exist(doc):
    missing = []
    text = doc.read_text()
    for kind, value in set(mentions(doc)):
        if kind == "link":
            if not (doc.parent / value).exists():
                missing.append(f"link {value}")
            continue
        path = resolve_path(value)
        if path is not None and not path.exists():
            missing.append(f"path {value}")
        for group, option in re.findall(r"^\+?(experiment|sweep)=([\w/]+)$", value):
            if not (ROOT / "configs" / group / f"{option}.yaml").exists():
                missing.append(f"+{group}={option}")
        match = re.match(r"^\+{0,2}([a-z_]+(?:/[a-z_]+)*)(?:@[\w.]+)?=([\w/]+)$", value)
        if match and (ROOT / "configs" / match.group(1)).is_dir():
            group, option = match.groups()
            if not (ROOT / "configs" / group / f"{option}.yaml").exists():
                missing.append(f"override {value}")
    for group, option in re.findall(r"-\s+(?:override\s+)?/([a-z_/]+)(?:@[\w.]+)?:\s*([\w/]+)", text):
        if not (ROOT / "configs" / group / f"{option}.yaml").exists():
            missing.append(f"default /{group}: {option}")
    for module in re.findall(r"python -m ([\w.]+)", text):
        if module.split(".")[0] in {"core", "scripts"} and importlib.util.find_spec(module) is None:
            missing.append(f"module {module}")
    assert not missing, f"{doc.relative_to(ROOT)} names missing things: {sorted(set(missing))}"
