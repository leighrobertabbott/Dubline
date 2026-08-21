"""Guards against the README drifting away from the code it documents.

The configuration table had accumulated three stale defaults and one variable
that no longer existed anywhere, which is the kind of thing nobody notices until
a user copies a setting that does nothing.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ENV_EXAMPLE = ROOT / ".env.example"

# Defaults the table states in prose or abbreviates, which cannot be compared
# literally against .env.example.
NON_LITERAL = {"*None*", "bundled"}


def documented_variables() -> dict[str, str]:
    """Every `VAR` | default | row of the README configuration table."""
    rows = re.findall(r"^\|\s*`([A-Z][A-Z0-9_]*)`\s*\|\s*(.+?)\s*\|",
                      README.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return {name: default.strip().strip("`") for name, default in rows}


def example_variables() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip()
    return values


def source_text() -> str:
    files = [*(ROOT / "app").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in files)


def test_every_documented_variable_still_exists_in_the_code():
    sources = source_text()
    orphans = [name for name in documented_variables() if name not in sources]
    assert not orphans, f"README documents variables no code reads: {orphans}"


def test_documented_defaults_match_the_example_environment():
    documented = documented_variables()
    example = example_variables()
    mismatches = []
    for name, default in documented.items():
        if name not in example or default in NON_LITERAL or default.endswith("..."):
            continue
        expected = example[name]
        if expected and expected != default:
            mismatches.append(f"{name}: README says {default!r}, .env.example says {expected!r}")
    assert not mismatches, "README defaults are stale:\n  " + "\n  ".join(mismatches)


def test_the_bundled_catalogue_token_is_documented_as_public():
    readme = README.read_text(encoding="utf-8")
    config = (ROOT / "app" / "config.py").read_text(encoding="utf-8")
    # A maintainer must not be able to paste a token without meeting the warning.
    assert "BUNDLED_TMDB_TOKEN" in readme and "BUNDLED_TMDB_TOKEN" in config
    assert "PUBLIC BY DESIGN" in config
    assert "public" in readme.lower()


def test_the_help_page_only_names_controls_the_interface_actually_has():
    """The FAQ tells users to click specific labels; renaming one in the UI
    without updating the help text sends people looking for a missing button."""
    faq = (ROOT / "web" / "faq.html").read_text(encoding="utf-8")
    interface = "\n".join((ROOT / "web" / name).read_text(encoding="utf-8")
                          for name in ("index.html", "app.js", "setup.js"))
    # Bolded phrases in the FAQ are how it points at on-screen controls.
    quoted = set(re.findall(r"<strong>([A-Z][^<]{2,40})</strong>", faq))
    # Format hints and glossary examples are not controls.
    quoted -= {"MM:SS", "HH:MM:SS"}
    quoted = {phrase for phrase in quoted if "=" not in phrase}
    missing = sorted(phrase for phrase in quoted if phrase not in interface)
    assert not missing, f"Help page names controls the UI no longer has: {missing}"
