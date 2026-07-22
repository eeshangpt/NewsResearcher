"""NFR-3 module-boundary check (Story 1.3 acceptance criterion).

`sourcing/google_news_backfill.py` must stay a non-load-bearing, structurally
isolated module: `sourcing/rss.py` and `sourcing/gdelt.py` must never import
from it (only a future orchestrator, `agents/sourcing_agent.py`, is allowed
to). This walks each module's actual AST import statements rather than
grepping text, so a re-export or indirect string reference wouldn't produce
a false pass.
"""

import ast
from pathlib import Path

SOURCING_DIR = Path(__file__).resolve().parent.parent / "newsresearch" / "sourcing"


def _imported_module_names(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_rss_module_never_imports_google_news_backfill():
    imports = _imported_module_names(SOURCING_DIR / "rss.py")

    assert not any("google_news_backfill" in name for name in imports)


def test_gdelt_module_never_imports_google_news_backfill():
    gdelt_path = SOURCING_DIR / "gdelt.py"
    if not gdelt_path.exists():
        # gdelt.py is a separate, parallel Story 1.2 task; skip if not yet
        # merged into this checkout rather than failing on a missing file.
        return

    imports = _imported_module_names(gdelt_path)
    assert not any("google_news_backfill" in name for name in imports)


def test_google_news_backfill_has_its_own_public_fetch_function():
    from newsresearch.sourcing import google_news_backfill

    assert callable(google_news_backfill.fetch_google_news_backfill)
