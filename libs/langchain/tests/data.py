"""Module defines common test data."""

from pathlib import Path

_THIS_DIR = Path(__file__).parent

_EXAMPLES_DIR = _THIS_DIR / "integration_tests" / "examples"

# Paths to test PDF files
HELLO_PDF = _EXAMPLES_DIR / "hello.pdf"
LAYOUT_PARSER_PAPER_PDF = _EXAMPLES_DIR / "layout-parser-paper.pdf"
DUPLICATE_CHARS = _EXAMPLES_DIR / "duplicate-chars.pdf"

# Paths to data files
MLB_TEAMS_2012_CSV = _EXAMPLES_DIR / "mlb_teams_2012.csv"
MLB_TEAMS_2012_SQL = _EXAMPLES_DIR / "mlb_teams_2012.sql"
