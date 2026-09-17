import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_publication_files_exist():
    expected = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "docs/architecture.md",
        "apps/transmitter.py",
        "apps/receiver.py",
        "src/edge_semcom/neural_phy.py",
    ]
    missing = [path for path in expected if not (ROOT / path).is_file()]
    assert not missing, f"Missing publication files: {missing}"


def test_reference_results_are_valid_json():
    result_files = sorted((ROOT / "results").glob("*.json"))
    assert result_files
    for path in result_files:
        with path.open("r", encoding="utf-8") as handle:
            assert json.load(handle) is not None


def test_no_oversized_files():
    oversized = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.stat().st_size > 50 * 1024 * 1024
    ]
    assert not oversized, f"Files larger than 50 MiB: {oversized}"

