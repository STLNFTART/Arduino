"""Tests for repository inventory tooling."""

from __future__ import annotations

from pathlib import Path

from primal_logic.inventory import gather_inventory, render_markdown_table, write_inventory_csv


def test_gather_inventory_returns_entries(tmp_path: Path) -> None:
    (tmp_path / "empty_dir").mkdir()
    file_path = tmp_path / "file.txt"
    file_path.write_text("demo", encoding="utf-8")

    entries = gather_inventory(tmp_path)
    paths = {entry.path for entry in entries}

    assert "file.txt" in paths
    assert "empty_dir/" in paths

    # CSV should write without error.
    csv_path = tmp_path / "inventory.csv"
    write_inventory_csv(csv_path, entries)
    assert csv_path.exists()

    # Markdown render should include both paths.
    table = render_markdown_table(entries)
    assert "file.txt" in table
    assert "empty_dir/" in table
