"""Tests for repository inventory tooling."""

from __future__ import annotations

from pathlib import Path

from primal_logic.inventory import gather_inventory, render_markdown_table, write_inventory_csv


def test_gather_inventory_returns_entries(tmp_path: Path) -> None:
    (tmp_path / "empty_dir").mkdir()
    nested_dir = tmp_path / "nested" / "deeper"
    nested_dir.mkdir(parents=True)
    file_path = tmp_path / "file.txt"
    file_path.write_text("demo", encoding="utf-8")
    nested_file = nested_dir / "inner.txt"
    nested_file.write_text("more", encoding="utf-8")

    entries = gather_inventory(tmp_path)
    paths = {entry.path for entry in entries}

    assert entries[0].path == "./"
    assert "file.txt" in paths
    assert "empty_dir/" in paths
    assert "nested/deeper/inner.txt" in paths
    assert "nested/" in paths
    assert "nested/deeper/" in paths

    root_entry = entries[0]
    # Two files present in the fixture and both should be counted in the root summary.
    assert root_entry.file_count == 2
    assert root_entry.byte_size == len("demo") + len("more")

    # CSV should write without error.
    csv_path = tmp_path / "inventory.csv"
    write_inventory_csv(csv_path, entries)
    assert csv_path.exists()

    # Markdown render should include both paths.
    table = render_markdown_table(entries)
    assert "file.txt" in table
    assert "empty_dir/" in table
