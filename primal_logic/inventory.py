"""Repository inventory utilities.

This module generates presentation-ready metadata tables describing the
current repository layout. It is useful for documenting baselines when a
linked project (e.g., an upstream Arduino repository) is empty or only
partially populated. The helpers operate purely on the standard library so
the workflow remains reproducible in offline environments.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence

logger = logging.getLogger(__name__)


@dataclass
class Entry:
    """Metadata describing a single repository node.

    Attributes
    ----------
    path:
        Repository-relative path rendered using POSIX separators for
        portability.
    kind:
        One of ``"file"`` or ``"directory"``.
    file_count:
        Number of files contained within the node. For files this is ``1``;
        for directories it reflects the recursive file total.
    byte_size:
        Aggregate size in bytes across all files in the node. Directories sum
        the sizes of their descendant files.
    notes:
        Optional textual annotation highlighting empty directories or other
        conditions that are useful when reviewing a minimal or placeholder
        repository.
    """

    path: str
    kind: str
    file_count: int
    byte_size: int
    notes: str = ""


def gather_inventory(root: Path) -> List[Entry]:
    """Collect repository metadata starting at *root*.

    The earlier implementation only enumerated the first level of the
    repository tree, which meant nested changes failed to appear in generated
    inventories. To guarantee that "all changes show up" we now traverse the
    tree recursively, recording a row for every file and directory (excluding
    the Git metadata folder). Directory statistics aggregate the total number of
    descendant files and their combined size to provide a concise overview.
    """

    if not root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {root}")

    files: List[Path] = []
    file_sizes: dict[Path, int] = {}
    directories: set[Path] = set()

    for path in root.rglob("*"):
        relative = path.relative_to(root)

        if ".git" in relative.parts:
            # Internal VCS data; not part of the presentation dataset.
            continue

        if relative == Path("."):
            continue

        if path.is_file():
            size = path.stat().st_size
            files.append(relative)
            file_sizes[relative] = size
            for parent in relative.parents:
                if parent == Path("."):
                    break
                directories.add(parent)
        elif path.is_dir():
            directories.add(relative)
        else:
            logger.debug("Skipping non-standard path: %s", path)

    entries: List[Entry] = []

    for relative in sorted(files, key=lambda p: p.as_posix()):
        size = file_sizes[relative]
        entries.append(
            Entry(
                path=relative.as_posix(),
                kind="file",
                file_count=1,
                byte_size=size,
                notes="empty" if size == 0 else "",
            )
        )

    for directory in sorted(directories, key=lambda p: p.as_posix()):
        dir_parts = directory.parts
        count = 0
        total_size = 0
        for file_path, size in file_sizes.items():
            if file_path.parts[: len(dir_parts)] == dir_parts:
                count += 1
                total_size += size

        note = "no files" if count == 0 else ""
        entries.append(
            Entry(
                path=directory.as_posix().rstrip("/") + "/",
                kind="directory",
                file_count=count,
                byte_size=total_size,
                notes=note,
            )
        )

    # Include a leading aggregate entry for the repository root so reports
    # always display an overall summary even if callers only inspect the first
    # few rows.  This helps reviewers notice that new files were discovered
    # during recursive traversal and mirrors the behaviour of conventional
    # disk-usage tools that front-load totals.
    total_files = sum(entry.file_count for entry in entries if entry.kind == "file")
    total_size = sum(entry.byte_size for entry in entries if entry.kind == "file")
    entries.insert(
        0,
        Entry(
            path="./",
            kind="directory",
            file_count=total_files,
            byte_size=total_size,
            notes="root summary",
        ),
    )

    return entries


def write_inventory_csv(path: Path, entries: Sequence[Entry]) -> Path:
    """Write *entries* to *path* in CSV format and return the destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "kind", "file_count", "byte_size", "notes"])
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))
    logger.info("Saved repository inventory CSV to %s", path)
    return path


def render_markdown_table(entries: Sequence[Entry]) -> str:
    """Return a Markdown table summarizing *entries*."""

    header = "| Path | Kind | Files | Size (bytes) | Notes |"
    separator = "| --- | --- | ---: | ---: | --- |"
    rows = [header, separator]
    for entry in entries:
        notes = entry.notes or "—"
        rows.append(
            f"| {entry.path} | {entry.kind} | {entry.file_count} | {entry.byte_size} | {notes} |"
        )
    return "\n".join(rows)


def write_markdown_table(path: Path, entries: Sequence[Entry]) -> Path:
    """Write a Markdown table representation of *entries* to *path*."""

    path.parent.mkdir(parents=True, exist_ok=True)
    table_text = render_markdown_table(entries)
    path.write_text(table_text, encoding="utf-8")
    logger.info("Saved repository inventory table to %s", path)
    return path


def generate_inventory_artifacts(root: Path, csv_path: Path, markdown_path: Path) -> None:
    """Produce CSV and Markdown artifacts summarising *root*.

    The function wraps :func:`gather_inventory` and writes both artifact formats
    so downstream presentation decks can leverage whichever representation best
    fits the workflow.
    """

    entries = gather_inventory(root)
    write_inventory_csv(csv_path, entries)
    write_markdown_table(markdown_path, entries)

