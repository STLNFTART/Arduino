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
from typing import Iterable, List, Sequence

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


def _iter_files(target: Path) -> Iterable[Path]:
    """Yield all files under *target* recursively."""

    for item in target.rglob("*"):
        if item.is_file():
            yield item


def _format_path(path: Path, root: Path) -> str:
    """Return a POSIX-style, repository-relative path string."""

    return path.relative_to(root).as_posix()


def gather_inventory(root: Path) -> List[Entry]:
    """Collect repository metadata starting at *root*.

    Parameters
    ----------
    root:
        Repository root directory. The caller is expected to pass
        ``Path(__file__).resolve().parents[1]`` or similar.

    Returns
    -------
    List[Entry]
        Sorted list of metadata entries, suitable for tabular presentation.
    """

    if not root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {root}")

    entries: List[Entry] = []
    for item in sorted(root.iterdir()):
        if item.name == ".git":
            # Internal VCS data; not part of the presentation dataset.
            continue

        if item.is_file():
            size = item.stat().st_size
            entries.append(
                Entry(
                    path=_format_path(item, root),
                    kind="file",
                    file_count=1,
                    byte_size=size,
                    notes="empty" if size == 0 else "",
                )
            )
        elif item.is_dir():
            files = list(_iter_files(item))
            file_count = len(files)
            byte_size = sum(f.stat().st_size for f in files)
            note = "no files" if file_count == 0 else ""
            entries.append(
                Entry(
                    path=_format_path(item, root) + "/",
                    kind="directory",
                    file_count=file_count,
                    byte_size=byte_size,
                    notes=note,
                )
            )
        else:
            logger.debug("Skipping non-standard path: %s", item)

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

