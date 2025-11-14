"""Comprehensive tests for utility functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from primal_logic.utils import (
    flatten,
    laplacian_2d,
    mean,
    safe_clip,
    write_csv,
    zeros,
    zeros_like,
)


def test_safe_clip_within_bounds() -> None:
    """Test that values within bounds are unchanged."""
    assert safe_clip(5.0, 0.0, 10.0) == 5.0
    assert safe_clip(0.5, 0.0, 1.0) == 0.5
    assert safe_clip(-2.0, -5.0, 5.0) == -2.0


def test_safe_clip_at_boundaries() -> None:
    """Test clipping at exact boundaries."""
    assert safe_clip(0.0, 0.0, 10.0) == 0.0
    assert safe_clip(10.0, 0.0, 10.0) == 10.0
    assert safe_clip(-5.0, -5.0, 5.0) == -5.0
    assert safe_clip(5.0, -5.0, 5.0) == 5.0


def test_safe_clip_exceeds_bounds() -> None:
    """Test clipping values that exceed bounds."""
    assert safe_clip(15.0, 0.0, 10.0) == 10.0
    assert safe_clip(-5.0, 0.0, 10.0) == 0.0
    assert safe_clip(1.5, 0.0, 1.0) == 1.0
    assert safe_clip(-0.1, 0.0, 1.0) == 0.0


def test_mean_empty_sequence() -> None:
    """Test mean of empty sequence returns 0.0."""
    assert mean([]) == 0.0


def test_mean_values() -> None:
    """Test mean calculation for various sequences."""
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([5.0]) == 5.0
    assert mean([0.0, 0.0, 0.0]) == 0.0
    assert mean([-1.0, 1.0]) == 0.0
    assert mean([10.0, 20.0, 30.0]) == 20.0


def test_flatten_2d_matrix() -> None:
    """Test flattening of 2D matrices."""
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    assert flatten(matrix) == [1.0, 2.0, 3.0, 4.0]

    single_row = [[1.0, 2.0, 3.0]]
    assert flatten(single_row) == [1.0, 2.0, 3.0]

    empty = [[]]
    assert flatten(empty) == []


def test_zeros_2d() -> None:
    """Test 2D zero matrix creation."""
    matrix = zeros((3, 4))
    assert len(matrix) == 3
    assert len(matrix[0]) == 4
    assert all(all(val == 0.0 for val in row) for row in matrix)


def test_zeros_3d() -> None:
    """Test 3D zero matrix creation."""
    matrix = zeros((2, 3, 4))
    assert len(matrix) == 2
    assert len(matrix[0]) == 3
    assert len(matrix[0][0]) == 4
    assert all(
        all(all(val == 0.0 for val in layer) for layer in row)
        for row in matrix
    )


def test_zeros_invalid_shape() -> None:
    """Test that invalid shapes raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported shape"):
        zeros((2,))  # 1D not supported

    with pytest.raises(ValueError, match="Unsupported shape"):
        zeros((2, 3, 4, 5))  # 4D not supported


def test_zeros_like() -> None:
    """Test zero matrix creation with matching dimensions."""
    original = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = zeros_like(original)
    assert len(result) == 2
    assert len(result[0]) == 3
    assert all(all(val == 0.0 for val in row) for row in result)

    empty = []
    result_empty = zeros_like(empty)
    assert result_empty == []


def test_laplacian_2d_interior() -> None:
    """Test Laplacian computation for interior points."""
    # Create a simple field with constant values except center
    field = [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]
    lap = laplacian_2d(field)

    # Interior point: (-4*2.0 + 1.0 + 1.0 + 1.0 + 1.0) / 1.0 = -4.0
    assert lap[1][1] == pytest.approx(-4.0)


def test_laplacian_2d_neumann_boundaries() -> None:
    """Test that Neumann boundary conditions are applied."""
    field = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    lap = laplacian_2d(field)

    # Boundary values should be copied from interior
    # Top boundary copies from row 1
    assert lap[0] == lap[1]
    # Bottom boundary copies from row -2
    assert lap[-1] == lap[-2]


def test_laplacian_2d_empty_field() -> None:
    """Test Laplacian of empty field."""
    empty_field = []
    lap = laplacian_2d(empty_field)
    assert lap == []


def test_write_csv_formatting(tmp_path: Path) -> None:
    """Test CSV writing with proper formatting."""
    output = tmp_path / "test.csv"
    header = ["col1", "col2", "col3"]
    rows = [
        [1.0, 2.5, 3.123456789],
        [4.0, 5.0, 6.0],
    ]

    write_csv(str(output), header, rows)

    assert output.exists()
    with open(output, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert lines[0].strip() == "col1,col2,col3"
    assert "1.000000000" in lines[1]
    assert "2.500000000" in lines[1]
    assert "3.123456789" in lines[1]
