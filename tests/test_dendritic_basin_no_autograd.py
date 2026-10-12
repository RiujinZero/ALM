"""Design doc section 3.1 / 44: this package must not depend on autograd
at all -- checked here by asserting no module in it imports torch (or any
other autograd-capable framework), not just that `.backward()` happens to
go unused. NumPy provides the hard guarantee: it has no autograd, so
nothing in this package can accidentally route a gradient anywhere.
"""

from __future__ import annotations

import ast
from pathlib import Path

import manifoldflow.dendritic_basin as dendritic_basin

FORBIDDEN_MODULES = {"torch", "jax", "tensorflow", "autograd"}


def _package_source_files() -> list[Path]:
    package_dir = Path(dendritic_basin.__file__).parent
    return list(package_dir.rglob("*.py"))


def test_package_has_source_files_to_check() -> None:
    assert len(_package_source_files()) >= 5


def test_no_module_in_package_imports_an_autograd_framework() -> None:
    for path in _package_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {node.module.split(".")[0]} if node.module else set()
            else:
                continue
            offenders = names & FORBIDDEN_MODULES
            assert not offenders, f"{path} imports forbidden module(s): {offenders}"
