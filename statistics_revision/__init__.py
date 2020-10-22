from __future__ import annotations

from pathlib import Path

from git import Repo


__version__ = "0.0.5"


CODE_ROOT = Path(
    Repo(".", search_parent_directories=True).working_tree_dir,
).joinpath("statistics_revision")
