from __future__ import annotations

from pathlib import Path

from git import Repo
from holoviews import extension


__version__ = "0.0.6"


CODE_ROOT = Path(
    Repo(".", search_parent_directories=True).working_tree_dir,
).joinpath("statistics_revision")
extension("bokeh")
