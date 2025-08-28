"""
Progress utilities for colorful CLI progress bars using tqdm.

Goals:
- Easy to use when processing data collections (iterating items)
- Easy to use alongside language model batch generation (generate())
- Colorful output that looks good in typical terminals

Usage examples
--------------

1) Wrap a simple iterable

    from some.progress import progress_iterable
    for item in progress_iterable(items, desc="Processing", unit="item", colour="cyan"):
        ...

2) Manual bar for custom loops

    from some.progress import progress_bar
    with progress_bar(total=len(items), desc="Processing", unit="item", colour="cyan") as pbar:
        for item in items:
            ... # work
            pbar.update(1)

3) Use with concurrent futures (e.g., inside a provider's generate())

    from concurrent.futures import ThreadPoolExecutor
    from some.progress import as_completed_with_tqdm

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(task, i, item) for i, item in enumerate(inputs)]
        for fut in as_completed_with_tqdm(futures, total=len(futures), desc="LLM", unit="item", colour="magenta"):
            _ = fut.result()

Notes
-----
- We detect non-TTY environments and automatically disable progress bars to avoid noisy logs
- tqdm uses the British spelling: "colour" (supported in recent versions)
"""
from __future__ import annotations

from typing import Iterable, Iterator, Sequence, Optional, TypeVar
import os
import sys

import logging
import tqdm as _tqdm

from tqdm import tqdm  # type: ignore
from concurrent.futures import Future, as_completed

T = TypeVar("T")

# Global switch to enable/disable progress output at runtime
_ENABLED: bool = True

def set_progress_enabled(enabled: bool) -> None:
    """Enable or disable all progress bars globally."""
    global _ENABLED
    _ENABLED = bool(enabled)


class TqdmLoggingHandler(logging.Handler):
    """A logging handler that writes via tqdm to keep the bar at bottom.

    Use this handler when verbose logs are enabled so that logs print above
    the progress bar using tqdm.write(), preventing bar disruption.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            # Fallback to plain stdout on any failure
            print(record.getMessage())


# --- Coloring helpers -------------------------------------------------------
_COLOR_CODES = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "reset": "\033[0m",
}


def _supports_color() -> bool:
    # Disable color if not a TTY or if explicitly disabled
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def color_text(text: str, color: str | None) -> str:
    if not color or not _supports_color():
        return text
    code = _COLOR_CODES.get(color.lower())
    if not code:
        return text
    return f"{code}{text}{_COLOR_CODES['reset']}"


# --- Public API -------------------------------------------------------------

def progress_iterable(
    iterable: Iterable[T] | Sequence[T],
    *,
    desc: str = "",
    total: Optional[int] = None,
    colour: Optional[str] = None,
    unit: str = "it",
    leave: bool = True,
    position: Optional[int] = None,
) -> Iterable[T]:
    """Wrap any iterable with a colorful tqdm progress bar.

    Parameters
    - iterable: items to iterate
    - desc: left-hand description (will be colorized)
    - total: explicit total; if None, will attempt to infer from len()
    - colour: tqdm bar color (e.g., "cyan", "magenta", "green")
    - unit: unit label (e.g., "item", "record")
    - leave: whether to leave the bar on screen after completion
    - position: bar row to use (auto: 0 when verbose, else None)
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = None

    disable = (not _ENABLED) or (not sys.stdout.isatty())
    pbar_desc = color_text(desc, colour)

    if position is None:
        # Under verbose logging, keep this bar pinned at the bottom row 0
        if logging.getLogger().getEffectiveLevel() <= logging.INFO:
            position = 0

    return tqdm(
        iterable,
        total=total,
        desc=pbar_desc,
        unit=unit,
        dynamic_ncols=True,
        colour=colour,
        leave=leave,
        position=position,
        disable=disable,
    )


def progress_bar(
    *,
    total: Optional[int] = None,
    desc: str = "",
    colour: Optional[str] = None,
    unit: str = "it",
    leave: bool = True,
) -> tqdm:
    """Create a manual tqdm progress bar with colorful description.

    Use as a context manager:
        with progress_bar(total=10, desc="Work", colour="green") as pbar:
            ...
            pbar.update(1)
    """
    disable = (not _ENABLED) or (not sys.stdout.isatty())
    pbar_desc = color_text(desc, colour)
    return tqdm(
        total=total,
        desc=pbar_desc,
        unit=unit,
        dynamic_ncols=True,
        colour=colour,
        leave=leave,
        disable=disable,
    )


def as_completed_with_tqdm(
    futures: Iterable[Future[T]],
    *,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
    colour: Optional[str] = None,
    leave: bool = True,
    position: Optional[int] = None,
) -> Iterator[Future[T]]:
    """Yield futures as they complete and update a colorful tqdm bar.

    Designed to be a drop-in for concurrent.futures.as_completed in batch
    processing code (e.g., inside provider generate() implementations).

    Example:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(task, i, item) for i, item in enumerate(inputs)]
            for fut in as_completed_with_tqdm(futures, total=len(futures), desc="LLM", unit="item", colour="magenta"):
                ...
    """
    # Try inferring total if not provided
    if total is None:
        try:
            total = len(futures)  # type: ignore[arg-type]
        except Exception:
            total = None

    disable = (not _ENABLED) or (not sys.stdout.isatty())
    pbar_desc = color_text(desc, colour)

    with tqdm(
        total=total,
        desc=pbar_desc,
        unit=unit,
        dynamic_ncols=True,
        colour=colour,
        leave=leave,
        position=position,
        disable=disable,
    ) as pbar:
        for fut in as_completed(futures):
            # Always increment by 1; tqdm supports unknown totals and will display count
            pbar.update(1)
            yield fut


# Convenience presets for common tasks --------------------------------------

def llm_progress(num_items: int) -> tqdm:
    """Preset bar for LLM batch generation step (by count of inputs)."""
    return progress_bar(total=num_items, desc="LLM generate", unit="item", colour="magenta")

