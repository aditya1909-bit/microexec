from __future__ import annotations

import sys


def progress_bar(current: int, total: int, prefix: str = "", width: int = 30) -> None:
    total = max(1, total)
    current = max(0, min(current, total))
    frac = current / total
    filled = int(round(frac * width))
    bar = "=" * filled + "-" * (width - filled)
    if prefix and not prefix.endswith(" "):
        prefix = f"{prefix} "
    msg = f"\r{prefix}[{bar}] {current}/{total}"
    sys.stdout.write(msg)
    if current >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()
