import math
from typing import Iterable, List, Tuple


def safe_calculate(expr: str) -> str:
    """Evaluate math expression using math module and block builtins."""
    allowed = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
    allowed.update({"__builtins__": {}})
    try:
        return str(eval(expr, allowed, {}))
    except Exception as exc:  # noqa: BLE001
        return f"calc_error: {exc}"


def cite_blocks(blocks: Iterable[Tuple[str, dict]], limit: int = 4) -> List[str]:
    """Format citation strings from RAG blocks."""
    cites = []
    for _, meta in blocks:
        source = meta.get("source", "Doc")
        page = meta.get("page", "?")
        cites.append(f"[{source} p.{page}]")
    deduped = []
    for cite in cites:
        if cite not in deduped:
            deduped.append(cite)
        if len(deduped) == limit:
            break
    return deduped
