from __future__ import annotations


def test_star_import_exposes_declared_all_symbols() -> None:
    ns: dict[str, object] = {}
    exec("from eval3r.presets import *", {}, ns)

    assert "PRESETS" in ns
    assert "tnt" in ns
