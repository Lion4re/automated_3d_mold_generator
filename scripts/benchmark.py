"""Run moldgen on every model in a folder and tabulate the results.

    python scripts/benchmark.py MODELS_DIR [--size 60] [--out results.json] [--flat]

Each model is scaled so its longest side is ``--size`` mm (real files come in
metres, inches or millimetres), then molded with the default settings. The
table lists what matters for a usable mold: how many pieces, how much of the
surface no piece releases, how much was filled, whether any piece catches,
how long it took and what was warned about. Use it before and after a change
to see what the change did across many shapes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import trimesh

from moldgen import MoldConfig, generate_mold
from moldgen.config import SUPPORTED_SUFFIXES
from moldgen.meshio import load_mesh


def run(path: Path, size: float, config: MoldConfig) -> dict:
    row: dict = {"model": path.name}
    start = time.perf_counter()
    try:
        mesh = load_mesh(path)
        scale = size / float(max(mesh.extents))
        mesh = mesh.copy().apply_scale(scale)
        result = generate_mold(mesh, config)
    except Exception as exc:  # the benchmark records failures instead of stopping
        row.update(
            error=f"{type(exc).__name__}: {exc}",
            seconds=round(time.perf_counter() - start, 1),
            trace=traceback.format_exc(limit=3),
        )
        return row
    layout = result.layout
    caps = layout.caps if layout is not None else []
    row.update(
        seconds=round(time.perf_counter() - start, 1),
        faces=len(result.part.mesh.faces),
        repaired=bool(result.part.repair.actions),
        pieces=len(result.pieces),
        side_pieces=len(caps),
        curved_side=sum(cap.cut is not None for cap in caps),
        curved_parting=result.surface is not None,
        undercut_two_piece=round(result.parting.undercut_fraction, 4),
        locked=None if layout is None else round(layout.locked_fraction, 4),
        filled_cm3=None if layout is None else round(layout.filled_volume / 1000.0, 3),
        catch=result.pieces_catch,
        keys=int(sum(len(plan.positions) for plan in result.keys)),
        watertight=all(piece.mesh.is_watertight for piece in result.pieces),
        warnings=result.warnings,
    )
    return row


def table(rows: list[dict]) -> str:
    head = "| model | s | pieces | side (curved) | parting | 2-piece undercut | locked | filled cm³ | catch | keys | warnings |"
    lines = [head, "|" + "---|" * 11]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['model']} | {r['seconds']} | ERROR: {r['error'][:80]} |" + " |" * 8)
            continue
        locked = "-" if r["locked"] is None else f"{100 * r['locked']:.2f} %"
        filled = "-" if r["filled_cm3"] is None else f"{r['filled_cm3']:.2f}"
        lines.append(
            f"| {r['model']} | {r['seconds']} | {r['pieces']} | {r['side_pieces']} ({r['curved_side']}) "
            f"| {'curved' if r['curved_parting'] else 'flat'} | {100 * r['undercut_two_piece']:.1f} % "
            f"| {locked} | {filled} | {'YES' if r['catch'] else 'no'} | {r['keys']} | {len(r['warnings'])} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", type=Path)
    parser.add_argument("--size", type=float, default=60.0, help="Longest side after scaling (mm).")
    parser.add_argument("--out", type=Path, help="Also write all results as JSON.")
    parser.add_argument("--flat", action="store_true", help="Flat side pieces and parting only.")
    args = parser.parse_args(argv)

    config = MoldConfig()
    if args.flat:
        config = MoldConfig(side_piece_cuts="flat", parting_surface="flat")
    paths = sorted(p for p in args.models.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    rows = []
    for path in paths:
        row = run(path, args.size, config)
        rows.append(row)
        print(f"{row['model']}: {row.get('error', 'ok')} ({row['seconds']} s)", file=sys.stderr)
    print(table(rows))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    trimesh.util.log.setLevel("ERROR")
    sys.exit(main())
