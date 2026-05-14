#!/usr/bin/env python3
"""
generate_projection_figures_standalone.py

Standalone, auditable renderer for IBAM/CCMHCG MG contact-projection figures.

Required input files in the same directory by default:
  - C12_aligned.fa
  - MG_column_map_n26_core60_chem90.tsv
  - MG_projected_trimmed_n26_core60_chem90.fa

What it does:
  1. Reads projected interface columns from the TSV column map.
  2. Reconstructs each taxon's ungapped/raw sequence from the MAFFT alignment.
  3. Extracts each projected cassette directly from those mapped alignment columns.
  4. Cross-checks the extracted cassette against the supplied projected FASTA.
  5. Draws one PDF per taxon plus one combined all-taxon PDF.
  6. Writes a validation/summary TSV documenting cassette, support, gaps, and mismatches.

Usage:
  python generate_projection_figures_standalone.py
  python generate_projection_figures_standalone.py --out output_figures --strict

Notes:
  - Alignment-column coordinates in the TSV are expected to be 1-based.
  - Raw residue positions printed under cassette boxes are 1-based ungapped positions.
  - The blue tier means exact residue identity among taxa represented at that projected slot,
    not necessarily presence in all supporting taxa.
  - Under the 60/90 dual-gate column map used here, no moderate/variable
    projected slots are retained; the legend therefore shows only blue and green tiers.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages


# ----------------------------- Defaults ----------------------------------
DEFAULT_ALIGNED_FA = "C12_aligned.fa"
DEFAULT_COLUMN_MAP = "MG_column_map_n26_core60_chem90.tsv"
DEFAULT_PROJECTED_FA = "MG_projected_trimmed_n26_core60_chem90.fa"
DEFAULT_OUTPUT_DIR = "output_figures_standalone"

# Figure geometry
CHARS_PER_ROW = 70
CHAR_W = 0.185
ROW_H = 0.36
BOX_H = 0.50
BOX_W = 0.48
BOX_GAP = 0.045
CASSETTE_MARGIN = 0.40
FIG_W = 13.8

# Colours
COL_INVAR = "#4B52A0"   # purple-blue
COL_HIGH = "#2E8B4A"    # green
COL_GAP_ED = "#AAAAAA"
COL_TEXT = "#B8B8B8"

# Amino-acid chemistry classes used only for optional descriptive summary.
AA_CLASS = {
    **dict.fromkeys(list("KRH"), "BASIC"),
    **dict.fromkeys(list("DE"), "ACIDIC"),
    **dict.fromkeys(list("AVLIMFWY"), "HYDRO/AROM"),
    **dict.fromkeys(list("STNQ"), "POLAR"),
    "C": "SPECIAL",
    "G": "SPECIAL",
    "P": "SPECIAL",
}


@dataclass(frozen=True)
class ColumnSlot:
    slot_index: int              # 0-based projected slot
    aln_col_1based: int
    n_support: int
    dom_res: str
    dom_res_frac: float
    dom_class: str
    dom_class_frac: float
    supporters: Tuple[str, ...]

    @property
    def aln_col_0based(self) -> int:
        return self.aln_col_1based - 1

    @property
    def tier(self) -> str:
        """Colour tier. Exact identity gets blue; strong residue/chemistry conservation gets green."""
        if self.dom_res_frac >= 0.96:
            return "invariant_supporting"
        if self.dom_res_frac >= 0.82 or self.dom_class_frac >= 0.90:
            return "high"
        return "moderate"


@dataclass(frozen=True)
class ProjectedResidue:
    residue: str
    raw_pos_1based: Optional[int]
    slot: ColumnSlot


def parse_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    name: Optional[str] = None
    buf: List[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name = line[1:].strip()
                buf = []
            else:
                buf.append(line)
        if name is not None:
            seqs[name] = "".join(buf)
    if not seqs:
        raise ValueError(f"No FASTA records found in {path}")
    return seqs


def read_column_map(path: Path) -> List[ColumnSlot]:
    slots: List[ColumnSlot] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "aln_col_1based", "n_support_taxa", "dom_res", "dom_res_frac",
            "dom_class", "dom_class_frac", "supporters",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Column map is missing required fields: {sorted(missing)}")
        for idx, row in enumerate(reader):
            supporters = tuple(s for s in row["supporters"].split(",") if s)
            slots.append(ColumnSlot(
                slot_index=idx,
                aln_col_1based=int(row["aln_col_1based"]),
                n_support=int(row["n_support_taxa"]),
                dom_res=row["dom_res"],
                dom_res_frac=float(row["dom_res_frac"]),
                dom_class=row["dom_class"],
                dom_class_frac=float(row["dom_class_frac"]),
                supporters=supporters,
            ))
    if not slots:
        raise ValueError(f"No projection slots found in {path}")
    return slots


def ungap(seq: str) -> str:
    return seq.replace("-", "")


def alignment_col_to_raw_positions(aligned_seq: str) -> Dict[int, int]:
    """Map 0-based alignment columns to 1-based ungapped residue positions."""
    mapping: Dict[int, int] = {}
    raw_pos = 0
    for col, ch in enumerate(aligned_seq):
        if ch != "-":
            raw_pos += 1
            mapping[col] = raw_pos
    return mapping


def compute_projection_from_alignment(taxon: str, aligned_seq: str, slots: List[ColumnSlot]) -> List[ProjectedResidue]:
    col_to_raw = alignment_col_to_raw_positions(aligned_seq)
    projected: List[ProjectedResidue] = []
    for slot in slots:
        col = slot.aln_col_0based
        if col >= len(aligned_seq) or aligned_seq[col] == "-":
            projected.append(ProjectedResidue("-", None, slot))
        else:
            projected.append(ProjectedResidue(aligned_seq[col], col_to_raw[col], slot))
    return projected


def cassette_string(projection: Iterable[ProjectedResidue]) -> str:
    return "".join(p.residue for p in projection)


def tier_color(tier: str) -> str:
    return {
        "invariant_supporting": COL_INVAR,
        "high": COL_HIGH,
    }.get(tier, COL_HIGH)


def format_species_name(taxon: str) -> str:
    parts = taxon.split("_")
    if len(parts) >= 2:
        return f"{parts[0]} {' '.join(parts[1:])}"
    return taxon


def validate_inputs(aln: Dict[str, str], proj: Dict[str, str], slots: List[ColumnSlot], strict: bool) -> List[str]:
    warnings: List[str] = []
    aln_taxa = set(aln)
    proj_taxa = set(proj)
    if aln_taxa != proj_taxa:
        warnings.append(f"Taxon sets differ: alignment-only={sorted(aln_taxa - proj_taxa)}; projected-only={sorted(proj_taxa - aln_taxa)}")
    if len({len(s) for s in aln.values()}) != 1:
        warnings.append("Aligned FASTA records do not all have the same length.")
    aln_len = max(len(s) for s in aln.values())
    bad_cols = [slot.aln_col_1based for slot in slots if slot.aln_col_0based >= aln_len]
    if bad_cols:
        warnings.append(f"Column map contains alignment columns beyond alignment length {aln_len}: {bad_cols}")

    mismatches = []
    for taxon in sorted(aln_taxa & proj_taxa):
        got = cassette_string(compute_projection_from_alignment(taxon, aln[taxon], slots))
        expected = proj[taxon]
        if got != expected:
            mismatches.append((taxon, got, expected))
    if mismatches:
        warnings.append(f"{len(mismatches)} taxa differ between column-map extraction and projected FASTA.")
        for taxon, got, expected in mismatches[:10]:
            warnings.append(f"  {taxon}: extracted={got} projected_fasta={expected}")
    if strict and warnings:
        raise ValueError("Strict validation failed:\n" + "\n".join(warnings))
    return warnings


def draw_taxon_figure(taxon: str, raw_seq: str, projection: List[ProjectedResidue], out_path: Optional[Path] = None):
    hl_map = {p.raw_pos_1based: p.slot for p in projection if p.raw_pos_1based is not None}
    n_seq_rows = math.ceil(len(raw_seq) / CHARS_PER_ROW)

    legend_h = 0.45
    title_h = 0.40
    seq_h = n_seq_rows * ROW_H
    arrow_h = 0.50
    cassette_h = BOX_H + 0.40
    margin_t = 0.25
    margin_b = 0.35
    fig_h = title_h + legend_h + seq_h + arrow_h + cassette_h + margin_t + margin_b

    fig = plt.figure(figsize=(FIG_W, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    y = fig_h - margin_t

    # Title
    y -= title_h
    ax.text(0.30, y + title_h * 0.5, format_species_name(taxon),
            fontsize=13, style="italic", va="center", ha="left",
            fontfamily="DejaVu Sans", transform=ax.transData)

    # Legend: wording avoids claiming all slots are present in all supporting taxa.
    # No moderate/variable slots are retained by the current 60/90 dual-gate map,
    # so the legend intentionally shows only the categories actually used.
    y -= legend_h
    legend_items = [
        (COL_INVAR, "Invariant among supporting taxa"),
        (COL_HIGH, "Highly conserved residue/chemistry"),
    ]
    lx = 0.30
    for color, label in legend_items:
        rect = mpatches.FancyBboxPatch((lx, y + 0.08), 0.18, 0.22,
                                       boxstyle="square,pad=0", linewidth=0,
                                       facecolor=color, transform=ax.transData, zorder=3)
        ax.add_patch(rect)
        ax.text(lx + 0.30, y + 0.20, label, fontsize=9.5,
                va="center", ha="left", transform=ax.transData, color="#333333")
        lx += 3.85

    # Sequence rows
    font_size_seq = 10
    for row_i in range(n_seq_rows):
        y -= ROW_H
        start = row_i * CHARS_PER_ROW
        end = min(start + CHARS_PER_ROW, len(raw_seq))
        row_seq = raw_seq[start:end]
        x_start = 0.30
        for char_i, ch in enumerate(row_seq):
            raw_p = start + char_i + 1
            x = x_start + char_i * CHAR_W
            if raw_p in hl_map:
                slot = hl_map[raw_p]
                color = tier_color(slot.tier)
                rect = mpatches.FancyBboxPatch((x - 0.01, y + 0.02), CHAR_W, ROW_H * 0.80,
                                               boxstyle="square,pad=0", linewidth=0,
                                               facecolor=color, transform=ax.transData, zorder=2)
                ax.add_patch(rect)
                ax.text(x + CHAR_W * 0.45, y + ROW_H * 0.44, ch,
                        fontsize=font_size_seq, va="center", ha="center",
                        fontfamily="DejaVu Sans Mono", fontweight="bold",
                        color="white", zorder=3, transform=ax.transData)
            else:
                ax.text(x + CHAR_W * 0.45, y + ROW_H * 0.44, ch,
                        fontsize=font_size_seq, va="center", ha="center",
                        fontfamily="DejaVu Sans Mono", color=COL_TEXT,
                        transform=ax.transData)

    # Arrow
    y -= arrow_h
    arrow_cx = FIG_W * 0.5
    ax.annotate("", xy=(arrow_cx, y + 0.08), xytext=(arrow_cx, y + arrow_h - 0.08),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.5),
                transform=ax.transData)
    ax.text(arrow_cx + 0.15, y + arrow_h * 0.5,
            "3D contact projection (contact-defined column map)",
            fontsize=10.5, va="center", ha="left", color="#333333",
            transform=ax.transData)

    # Cassette strip
    # Fit the entire contact cassette inside the axes.  The previous n=25/n=26
    # renderer used fixed BOX_W/BOX_GAP values with a fixed FIG_W, which made
    # total_strip_w exceed FIG_W for n=26.  Matplotlib then clipped the first
    # and last residue boxes while their position labels remained visible.
    n_strip = len(projection)
    available_strip_w = FIG_W - 2 * CASSETTE_MARGIN
    strip_gap = BOX_GAP
    strip_box_w = min(BOX_W, (available_strip_w - (n_strip - 1) * strip_gap) / n_strip)
    if strip_box_w <= 0:
        raise ValueError(f"Cassette cannot be fitted: n_strip={n_strip}, FIG_W={FIG_W}")
    total_strip_w = n_strip * strip_box_w + (n_strip - 1) * strip_gap
    strip_x0 = (FIG_W - total_strip_w) / 2.0
    residue_font = 13 if strip_box_w >= 0.40 else 11
    pos_font = 9 if strip_box_w >= 0.40 else 7.5
    y -= cassette_h

    for i, p in enumerate(projection):
        bx = strip_x0 + i * (strip_box_w + strip_gap)
        by = y + 0.38
        cx = bx + strip_box_w * 0.5
        if p.residue == "-":
            rect = mpatches.FancyBboxPatch((bx, by), strip_box_w, BOX_H,
                                           boxstyle="square,pad=0", linewidth=1.0,
                                           linestyle="--", edgecolor=COL_GAP_ED,
                                           facecolor="none", transform=ax.transData, zorder=2)
            ax.add_patch(rect)
            ax.text(cx, by + BOX_H * 0.5, "-", fontsize=residue_font,
                    va="center", ha="center", color=COL_GAP_ED, transform=ax.transData)
            ax.text(cx, y + 0.18, "-", fontsize=pos_font,
                    va="center", ha="center", color="#888888", transform=ax.transData)
        else:
            color = tier_color(p.slot.tier)
            rect = mpatches.FancyBboxPatch((bx, by), strip_box_w, BOX_H,
                                           boxstyle="square,pad=0", linewidth=0,
                                           facecolor=color, transform=ax.transData, zorder=2)
            ax.add_patch(rect)
            ax.text(cx, by + BOX_H * 0.5, p.residue,
                    fontsize=residue_font, va="center", ha="center",
                    fontfamily="DejaVu Sans", fontweight="bold",
                    color="white", zorder=3, transform=ax.transData)
            ax.text(cx, y + 0.18, str(p.raw_pos_1based),
                    fontsize=pos_font, va="center", ha="center", color="#555555",
                    transform=ax.transData)

    if out_path is not None:
        fig.savefig(out_path, format="pdf", facecolor="white", dpi=300)
        plt.close(fig)
    return fig


def write_summary(out_path: Path, taxa: List[str], aln: Dict[str, str], projected_fasta: Dict[str, str], slots: List[ColumnSlot]):
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "taxon", "cassette_from_column_map", "projected_fasta", "matches_projected_fasta",
            "ungapped_length", "n_slots", "n_non_gap_slots", "n_gap_slots", "gap_slot_indices_1based",
            "raw_positions_1based", "alignment_columns_1based",
        ])
        for taxon in taxa:
            projection = compute_projection_from_alignment(taxon, aln[taxon], slots)
            cass = cassette_string(projection)
            expected = projected_fasta.get(taxon, "")
            gaps = [str(p.slot.slot_index + 1) for p in projection if p.residue == "-"]
            positions = ["-" if p.raw_pos_1based is None else str(p.raw_pos_1based) for p in projection]
            cols = [str(p.slot.aln_col_1based) for p in projection]
            writer.writerow([
                taxon, cass, expected, cass == expected, len(ungap(aln[taxon])), len(projection),
                sum(1 for p in projection if p.residue != "-"), sum(1 for p in projection if p.residue == "-"),
                ",".join(gaps), ",".join(positions), ",".join(cols),
            ])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render validated IBAM MG projection figures from alignment, column map, and projected FASTA.")
    parser.add_argument("--aligned-fa", default=DEFAULT_ALIGNED_FA, help="MAFFT-aligned IBAM FASTA")
    parser.add_argument("--column-map", default=DEFAULT_COLUMN_MAP, help="TSV with projected alignment columns")
    parser.add_argument("--projected-fa", default=DEFAULT_PROJECTED_FA, help="Projected cassette FASTA for validation")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--strict", action="store_true", help="Fail if validation warnings are detected")
    parser.add_argument("--combined-name", default="ALL_taxa_projections_validated.pdf", help="Combined PDF file name")
    args = parser.parse_args(argv)

    base = Path.cwd()
    aligned_path = Path(args.aligned_fa)
    column_map_path = Path(args.column_map)
    projected_path = Path(args.projected_fa)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in (aligned_path, column_map_path, projected_path):
        if not path.exists():
            print(f"ERROR: missing input file: {path}", file=sys.stderr)
            return 2

    aln = parse_fasta(aligned_path)
    projected_fasta = parse_fasta(projected_path)
    slots = read_column_map(column_map_path)
    taxa = list(aln.keys())

    warnings = validate_inputs(aln, projected_fasta, slots, args.strict)
    log_path = out_dir / "validation_log.txt"
    with log_path.open("w") as log:
        log.write(f"Aligned FASTA: {aligned_path}\n")
        log.write(f"Column map: {column_map_path}\n")
        log.write(f"Projected FASTA: {projected_path}\n")
        log.write(f"Taxa in alignment: {len(aln)}\n")
        log.write(f"Projection slots: {len(slots)}\n")
        tier_counts = {"invariant_supporting": 0, "high": 0, "moderate": 0}
        for slot in slots:
            tier_counts[slot.tier] = tier_counts.get(slot.tier, 0) + 1
        log.write("Tier counts: " + ", ".join(f"{k}={v}" for k, v in tier_counts.items()) + "\n")
        if tier_counts.get("moderate", 0) == 0:
            log.write("No moderate/variable projected slots are retained under this 60/90 column map.\n")
        if warnings:
            log.write("\nWARNINGS:\n")
            for w in warnings:
                log.write(w + "\n")
        else:
            log.write("\nValidation passed: column-map extraction matches projected FASTA for all shared taxa.\n")

    if warnings:
        print("Validation warnings detected; see", log_path)
        for w in warnings:
            print("WARNING:", w, file=sys.stderr)
    else:
        print("Validation passed: extracted cassettes match projected FASTA.")

    # Individual PDFs
    print(f"Generating {len(taxa)} individual taxon PDFs...")
    for taxon in taxa:
        raw_seq = ungap(aln[taxon])
        projection = compute_projection_from_alignment(taxon, aln[taxon], slots)
        draw_taxon_figure(taxon, raw_seq, projection, out_path=out_dir / f"{taxon}_projection.pdf")

    # Combined PDF
    combined_path = out_dir / args.combined_name
    print(f"Generating combined PDF: {combined_path}")
    with PdfPages(combined_path) as pdf:
        for taxon in taxa:
            raw_seq = ungap(aln[taxon])
            projection = compute_projection_from_alignment(taxon, aln[taxon], slots)
            fig = draw_taxon_figure(taxon, raw_seq, projection, out_path=None)
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)

    summary_path = out_dir / "projection_summary.tsv"
    write_summary(summary_path, taxa, aln, projected_fasta, slots)

    print("Wrote:")
    print(f"  {combined_path}")
    print(f"  {summary_path}")
    print(f"  {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
