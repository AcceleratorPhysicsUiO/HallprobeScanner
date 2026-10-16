#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:23:20 2026

@author: pierredr

Objective: correct the recorded csv, so that we flip the values of the column called Bz
(there was a mistake in the way we changed the axis on 20261014-15)

"""

from pathlib import Path
import csv


# ---------------------------------------------------------------------
# Paths to the folders: start and arrival
# ---------------------------------------------------------------------

source_dir = Path(
    "/mn/fys-server1/a1/pierredr/Documents/3-code/HallprobeScanner/data/20261015/csv/1-wrongBzOrientation"
)

destination_dir = Path(
    "/mn/fys-server1/a1/pierredr/Documents/3-code/HallprobeScanner/data/20261015/csv/2-correctedBzOrientation"
)


# ---------------------------------------------------------------------
# Treating al CSVs
# ---------------------------------------------------------------------

csv_files = sorted(source_dir.glob("*.csv"))

if not csv_files:
    print(f"No CSV found in :\n{source_dir}")

for input_file in csv_files:
    output_file = destination_dir / input_file.name

    print(f"Correcting : {input_file.name}")

    with input_file.open("r", newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        rows = list(reader)

    if not rows:
        print(f"  Careful : empty file, ignored.")
        continue

    # Definig header
    header = rows[0]


    # Optional verification of the third column name.
    if header[2].strip() != "Bz":
        print(
            f"  Problem : column C is named"
            f"'{header[2]}' and not 'Bz'."
        )

    # Conserving the header : rows[0].
    # Inverting Bz for all lines.
    for line_number, row in enumerate(rows[1:], start=2):

        try:
            bz_value = float(row[2])
            row[2] = f"{-bz_value:.12g}"

        except ValueError:
            print(
                f"  Attention : Bz value invalid at line "
                f"{line_number}: '{row[2]}'. Line unchanged."
            )

    # Writing the corrected CSV, with exactly the same name.
    with output_file.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerows(rows)

    print(f"  Saved in: {output_file}")

print("\nCorrection completed.")


