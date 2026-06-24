#!/usr/bin/env python3

"""
Create minimal BIDS JSON sidecars for fMRIPrep from NIfTI headers.

What this does:
- Finds all *_bold.nii.gz files
- Extracts TR from the NIfTI header
- Creates matching JSON sidecars with:
    {
      "RepetitionTime": <TR>,
      "TaskName": "<task>"
    }

What this does NOT do:
- SliceTiming
- PhaseEncodingDirection
- EffectiveEchoSpacing
- Fieldmap metadata

This is often enough to get fMRIPrep running if you add:
    --ignore slicetiming

Usage:
    python create_bids_jsons.py /path/to/NESTM_bids
"""

import json
import re
import sys
from pathlib import Path

try:
    import nibabel as nib
except ImportError:
    print("Installing nibabel is required:")
    print("pip install nibabel")
    sys.exit(1)

if len(sys.argv) != 2:
    print("Usage:")
    print("python create_bids_jsons.py /path/to/NESTM_bids")
    sys.exit(1)

bids_root = Path(sys.argv[1])

if not bids_root.exists():
    print(f"Directory does not exist: {bids_root}")
    sys.exit(1)

bold_files = list(bids_root.rglob("*_bold.nii.gz"))

if len(bold_files) == 0:
    print("No BOLD files found.")
    sys.exit(1)

created = 0
skipped = 0

for nifti_file in bold_files:

    json_file = nifti_file.with_suffix("").with_suffix(".json")

    if json_file.exists():
        print(f"Skipping existing JSON: {json_file}")
        skipped += 1
        continue

    try:
        img = nib.load(str(nifti_file))
        zooms = img.header.get_zooms()

        if len(zooms) < 4:
            print(f"Could not determine TR for: {nifti_file}")
            continue

        tr = float(zooms[3])

        # Extract task name from filename
        match = re.search(r'task-([a-zA-Z0-9]+)', nifti_file.name)

        if match:
            task_name = match.group(1)
        else:
            task_name = "unknown"

        metadata = {
            "RepetitionTime": tr,
            "TaskName": task_name
        }

        with open(json_file, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Created: {json_file}")
        created += 1

    except Exception as e:
        print(f"Error processing {nifti_file}:")
        print(e)

print()
print(f"Created {created} JSON files")
print(f"Skipped {skipped} existing JSON files")
