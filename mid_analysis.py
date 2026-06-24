#!/usr/bin/env python3
"""
MID Task fMRI Analysis Pipeline – NESTM Study. Designed to be run after preprocessing with fMRIPrep
==============================================
Replicates MacNiven, Mortazavi & Knutson (2024, Biol Psychiatry) Figures 4c/4d.

GLM MODEL (matches Leili's R script / paper):
----------------------------------------------
  • Six separate anticipation regressors, one per trial type:
      gain5  (+$5)   loss5  (-$5)
      gain1  (+$1)   loss1  (-$1)
      gain0  (+$0)   loss0  (-$0)
  • Each modelled as a 2 s epoch at cue onset (cue TR = 1 s +
    immediately-following fixation TR = 1 s  → 2 s total anticipation window)
  • Both runs analysed separately; condition betas averaged across runs
    within each session before entering the RT-correlation step
  • Confounds: 6 motion params + derivatives + WM + CSF  (14 regressors)

OUTPUTS  (written to OUTPUT_DIR)
---------------------------------
first_level/
  sub-{ID}_ses-T{N}_run-{1|2}/
    {condition}_condition_effect_size.nii.gz   ← per-condition beta map
    {condition}_condition_z_score.nii.gz
    {a}_vs_{b}_effect_size.nii.gz              ← pairwise contrast maps
    {a}_vs_{b}_z_score.nii.gz
    {a}_vs_{b}_glass_brain.png
    design_matrix.png

group_level/
  {a}_vs_{b}/
    z_map.nii.gz | stat_map.nii.gz
    glass_brain.png | stat_map_slices.png

roi/
  AIns_mask.nii.gz
  AIns_betas_per_run.csv          ← per run, per condition
  AIns_betas_per_session.csv      ← run-averaged within session
  AIns_RT_merged.csv              ← betas + mean RT per subject×session×condition
  AIns_betas_summary_stats.csv

figures/
  figure_4c.png    ← scatter: RT vs AIns loss5 beta (one point per subject)
  figure_4d.png    ← bar: regression coefficients (RT→AIns beta) for all 6 conditions

Usage
-----
  python mid_analysis.py [--no-skip-existing] [--n-jobs 1]

  1. Set the SESS variable to the session you wish to analyze. The spelling must match the session variable in file path (e.g. "ses-T3")
  2. Set all paths in the --CONFIGURE PATHS HERE-- section to the correct paths for your analysis. NOTE: the session number
     saved in the SESS variable is automatically added to the OUTPUT_DIR path along with a preceding "_"
  3. Specify the Brainnetome ROI or ROIs you wish to analyze in the AINS_LABELS list variable

  NOTE: A log of the output of this script will be found in the same directory as this script with the same filename except .log at the end
"""

import argparse
import glob
import itertools
import json
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

import nibabel as nib
from nilearn import image, plotting
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.maskers import NiftiMasker

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=Path(__file__).with_suffix('.log')
)
log = logging.getLogger(__name__)

# Which session do you want to analyze, currently only able to do one session per run
SESS = "ses-T3"

# ─── CONFIGURE PATHS HERE ────────────────────────────────────────────────────
BIDS_DIR    = Path("/Users/braveDP/Desktop/NEST-M/NESTM_bids")
DERIV_DIR   = BIDS_DIR / "derivatives"
EVENTS_ROOT = Path("/Users/braveDP/Desktop/NEST-M/Events")
OUTPUT_DIR  = Path("/Users/braveDP/Desktop/NEST-M/Outputs/MID_Analysis_Output"+"_"+SESS)
ATLAS_PATH  = Path("/Users/braveDP/Desktop/BN_Atlas_246_2mm.nii.gz")

# List of Brainnetome AIns label indices (167 = left AIns, 168 = right AIns)
AINS_LABELS = [167, 168]

# ── BEHAVIORAL COLUMN NAMES ──────────────────────────────────────────────────
# VERIFY these against your actual CSV before running.
# Open one _b1.csv and confirm the exact column names.
RT_COLUMN      = "rt"          # reaction time column in behavioral CSVs
ONSET_COLUMN   = "trialonset"  # trial onset (seconds) column
CUE_VAL_COLUMN = "cue_value"   # cue value string column ("+$5", "-$5", etc.)
TRIAL_COLUMN   = "trial"       # trial number column (for dedup)

# ── GLM PARAMETERS ───────────────────────────────────────────────────────────
# Anticipation epoch = cue (2 s) + fixation cross (2 s) = 4 s total,
# per the NESTM task design and MacNiven et al. (2024) model definition.
CUE_DURATION = 4.0   # seconds

# Six conditions matching Leili's trialtype codes:
#   trialtype 6 → +$5 (gain5),  trialtype 3 → -$5 (loss5)
#   trialtype 5 → +$1 (gain1),  trialtype 2 → -$1 (loss1)
#   trialtype 4 → +$0 (gain0),  trialtype 1 → -$0 (loss0)
# cue_value strings in the CSV are +$5, -$5, +$1, -$1, +$0, -$0 — all distinct.
CONDITIONS = ["gain5", "loss5", "gain1", "loss1", "gain0", "loss0"]

# Human-readable labels for plots (same order as CONDITIONS)
CONDITION_LABELS = {
    "gain5": "+$5",
    "loss5": "-$5",
    "gain1": "+$1",
    "loss1": "-$1",
    "gain0": "+$0",
    "loss0": "-$0",
}

# Confound columns (14 regressors matching Leili's motion + WM/CSF model)
CONFOUND_COLS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x",   "rot_y",   "rot_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1",   "rot_y_derivative1",   "rot_z_derivative1",
    "white_matter", "csf",
]

# All pairwise contrasts C(6,2) = 15
PAIRWISE_CONTRASTS = {
    f"{a}_vs_{b}": (a, b)
    for a, b in itertools.combinations(CONDITIONS, 2)
}

# Plot aesthetics for 4d bar chart
CONDITION_COLORS = {
    "gain5": "#2166AC",
    "gain1": "#74ADD1",
    "gain0": "#ABD9E9",
    "loss5": "#D73027",
    "loss1": "#F46D43",
    "loss0": "#FDAE61",
}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def safe_title(s: str) -> str:
    """Escape $ so matplotlib does not treat it as LaTeX."""
    return s.replace("$", r"\$")


def normalise_cue(raw: str) -> str | None:
    """
    Map raw cue_value strings → one of the 6 canonical condition names.

    The NESTM behavioral CSVs use +$0 and -$0 as distinct cue_value strings
    (trialtype 4 = +$0, trialtype 1 = -$0), so gain-neutral and loss-neutral
    are distinguishable and modelled separately, matching Leili's 6-regressor model.
    """
    raw = str(raw).strip()
    mapping = {
        "+$5":  "gain5",
        "-$5":  "loss5",
        "+$1":  "gain1",
        "-$1":  "loss1",
        "+$0":  "gain0",
        "-$0":  "loss0",
    }
    return mapping.get(raw)


def get_tr(json_path: Path) -> float:
    with open(json_path) as f:
        return float(json.load(f)["RepetitionTime"])


def build_events_df(csv_path: Path) -> pd.DataFrame:
    """
    Build a nilearn-compatible events DataFrame (onset / duration / trial_type)
    from a single behavioral run CSV.

    Anticipation epoch = 2 s from cue onset, matching Leili's model.
    """
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=TRIAL_COLUMN, keep="first").copy()
    df["trial_type"] = df[CUE_VAL_COLUMN].apply(normalise_cue)
    df = df.dropna(subset=["trial_type"])
    if df.empty:
        raise ValueError(
            f"No valid trials found in {csv_path.name}. "
            f"Check CUE_VAL_COLUMN ('{CUE_VAL_COLUMN}') and normalise_cue() mappings."
        )
    return (
        pd.DataFrame({
            "onset":      df[ONSET_COLUMN].values,
            "duration":   CUE_DURATION,
            "trial_type": df["trial_type"].values,
        })
        .sort_values("onset")
        .reset_index(drop=True)
    )


def extract_rt_per_condition(csv_paths: list[Path]) -> dict[str, float]:
    """
    Compute mean RT per condition across all provided run CSVs
    (both runs concatenated, matching Leili's concat approach for behavioral data).

    Returns dict: {condition_name: mean_rt_seconds}
    Missing conditions (no trials) are excluded from the dict.
    """
    frames = []
    for p in csv_paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            log.warning("    Could not read RT CSV %s: %s", p.name, e)

    if not frames:
        return {}

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=TRIAL_COLUMN, keep="first").copy()
    df["condition"] = df[CUE_VAL_COLUMN].apply(normalise_cue)
    df = df.dropna(subset=["condition"])

    if RT_COLUMN not in df.columns:
        log.warning(
            "    RT column '%s' not found in behavioral CSVs. "
            "Available columns: %s\n"
            "    Update RT_COLUMN at the top of this script.",
            RT_COLUMN, list(df.columns),
        )
        return {}

    df = df.dropna(subset=[RT_COLUMN])
    # Keep only trials where participant actually responded (RT > 0)
    df[RT_COLUMN] = pd.to_numeric(df[RT_COLUMN], errors='coerce')
    df = df[df[RT_COLUMN] > 0]
    # Convert RT from seconds to milliseconds before averaging
    df[RT_COLUMN] = df[RT_COLUMN] * 1000

    rt_dict = (
        df.groupby("condition")[RT_COLUMN]
        .mean()
        .to_dict()
    )
    return rt_dict


def load_confounds(conf_path: Path, n_scans: int) -> pd.DataFrame:
    conf = pd.read_csv(conf_path, sep="\t")
    cols = [c for c in CONFOUND_COLS if c in conf.columns]
    missing = set(CONFOUND_COLS) - set(cols)
    if missing:
        log.warning("  Confound columns absent (filled with 0): %s", sorted(missing))
    conf = conf[cols].fillna(0.0)
    if len(conf) != n_scans:
        raise ValueError(
            f"Confound rows ({len(conf)}) ≠ BOLD volumes ({n_scans})"
        )
    return conf


def make_ains_mask(
    atlas_path: Path,
    labels: list[int],
    target_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    atlas = nib.load(str(atlas_path))
    data  = np.asarray(atlas.dataobj, dtype=np.int32)
    mask  = np.zeros_like(data, dtype=np.uint8)
    for lbl in labels:
        mask[data == lbl] = 1
    mask_img = nib.Nifti1Image(mask, atlas.affine, atlas.header)
    return image.resample_to_img(mask_img, target_img, interpolation="nearest")


# ─── FILE DISCOVERY ──────────────────────────────────────────────────────────

def _find_events_csv(
    subj_id_lower: str,
    session: str,
    run_num: str,
    events_root: Path,
) -> Path | None:
    """
    Locate the behavioral CSV for (subject, session, run).
    Handles every naming variant in the NESTM data tree
    (h4m001_b1.csv, nestm_h4m001_b2.csv, h4m007t1_b2.csv, H4M012T1_b2.csv, …)
    """
    run_suffix = f"_b{run_num}.csv"
    tp_num     = session.replace("ses-T", "")

    subj_folder = None
    for d in events_root.iterdir():
        if d.is_dir() and d.name.lower() == subj_id_lower:
            subj_folder = d
            break
    if subj_folder is None:
        log.debug("    No events folder for %s", subj_id_lower)
        return None

    tp_dirs = [
        d for d in subj_folder.iterdir()
        if d.is_dir() and re.search(r"(timepoint|t)\s*" + tp_num, d.name, re.I)
    ]
    search_dirs = tp_dirs if tp_dirs else [subj_folder]

    candidates = []
    for search_dir in search_dirs:
        for csv in search_dir.rglob("*.csv"):
            name_lower = csv.name.lower()
            if name_lower.endswith("_p.csv"):
                continue
            if re.search(r"test_?p\.csv$", name_lower):
                continue
            if name_lower.endswith("_m.csv"):
                continue
            if subj_id_lower not in name_lower:
                continue
            if not name_lower.endswith(run_suffix):
                continue
            candidates.append(csv)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        log.warning(
            "  Multiple CSVs matched for %s %s run-%s – using first alphabetically:\n    %s",
            subj_id_lower, session, run_num,
            "\n    ".join(str(c) for c in sorted(candidates)),
        )
        return sorted(candidates)[0]

    log.warning(
        "  No events CSV for %s %s run-%s (suffix=%s, dirs=%s)",
        subj_id_lower, session, run_num, run_suffix,
        [str(d) for d in search_dirs],
    )
    return None


def discover_runs() -> list[dict]:
    """
    Walk fMRIPrep derivatives for preprocessed MID BOLD files and match each
    to its behavioral CSV, confounds TSV, and BIDS JSON.
    Returns a list of run-info dicts ready for GLM fitting.
    """
    bold_glob = str(
        DERIV_DIR /
        "sub-*" / "ses-T*" / "func" /
        "*_task-mid_run-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    )
    bold_files = sorted(Path(p) for p in glob.glob(bold_glob))
    log.info("fMRIPrep BOLD files found: %d", len(bold_files))

    if not bold_files:
        log.error(
            "No preprocessed BOLD files under:\n  %s\n"
            "Check DERIV_DIR and that fMRIPrep has been run.",
            DERIV_DIR,
        )
        return []

    runs = []
    for bold_nii in bold_files:
        fname = bold_nii.name
        m = re.match(r"(sub-\S+?)_(ses-T\d+)_task-mid_run-(\d+)_", fname)
        if not m:
            log.warning("  Cannot parse BIDS entities from: %s", fname)
            continue

        sub_id  = m.group(1)   # e.g. sub-H4M007
        session = m.group(2)   # e.g. ses-T1
        run_num = m.group(3)   # "1" or "2"

        #choose session or sessions to analyze
        if session == SESS:
            subj_id_lower = sub_id.replace("sub-", "").lower()  # h4m007

            # Confounds TSV: same folder as BOLD, no space/res entities
            conf_fname = re.sub(
                r"_(space|res)-.*_desc-preproc_bold\.nii\.gz",
                "_desc-confounds_timeseries.tsv",
                fname,
            )
            conf_path = bold_nii.parent / conf_fname

            # Raw BIDS JSON (supplies TR)
            json_path = (
                BIDS_DIR / sub_id / session / "func" /
                f"{sub_id}_{session}_task-mid_run-{run_num}_bold.json"
            )

            # Behavioral events CSV
            events_csv = _find_events_csv(subj_id_lower, session, run_num, EVENTS_ROOT)

            missing = []
            if not conf_path.exists():
                missing.append(f"confounds TSV : {conf_path}")
            if not json_path.exists():
                missing.append(f"BIDS JSON     : {json_path}")
            if events_csv is None:
                missing.append("events CSV    : (not found – see warning above)")

            if missing:
                log.warning(
                    "Skipping %s %s run-%s – missing:\n  %s",
                    sub_id, session, run_num, "\n  ".join(missing),
                )
                continue

            runs.append({
                "subject":       sub_id,
                "session":       session,
                "run_num":       run_num,
                "events_csv":    events_csv,
                "bold_nii":      bold_nii,
                "bold_json":     json_path,
                "confounds_tsv": conf_path,
            })
            log.info(
                "Discovered: %s %s run-%s  ← %s",
                sub_id, session, run_num, events_csv.name,
            )
        else:
            log.info("Skipping non-first session")
    log.info("Total analysable runs: %d", len(runs))
    return runs


# ─── FIRST-LEVEL GLM ─────────────────────────────────────────────────────────

def run_first_level(
    run_info: dict,
    out_root: Path,
    skip_existing: bool = True,
) -> None:
    """
    Fit a first-level GLM for one run.

    Saves:
      • Per-condition beta maps:  {cond}_condition_effect_size.nii.gz
      • Per-condition z maps:     {cond}_condition_z_score.nii.gz
      • Pairwise contrast maps:   {a}_vs_{b}_effect_size.nii.gz  (+ z + glass brain)
      • Design matrix plot
    """
    sub    = run_info["subject"]
    ses    = run_info["session"]
    run    = run_info["run_num"]
    tag    = f"{sub}_{ses}_run-{run}"
    outdir = out_root / "first_level" / tag

    if skip_existing and outdir.exists():
        log.info("Skipping (already done): %s", tag)
        return

    outdir.mkdir(parents=True, exist_ok=True)
    log.info("Fitting first-level GLM: %s", tag)

    tr       = get_tr(run_info["bold_json"])
    bold_img = nib.load(str(run_info["bold_nii"]))
    n_scans  = bold_img.shape[3]
    log.info("  TR=%.3f s   volumes=%d", tr, n_scans)

    events    = build_events_df(run_info["events_csv"])
    confounds = load_confounds(run_info["confounds_tsv"], n_scans)

    log.info(
        "  Condition trial counts:\n%s",
        events["trial_type"].value_counts().to_string(),
    )

    # ── Fit GLM ──────────────────────────────────────────────────────────────
    glm = FirstLevelModel(
        t_r             = tr,
        hrf_model       = "spm",
        drift_model     = "cosine",
        high_pass       = 1 / 128,
        noise_model     = "ar1",
        standardize     = False,
        signal_scaling  = 0,
        mask_img        = None,
        minimize_memory = True,
        n_jobs          = 1,
    )
    glm.fit(bold_img, events=events, confounds=confounds)

    # Design matrix plot
    dm  = glm.design_matrices_[0]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(dm.values, aspect="auto", interpolation="none", cmap="gray")
    ax.set_xticks(range(len(dm.columns)))
    ax.set_xticklabels(dm.columns, rotation=90, fontsize=7)
    ax.set_title(f"Design matrix – {tag}")
    fig.tight_layout()
    fig.savefig(str(outdir / "design_matrix.png"), dpi=120)
    plt.close(fig)

    col_index = {col: i for i, col in enumerate(dm.columns)}
    n_cols    = len(dm.columns)
    dm_conds  = [c for c in CONDITIONS if c in dm.columns]

    # ── Per-condition (identity) beta maps ───────────────────────────────────
    # These are the primary outputs for the RT-correlation analysis.
    # One +1 contrast vector per condition → direct estimate of that condition's
    # anticipatory BOLD response, analogous to Leili's gainant / lossant regressors.
    for cond in CONDITIONS:
        if cond not in col_index:
            log.warning("  Condition '%s' absent from design for %s – skipping", cond, tag)
            continue
        con_vec = np.zeros(n_cols)
        con_vec[col_index[cond]] = 1.0

        eff  = glm.compute_contrast(con_vec, output_type="effect_size")
        zmap = glm.compute_contrast(con_vec, output_type="z_score")
        eff.to_filename( str(outdir / f"{cond}_condition_effect_size.nii.gz"))
        zmap.to_filename(str(outdir / f"{cond}_condition_z_score.nii.gz"))

    log.info("  Saved per-condition beta maps for: %s", dm_conds)

    # ── Pairwise contrast maps ────────────────────────────────────────────────
    for cname, (cond_a, cond_b) in PAIRWISE_CONTRASTS.items():
        if cond_a not in dm_conds or cond_b not in dm_conds:
            log.warning("  Skipping contrast %s – condition missing", cname)
            continue

        con_vec = np.zeros(n_cols)
        con_vec[col_index[cond_a]] =  1.0
        con_vec[col_index[cond_b]] = -1.0

        eff  = glm.compute_contrast(con_vec, output_type="effect_size")
        zmap = glm.compute_contrast(con_vec, output_type="z_score")
        eff.to_filename( str(outdir / f"{cname}_effect_size.nii.gz"))
        zmap.to_filename(str(outdir / f"{cname}_z_score.nii.gz"))

        fig = plt.figure(figsize=(10, 3))
        plotting.plot_glass_brain(
            zmap, colorbar=True, threshold=2.3,
            title=f"{safe_title(cname)} z>2.3 – {tag}",
            figure=fig, display_mode="lyrz",
        )
        fig.savefig(str(outdir / f"{cname}_glass_brain.png"), dpi=100)
        plt.close(fig)

    log.info("  Saved pairwise contrasts → %s", outdir)


# ─── GROUP-LEVEL GLM ─────────────────────────────────────────────────────────

def run_group_level(all_fl_dirs: list[Path], out_root: Path) -> None:
    log.info("=== Group-level analysis ===")
    group_dir = out_root / "group_level"
    group_dir.mkdir(parents=True, exist_ok=True)

    for cname in PAIRWISE_CONTRASTS:
        log.info("  Contrast: %s", cname)
        cdir = group_dir / cname
        cdir.mkdir(exist_ok=True)

        session_maps: dict[tuple, list] = {}
        for fl_dir in all_fl_dirs:
            tag   = fl_dir.name
            parts = tag.split("_")
            sub, ses = parts[0], parts[1]
            eff = fl_dir / f"{cname}_effect_size.nii.gz"
            if eff.exists():
                session_maps.setdefault((sub, ses), []).append(nib.load(str(eff)))

        if not session_maps:
            log.warning("  No data for %s – skipping", cname)
            continue

        second_level_imgs = []
        for (sub, ses), imgs in sorted(session_maps.items()):
            second_level_imgs.append(
                imgs[0] if len(imgs) == 1 else image.mean_img(imgs)
            )

        n = len(second_level_imgs)
        log.info("    N sessions = %d", n)
        if n < 2:
            log.warning("    Too few sessions for group test – skipping")
            continue

        dm2  = pd.DataFrame({"intercept": np.ones(n)})
        glm2 = SecondLevelModel(smoothing_fwhm=6.0)
        glm2.fit(second_level_imgs, design_matrix=dm2)

        z_map    = glm2.compute_contrast("intercept", output_type="z_score")
        stat_map = glm2.compute_contrast("intercept", output_type="stat")
        z_map.to_filename(   str(cdir / "z_map.nii.gz"))
        stat_map.to_filename(str(cdir / "stat_map.nii.gz"))

        fig = plt.figure(figsize=(12, 4))
        plotting.plot_glass_brain(
            z_map, colorbar=True, threshold=2.3,
            title=f"Group: {safe_title(cname)}  (N={n}, z>2.3)",
            figure=fig, display_mode="lyrz", plot_abs=False,
        )
        fig.savefig(str(cdir / "glass_brain.png"), dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 4))
        plotting.plot_stat_map(
            z_map, threshold=2.3, colorbar=True,
            title=f"Group: {safe_title(cname)}  (z>2.3, N={n})",
            axes=ax, display_mode="z",
            cut_coords=[-20, -10, 0, 10, 20, 30],
        )
        fig.savefig(str(cdir / "stat_map_slices.png"), dpi=120)
        plt.close(fig)

        log.info("    Saved → %s", cdir)


# ─── ROI ANALYSIS (AIns) ─────────────────────────────────────────────────────

def run_roi_analysis(
    all_fl_dirs: list[Path],
    runs: list[dict],
    out_root: Path,
) -> pd.DataFrame | None:
    """
    Extract mean AIns beta per condition per run, average within session,
    merge with mean RT per condition per session, and save CSVs.

    Returns the merged DataFrame (subject × session × condition) or None on failure.
    """
    log.info("=== ROI analysis (AIns) ===")
    roi_dir = out_root / "roi"
    roi_dir.mkdir(parents=True, exist_ok=True)

    if not ATLAS_PATH.exists():
        log.error(
            "Brainnetome atlas not found at:\n  %s\n"
            "Download BN_Atlas_246_2mm.nii.gz from "
            "https://atlas.brainnetome.org and update ATLAS_PATH.",
            ATLAS_PATH,
        )
        return None

    # Build AIns mask using first available condition map as spatial reference
    ref_img = None
    for fl_dir in all_fl_dirs:
        for p in fl_dir.glob("*_condition_effect_size.nii.gz"):
            ref_img = nib.load(str(p))
            break
        if ref_img is not None:
            break
    if ref_img is None:
        log.error("No first-level condition maps found – cannot build AIns mask.")
        return None

    log.info("  Building AIns mask from Brainnetome atlas labels %s …", AINS_LABELS)
    ains_mask = make_ains_mask(ATLAS_PATH, AINS_LABELS, ref_img)
    ains_mask.to_filename(str(roi_dir / "AIns_mask.nii.gz"))
    n_vox = int(np.asarray(ains_mask.dataobj).sum())
    log.info("  AIns mask saved  (voxels = %d)", n_vox)
    if n_vox == 0:
        log.error(
            "AIns mask is empty. Check that AINS_LABELS %s are correct for your atlas.",
            AINS_LABELS,
        )
        return None

    masker = NiftiMasker(
        mask_img=ains_mask, standardize=False,
        memory="nilearn_cache", memory_level=1,
    )
    masker.fit()

    # ── Extract per-run betas ─────────────────────────────────────────────────
    rows = []
    for fl_dir in sorted(all_fl_dirs):
        tag   = fl_dir.name
        parts = tag.split("_")
        sub, ses, run = parts[0], parts[1], parts[2]
        for cond in CONDITIONS:
            eff_path = fl_dir / f"{cond}_condition_effect_size.nii.gz"
            if not eff_path.exists():
                continue
            signals   = masker.transform(str(eff_path))
            mean_beta = float(np.nanmean(signals))
            rows.append({
                "subject":        sub,
                "session":        ses,
                "run":            run,
                "condition":      cond,
                "AIns_mean_beta": mean_beta,
            })

    if not rows:
        log.warning("No ROI condition betas extracted.")
        return None

    df_runs = pd.DataFrame(rows)
    df_runs.to_csv(roi_dir / "AIns_betas_per_run.csv", index=False)
    log.info("  Saved AIns_betas_per_run.csv  (%d rows)", len(df_runs))

    # ── Average across runs within session (matches Leili's concatenated approach) ──
    df_ses = (
        df_runs
        .groupby(["subject", "session", "condition"])["AIns_mean_beta"]
        .mean()
        .reset_index()
        .rename(columns={"AIns_mean_beta": "AIns_mean_beta_ses_avg"})
    )
    df_ses.to_csv(roi_dir / "AIns_betas_per_session.csv", index=False)

    # ── Build RT lookup: subject × session × condition → mean RT ─────────────
    # For each subject-session, find both run CSVs and compute mean RT per cond
    # (concatenating runs exactly as Leili concatenated runs for her behavioral model)
    rt_rows = []
    for run_info in runs:
        sub_ses_key = (run_info["subject"], run_info["session"])
        rt_rows.append((sub_ses_key, run_info["events_csv"]))

    # Group csv paths by (subject, session)
    csv_by_subses: dict[tuple, list[Path]] = {}
    for (sub_ses, csv_path) in rt_rows:
        csv_by_subses.setdefault(sub_ses, []).append(csv_path)

    rt_data = []
    for (sub, ses), csv_paths in sorted(csv_by_subses.items()):
        rt_dict = extract_rt_per_condition(csv_paths)
        for cond, mean_rt in rt_dict.items():
            rt_data.append({
                "subject":  sub,
                "session":  ses,
                "condition": cond,
                "mean_RT":  mean_rt,
            })

    if not rt_data:
        log.warning(
            "  No RT data extracted. Check RT_COLUMN ('%s') is correct.", RT_COLUMN
        )
        df_merged = df_ses.copy()
        df_merged["mean_RT"] = np.nan
    else:
        df_rt = pd.DataFrame(rt_data)
        df_merged = df_ses.merge(df_rt, on=["subject", "session", "condition"], how="left")

    df_merged.to_csv(roi_dir / "AIns_RT_merged.csv", index=False)
    log.info("  Saved AIns_RT_merged.csv  (%d rows)", len(df_merged))

    # ── Summary stats ─────────────────────────────────────────────────────────
    summary = (
        df_ses
        .groupby("condition")["AIns_mean_beta_ses_avg"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_beta", "std": "sd_beta", "count": "n_sessions"})
    )
    summary["sem_beta"] = summary["sd_beta"] / np.sqrt(summary["n_sessions"])
    # Preserve condition order
    summary["condition"] = pd.Categorical(summary["condition"], categories=CONDITIONS, ordered=True)
    summary = summary.sort_values("condition").reset_index(drop=True)
    summary.to_csv(roi_dir / "AIns_betas_summary_stats.csv", index=False)

    log.info("  ROI outputs saved → %s", roi_dir)
    return df_merged


# ─── FIGURES 4C AND 4D ───────────────────────────────────────────────────────

def make_figures(df_merged: pd.DataFrame, out_root: Path) -> None:
    """
    Generate figures matching MacNiven et al. (2024) Figures 4c and 4d.

    Figure 4c:
        Scatter plot of mean RT (x) vs. AIns loss5 beta (y) across subjects.
        Each point = one subject (averaging across sessions if multiple).
        Regression line + 95 % CI band overlaid.

    Figure 4d:
        Bar graph of the regression coefficient (slope) from an independent
        simple linear regression (AIns_beta ~ mean_RT) fit separately for each
        of the 6 trial types.  Error bars = standard error of the slope.
        Mirrors the paper's approach of running 6 separate regressions.
    """
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if df_merged is None or df_merged.empty:
        log.error("No merged data available – cannot generate figures.")
        return

    required_cols = {"subject", "session", "condition", "AIns_mean_beta_ses_avg", "mean_RT"}
    missing_cols  = required_cols - set(df_merged.columns)
    if missing_cols:
        log.error("Merged DataFrame missing columns: %s", missing_cols)
        return

    if df_merged["mean_RT"].isna().all():
        log.warning(
            "All RT values are NaN. Figures will show betas only (no RT axis).\n"
            "Check RT_COLUMN in the script configuration."
        )

    # ── Aggregate to one row per subject × condition (mean across sessions) ──
    df_subj = (
        df_merged
        .groupby(["subject", "condition"])[["AIns_mean_beta_ses_avg", "mean_RT"]]
        .mean()
        .reset_index()
    )

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 4c: Scatter – RT vs. AIns loss5 anticipation beta
    # ──────────────────────────────────────────────────────────────────────────
    df_4c = df_subj[df_subj["condition"] == "loss5"].dropna(
        subset=["AIns_mean_beta_ses_avg", "mean_RT"]
    ).copy()

    fig4c, ax4c = plt.subplots(figsize=(5, 5))

    if len(df_4c) < 3:
        log.warning(
            "Figure 4c: only %d subjects with loss5 data + RT – scatter will be sparse.",
            len(df_4c),
        )

    # Paper Figure 4C: x = AIns anticipatory activity, y = Reaction Time
    # Regression: RT ~ AIns_beta  (matches paper Fig 4D y-axis:
    # "Reaction time ~ AIns activity (Coefficient ± SEM)")
    ax4c.scatter(
        df_4c["AIns_mean_beta_ses_avg"],
        df_4c["mean_RT"],
        color="#D73027",
        s=60,
        zorder=3,
        edgecolors="white",
        linewidths=0.5,
        label="Subjects",
    )

    if len(df_4c) >= 3:
        slope, intercept, r_value, p_value, se_slope = stats.linregress(
            df_4c["AIns_mean_beta_ses_avg"], df_4c["mean_RT"]
        )
        x_range = np.linspace(df_4c["AIns_mean_beta_ses_avg"].min(),
                              df_4c["AIns_mean_beta_ses_avg"].max(), 200)
        y_fit   = slope * x_range + intercept

        # 95 % confidence band via bootstrap (1000 resamples)
        # Skip any resample where all x values are identical (can happen with small N)
        n_boot = 1000
        rng    = np.random.default_rng(42)
        boot_y = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(df_4c), size=len(df_4c))
            bs  = df_4c.iloc[idx]
            if bs["AIns_mean_beta_ses_avg"].nunique() < 2:
                continue
            try:
                s, i, *_ = stats.linregress(bs["AIns_mean_beta_ses_avg"], bs["mean_RT"])
                boot_y.append(s * x_range + i)
            except ValueError:
                continue

        ax4c.plot(x_range, y_fit, color="#D73027", linewidth=2, zorder=4)

        if len(boot_y) >= 10:
            boot_y = np.array(boot_y)
            ci_lo  = np.percentile(boot_y, 2.5,  axis=0)
            ci_hi  = np.percentile(boot_y, 97.5, axis=0)
            ax4c.fill_between(x_range, ci_lo, ci_hi, color="#D73027", alpha=0.15, zorder=2)
        else:
            log.warning(
                "  Too few valid bootstrap resamples (%d) for CI band – "
                "likely due to small N. CI omitted.", len(boot_y)
            )

        r2_str = f"r = {r_value:.2f},  p = {p_value:.3f}"
        ax4c.text(
            0.05, 0.95, r2_str,
            transform=ax4c.transAxes,
            fontsize=9, va="top",
            color="#D73027",
        )
        log.info(
            "Figure 4c  loss5: slope=%.4f  r=%.3f  p=%.4f  n=%d",
            slope, r_value, p_value, len(df_4c),
        )
    else:
        log.warning("Too few points for regression in Figure 4c.")

    ax4c.axvline(0, color="gray", linewidth=0.7, linestyle="--", zorder=1)
    ax4c.set_xlabel("AIns Loss Anticipation Activity\n(−$5 beta)", fontsize=11) #editHere
    ax4c.set_ylabel("Mean Reaction Time (ms)", fontsize=11)
    ax4c.set_title("AIns −$5 anticipation vs. RT", fontsize=11) #editHere
    ax4c.spines["top"].set_visible(False)
    ax4c.spines["right"].set_visible(False)
    fig4c.tight_layout()
    fig4c.savefig(str(fig_dir / "figure_4c.png"), dpi=200)
    plt.close(fig4c)
    log.info("  Saved figure_4c.png")

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 4d: Bar graph – regression coefficients for all 6 trial types
    # Each bar = slope from an independent OLS regression of
    #   AIns_beta ~ mean_RT  across subjects, for that condition alone.
    # Error bars = standard error of the slope.
    # ──────────────────────────────────────────────────────────────────────────
    coef_rows = []
    for cond in CONDITIONS:
        df_c = df_subj[df_subj["condition"] == cond].dropna(
            subset=["AIns_mean_beta_ses_avg", "mean_RT"]
        )
        if len(df_c) < 3:
            log.warning(
                "  Figure 4d: condition '%s' has only %d subjects – skipping regression.",
                cond, len(df_c),
            )
            coef_rows.append({"condition": cond, "slope": np.nan, "se": np.nan, "p": np.nan, "n": len(df_c)})
            continue

        # Paper Fig 4D: regression is RT ~ AIns_beta
        # (y-axis: "Reaction time ~ AIns activity (Coefficient ± SEM)")
        slope, intercept, r_value, p_value, se_slope = stats.linregress(
            df_c["AIns_mean_beta_ses_avg"], df_c["mean_RT"]
        )
        coef_rows.append({
            "condition": cond,
            "slope":     slope,
            "se":        se_slope,
            "p":         p_value,
            "n":         len(df_c),
            "r":         r_value,
        })
        log.info(
            "  Figure 4d  %-6s: slope=%+.4f  SE=%.4f  r=%.3f  p=%.4f  n=%d",
            cond, slope, se_slope, r_value, p_value, len(df_c),
        )

    df_coef = pd.DataFrame(coef_rows)
    df_coef.to_csv(fig_dir / "figure_4d_regression_coefficients.csv", index=False)

    fig4d, ax4d = plt.subplots(figsize=(7, 5))

    x      = np.arange(len(CONDITIONS))
    bar_w  = 0.6

    for xi, cond in enumerate(CONDITIONS):
        row = df_coef[df_coef["condition"] == cond].iloc[0]
        if np.isnan(row["slope"]):
            continue
        color = CONDITION_COLORS[cond]
        ax4d.bar(
            xi, row["slope"],
            width=bar_w,
            color=color,
            yerr=row["se"],
            capsize=5,
            error_kw={"linewidth": 1.5, "ecolor": "black"},
            zorder=3,
        )
        # Significance asterisk if p < 0.05
        if row["p"] < 0.05:
            ypos = row["slope"] + (row["se"] * 1.3 if row["slope"] >= 0 else -row["se"] * 1.3)
            ax4d.text(xi, ypos, "*", ha="center", va="bottom" if row["slope"] >= 0 else "top",
                      fontsize=14, color="black")

    ax4d.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax4d.set_xticks(x)
    ax4d.set_xticklabels(
        [CONDITION_LABELS[c] for c in CONDITIONS],
        fontsize=11,
    )
    ax4d.set_xlabel("Trial Type", fontsize=11)
    ax4d.set_ylabel("Regression Coefficient\n(Reaction time ~ AIns activity)", fontsize=11) #editHere
    ax4d.set_title("AIns anticipation–RT regression by trial type", fontsize=11) #editHere
    ax4d.spines["top"].set_visible(False)
    ax4d.spines["right"].set_visible(False)
    fig4d.tight_layout()
    fig4d.savefig(str(fig_dir / "figure_4d.png"), dpi=200)
    plt.close(fig4d)
    log.info("  Saved figure_4d.png")
    log.info("  Regression coefficients saved → %s", fig_dir / "figure_4d_regression_coefficients.csv")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NESTM MID fMRI Analysis Pipeline – MacNiven et al. Figs 4c/4d"
    )
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing",
        action="store_false", default=True,
        help="Re-run first-level models even if output already exists.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel jobs for nilearn (default 1).",
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Restrict to one session, e.g. ses-T1  (default: all sessions).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", OUTPUT_DIR)

    runs = discover_runs()

    if args.session:
        runs = [r for r in runs if r["session"] == args.session]
        log.info("Filtered to session '%s': %d runs", args.session, len(runs))

    if not runs:
        log.error(
            "No runs discovered.\n"
            "  • Check DERIV_DIR: %s\n"
            "  • Check EVENTS_ROOT: %s\n"
            "  • Check BIDS_DIR: %s",
            DERIV_DIR, EVENTS_ROOT, BIDS_DIR,
        )
        return

    # ── First-level GLMs ──────────────────────────────────────────────────────
    all_fl_dirs: list[Path] = []
    for run_info in runs:
        try:
            run_first_level(run_info, OUTPUT_DIR, skip_existing=args.skip_existing)
            fl_dir = (
                OUTPUT_DIR / "first_level" /
                f"{run_info['subject']}_{run_info['session']}_run-{run_info['run_num']}"
            )
            if fl_dir.exists():
                all_fl_dirs.append(fl_dir)
        except Exception as exc:
            log.error(
                "First-level failed for %s %s run-%s: %s",
                run_info["subject"], run_info["session"],
                run_info["run_num"], exc, exc_info=True,
            )

    if not all_fl_dirs:
        log.error("No first-level results produced.")
        return

    # ── Group-level GLMs ──────────────────────────────────────────────────────
    try:
        run_group_level(all_fl_dirs, OUTPUT_DIR)
    except Exception as exc:
        log.error("Group-level failed: %s", exc, exc_info=True)

    # ── ROI analysis + RT extraction ──────────────────────────────────────────
    df_merged = None
    try:
        df_merged = run_roi_analysis(all_fl_dirs, runs, OUTPUT_DIR)
    except Exception as exc:
        log.error("ROI analysis failed: %s", exc, exc_info=True)

    # ── Figures 4c and 4d ─────────────────────────────────────────────────────
    try:
        make_figures(df_merged, OUTPUT_DIR)
    except Exception as exc:
        log.error("Figure generation failed: %s", exc, exc_info=True)

    log.info("=== Pipeline complete.  Results in: %s ===", OUTPUT_DIR)


if __name__ == "__main__":
    main()
