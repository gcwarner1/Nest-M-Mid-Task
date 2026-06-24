# Analyzing the MID task

The following is a python based pipeline used to analyze the MID task collected as part of Stanford's NEST-M study. This package is free to use but keep in mind that it was built specifically for our dataset. As such, it may require significant editing in order to make it work properly on your data. In particular, be sure to check things like directory paths, file names, preprocessing parameters (i.e. high-pass filter cutoff, gaussian smoothing kernel size, TR, and events file design) before running. 

## Before Analysis (structuring data, file naming, installing depedencies)
1. Download and install [fMRIPrep](https://github.com/nipreps/fmriprep) (I recommend using their [docker container](https://fmriprep.org/en/20.2.0/docker.html))
2. Download data from Flywheel
3. Run dcm2bids (or dcm2niix and then manually convert to BIDS format)
4. Run `moveBadScans.py` in order to exclude any incomplete scans and rename completed scans
5. If BIDS structure is faulty in any way run `create_bids_json.py` and/or `generateDatasetDescriptions.py`
6. Move events files (e.g. H4M001_b1.csv) to `/Users/braveDP/Desktop/NEST-M/Events` or equivalent directory path

## Running the pipeline
1. Preprocess data using `fmriPrep_preprocess.sh`
2. QC fmriPrep results
3. Analyze MID task by running `mid_analysis.py`
4. QC analysis results in the `Outputs` directory
