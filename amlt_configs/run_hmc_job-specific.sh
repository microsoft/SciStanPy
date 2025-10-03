#!/bin/bash
# This script is used to run a job on AMLT using the HMC method. All arguments
# are passed in from the AMLT job submission script.

# Handle arguments
job_ind=$1

# Pull the dataset information from the cmds.txt file
IFS=" " read -r -a cmd <<< "$(sed -n "${job_ind}p" cmds-small.txt)"

# Switch to the code directory
cd /opt/SciStanPy/scistanpy/pipelines || exit

# Report the job details
printf 'Full command: %s\n' "${cmd[@]}"

# Run the HMC job
python mcmc_flip.py \
    "${cmd[@]}" \
    --flip_data "$AMLT_DATA_DIR" \
    --output_dir "$AMLT_DIRSYNC_DIR" \
    --use_dask \
    --force_compile