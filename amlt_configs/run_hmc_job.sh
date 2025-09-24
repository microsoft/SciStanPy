#!/bin/bash
# This script is used to run a job on AMLT using the HMC method.

# Handle arguments
job_ind=$1
rate_dist=$2
growth_func=$3

# Pull the dataset information from the dsets.txt file
IFS=" " read -r -a dset_info <<< "$(sed -n "${job_ind}p" dsets.txt)"
dataset="${dset_info[0]}"
subset="${dset_info[1]}"

# Switch to the code directory
cd /opt/SciStanPy/scistanpy/pipelines || exit

# Report the job details
echo "Running MCMC for $dataset-$subset with the growth modeled by $growth_func \
and the growth rate modeled by the $rate_dist distribution."

# Run the HMC job
python mcmc_flip.py \
    --dataset "$dataset" \
    --subset "$subset" \
    --growth_rate "$rate_dist" \
    --growth_curve "$growth_func" \
    --flip_data "$AMLT_DATA_DIR" \
    --output_dir "$AMLT_DIRSYNC_DIR" \
    --use_dask \
    --force_compile \
    --ignore_od