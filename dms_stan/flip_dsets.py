"""Holds code relevant for the TrpB datasets."""

import numpy as np
import numpy.typing as npt
import pandas as pd

from Bio.Seq import Seq


def load_trpb_dataset(filepath: str) -> dict[str, npt.NDArray | list[str]]:
    """
    Load a TrpB dataset from Johnston et al.
    """

    def get_cols(coltype: str) -> list[str]:
        """Get the columns of a given type."""
        return sorted(
            (col for col in data.columns if col.startswith(coltype)),
            key=lambda x: int(x.split("_")[1]),
        )

    # Load in the data
    data = pd.read_csv(filepath)

    # Get the input and output columns
    input_cols, output_cols = get_cols("InputCount"), get_cols("OutputCount")

    # Get the input count data. The input counts should be repeated across all
    # timepoints. We assert that this is the case.
    t0_data = data[["AAs"] + input_cols].drop_duplicates()
    assert (t0_data.AAs.value_counts() == 1).all()

    # Get the input counts and the unique combinations
    t0_counts = t0_data[input_cols].to_numpy(dtype=int).T
    combo_order = t0_data.AAs.tolist()

    # If the first dimension of the input counts is '1', we can remove it. This
    # means that there were no replicates of the input counts. Otherwise, we need
    # to insert a new axis to handle the timepoint dimension.
    if t0_counts.shape[0] == 1:
        t0_counts = t0_counts[0]
    else:
        t0_counts = t0_counts[:, None]

    # Get the timepoint counts
    times = data["Time (h)"].unique().astype(float)
    times.sort()
    tg0_counts = np.zeros([len(output_cols), len(times), len(combo_order)], dtype=int)
    for timeind, time in enumerate(times):

        # Filter down to just the data for this time
        time_data = data[data["Time (h)"] == time]

        # Make sure the data is in the right order
        assert time_data.AAs.tolist() == combo_order

        # Get the counts
        tg0_counts[:, timeind, :] = time_data[output_cols].to_numpy(dtype=int).T

    return {
        "times": times,
        "starting_counts": t0_counts,
        "timepoint_counts": tg0_counts,
        "variants": combo_order,
    }


def load_pdz_dataset(filepath: str) -> dict[str, npt.NDArray[np.int64]]:
    """Loads a given PDZ3 dataset file and returns the starting and ending counts.
    This function is appropriate for the files sent by Taraneh Zarin.

    Args:
        filepath (str): Path to the file to load.

    Returns:
        dict[str, npt.NDArray[np.int64]]: A dictionary with two keys:
            'starting_counts' and 'ending_counts', each containing a numpy array of
            the respective counts.
    """
    # Load the data. We are grouping all data belonging to unique amino acid sequences
    # together by summing the counts
    df = (
        pd.read_csv(
            filepath,
            sep="\t",
            usecols=(
                "aa_seq",
                "input1_e1_s0_bNA_count",
                "input2_e2_s0_bNA_count",
                "input3_e3_s0_bNA_count",
                "output1A_e1_s1_b1_count",
                "output2A_e2_s1_b1_count",
                "output3A_e3_s1_b1_count",
            ),
        )
        .groupby("aa_seq")
        .sum()
        .reset_index()
    )

    # Get numpy arrays for the starting and ending counts
    return {
        "starting_counts": df[
            [
                "input1_e1_s0_bNA_count",
                "input2_e2_s0_bNA_count",
                "input3_e3_s0_bNA_count",
            ]
        ]
        .to_numpy()
        .T,
        "timepoint_counts": df[
            [
                "output1A_e1_s1_b1_count",
                "output2A_e2_s1_b1_count",
                "output3A_e3_s1_b1_count",
            ]
        ]
        .to_numpy()
        .T,
        "variants": df["aa_seq"].to_list(),
    }


def reformat_pdz_zenodo_dset(infile: str, outfile: str) -> None:
    """
    Converts the trans dataset files found on Zenodo into the format sent by Taraneh
    for the other libraries.
    """
    # Input data is a tab-separated file
    raw_data = pd.read_csv(infile, sep="\t")

    # We have to translate the nucleotide sequences into amino acid sequences
    assert (
        raw_data.nt_seq.str.len() % 3 == 0
    ).all(), "Nucleotide sequences are not divisible by 3"
    raw_data["aa_seq"] = raw_data.nt_seq.apply(lambda x: str(Seq(x).translate()))

    # Now rename the columns to match the other datasets. Save the data to a new
    # file
    raw_data.rename(
        columns={
            **{
                col: f"{col}_e{col.removeprefix('input')}_s0_bNA_count"
                for col in raw_data.columns
                if col.startswith("input")
            },
            **{
                col: f"{col}A_e{col.removeprefix('output')}_s0_bNA_count"
                for col in raw_data.columns
                if col.startswith("output")
            },
        }
    ).to_csv(
        outfile,
        sep="\t",
        index=False,
    )
