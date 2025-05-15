"""Holds code relevant for the TrpB datasets."""

import numpy as np
import numpy.typing as npt
import pandas as pd


def load_trpb_dataset(filepath: str) -> dict[str, npt.NDArray | list[str]]:
    """
    Load a TrpB dataset from Johnston et al.
    """
    # Load in the data
    data = pd.read_csv(filepath)

    # Get the output columns
    output_cols = sorted(
        (col for col in data.columns if col.startswith("OutputCount")),
        key=lambda x: int(x.split("_")[1]),
    )

    # Get unique combo to input counts
    t0_data = data[["AAs", "InputCount_1"]].drop_duplicates()
    assert (t0_data.AAs.value_counts() == 1).all()
    combo_order = t0_data.AAs.tolist()
    t0_counts = t0_data.InputCount_1.to_numpy(dtype=int)

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
        "times": np.concatenate([[0], times]),
        "starting_counts": t0_counts,
        "timepoint_counts": tg0_counts,
        "variants": combo_order,
    }


def load_pdz3_dataset(filepath: str) -> dict[str, npt.NDArray[np.int64]]:
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
        "ending_counts": df[
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
