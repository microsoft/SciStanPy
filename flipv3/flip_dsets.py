"""Holds code relevant for the TrpB datasets."""

import os.path
import re

from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

from Bio.Seq import Seq

# Note any timepoints that are skipped in the dataset. The key is the dataset
# and the value gives (first) the indices of the non-t=0 timepoints that are skipped
# and (second) the expected number of non-t=0 timepoints after the skipped timepoints
# are removed
TRPB_SKIPPED_TIMEPOINTS = {
    "libA": (np.array([], dtype=int), 2),
    "libB": (np.array([], dtype=int), 2),
    "libC": (np.array([], dtype=int), 2),
    "libD": (np.array([0]), 4),
    "libE": (np.array([0, 1]), 3),
    "libF": (np.array([0, 1, 3]), 2),
    "libG": (np.array([0]), 4),
    "libH": (np.array([0]), 4),
    "libI": (np.array([0]), 4),
    "four-site": (np.array([], dtype=int), 6),
}

TRPB_OD600 = {
    "libA": (np.array(0.1), np.array([0.72, 2.55])),
    "libB": (np.array(0.1), np.array([0.75, 3.3])),
    "libC": (np.array(0.1), np.array([0.74, 1.95])),
    "libD": (
        np.array(0.05),
        np.array([[0.19, 0.29, 0.51, 0.85, 1.42], [0.18, 0.28, 0.49, 0.97, 1.81]]),
    ),
    "libE": (
        np.array(0.05),
        np.array(
            [
                [0.20, 0.27, 0.47, 0.91, 1.41],
                [0.20, 0.26, 0.44, 0.94, 1.54],
            ]
        ),
    ),
    "libF": (
        np.array(0.05),
        np.array(
            [
                [0.17, 0.20, 0.23, 0.27, 0.79],
                [0.17, 0.20, 0.24, 0.27, 0.79],
            ]
        ),
    ),
    "libG": (
        np.array(0.05),
        np.array(
            [
                [0.14, 0.18, 0.23, 0.44, 2.0],
                [0.14, 0.18, 0.23, 0.44, 1.95],
            ]
        ),
    ),
    "libH": (
        np.array(0.05),
        np.array(
            [
                [0.15, 0.19, 0.26, 0.67, 2.9],
                [0.14, 0.18, 0.26, 0.58, 1.85],
            ]
        ),
    ),
    "libI": (
        np.array(0.05),
        np.array(
            [
                [0.36, 0.83, 1.24, 1.7, 1.95],
                [0.39, 0.87, 1.36, 2.1, 2.25],
            ]
        ),
    ),
    "four-site": (
        np.array(0.025),
        np.array(
            [
                [[0.19, 0.51, 1.26, 1.5, 1.675, 1.75]],
                [[0.19, 0.52, 1.34, 1.625, 1.75, 1.875]],
            ]
        ),
    ),
}

# Map from amino acid to ordinal
AA_TO_ORDINAL = dict(zip("ACDEFGHIKLMNPQRSTVWXY", range(0, 21)))


def load_trpb_dataset(
    filepath: str, libname: str | None = None
) -> dict[str, npt.NDArray | list[str]]:
    """
    Load a TrpB dataset from Johnston et al.
    """

    def get_cols(coltype: str) -> list[str]:
        """Get the columns of a given type."""
        return sorted(
            (col for col in data.columns if col.startswith(coltype)),
            key=lambda x: int(x.split("_")[1]),
        )

    # If not provided, attempt to get the library name from the file name
    if libname is None:
        possible_libs = [lib for lib in TRPB_SKIPPED_TIMEPOINTS if lib in filepath]
        if len(possible_libs) == 0:
            raise ValueError(
                "No library name provided and no library name found in the file name."
            )
        elif len(possible_libs) > 1:
            raise ValueError(
                "Multiple library names found in the file name. Please provide a "
                "library name."
            )
        libname = possible_libs[0]
    elif libname not in TRPB_SKIPPED_TIMEPOINTS:
        raise ValueError(
            f"library name {libname} not found in the list of known libraries. Please "
            "provide a valid library name."
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
    # should happen for all non-four-site libraries and  means that there were no
    # replicates (biological or otherwise) of the input counts.
    if t0_counts.shape[0] == 1:
        assert libname != "four-site", (
            "The four-site library should have replicates of the input counts. "
            "Please check the data."
        )
        t0_counts = t0_counts[0]

    # Get the times.
    times = data["Time (h)"].unique().astype(float)
    times.sort()

    # Remove the timepoints that are skipped
    skipped_timepoints, n_final_timepoints = TRPB_SKIPPED_TIMEPOINTS[libname]
    assert times[0] > 0
    times = np.delete(times, skipped_timepoints)
    assert len(times) == n_final_timepoints

    # Remove ODs for timepoints that are skipped.
    od600_t0, od600_tg0 = TRPB_OD600[libname]
    od600_tg0 = np.delete(od600_tg0, skipped_timepoints, axis=-1)
    assert od600_tg0.shape[-1] == n_final_timepoints

    # Get the timepoint counts
    tg0_counts = np.zeros([len(times), len(output_cols), len(combo_order)], dtype=int)
    for timeind, time in enumerate(times):

        # Filter down to just the data for this time
        time_data = data[data["Time (h)"] == time]

        # Make sure the data is in the right order
        assert time_data.AAs.tolist() == combo_order

        # Get the counts
        tg0_counts[timeind] = time_data[output_cols].to_numpy(dtype=int).T

    # If the first dimension of the timepoint counts is '1', we can remove it. This
    # means that there were no replicates of the timepoint counts.
    if tg0_counts.shape[1] == 1:
        tg0_counts = tg0_counts[:, 0]

    return {
        "times": times,
        "starting_od": od600_t0,
        "timepoint_od": od600_tg0,
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
                col: f"{col}A_e{col.removeprefix('output')}_s1_b1_count"
                for col in raw_data.columns
                if col.startswith("output")
            },
        }
    ).to_csv(
        outfile,
        sep="\t",
        index=False,
    )


# Types for the nuclease data
class PDZDatasetType(TypedDict):
    """Type for nuclease data"""

    dataset: pd.DataFrame
    data_indices: np.ndarray[np.int64]
    fiducial_indices: dict[str, np.ndarray[np.int64]]


def load_nuclease_data(
    processed_data_dir: str,
    processed_fiducial_data_dir: str,
    gen: Literal["G1", "G2", "G3", "G4"],
):
    """
    Loads the Nuclease dataset. The data is described in
    [this](https://doi.org/10.1016/j.cels.2025.101236) paper and was downloaded
    using its associated GitHub repository.
    """

    def load_basis_data(
        filepath: str, check_unique: bool = True, check_mutcount: bool = True
    ) -> pd.DataFrame:
        """
        Load the basis data for a Nuclease dataset.
        """
        # Regular expression to match the mutation list format
        list_regex = re.compile(r"\[{1,2}(.*?)\]")

        def build_mut_signature(mutlist: str) -> str:
            """
            Converts a string-formatted list of mutations into a single hashable string
            that can be used as a signature for the mutation.
            """
            # An empty list is just the wildtype, so return an empty string (no mutations)
            if mutlist == "[]":
                return ""

            # First identify the mutations in the string
            matches = re.findall(list_regex, mutlist)

            # Clean up the additional characters in the matches and format as a list sorted
            # by the mutation position
            cleaned = [[x.strip(', "') for x in el.split()] for el in matches]
            cleaned.sort(key=lambda x: int(x[1]))

            # Join the cleaned mutations into a single string
            return "_".join("".join(el) for el in cleaned)

        # Load in the data
        df = pd.read_csv(filepath)

        # Convert the mutations column to a list of signatures
        df["mutations"] = df["mutations"].apply(build_mut_signature)

        # Run some checks
        assert (not check_unique) or df.mutations.is_unique
        assert (not check_mutcount) or (
            df.mutations.str.split("_").apply(lambda x: 0 if x == [""] else len(x))
            == df.num_mutations
        ).all()

        return df

    def combine_dset_elements(
        data_filepath: str, negative_filepath: str
    ) -> PDZDatasetType:
        return pd.concat(
            [
                load_basis_data(data_filepath),
                load_basis_data(negative_filepath, check_mutcount=False),
            ],
            ignore_index=True,
        )

    def load_g1(filepath: str, negative_filepath: str) -> dict:
        """
        Load the G1 dataset.
        """
        # Load the basis data
        df = combine_dset_elements(filepath, negative_filepath)

        # Process the dataset
        dataset = {
            "variants": df["mutations"].tolist(),
            "ic1": df[["read_count_1_input_g1", "read_count_1_input_reseq_g1"]]
            .to_numpy(dtype=int)
            .T,
            "ic2": df[["read_count_2_input_g1", "read_count_2_input_reseq_g1"]]
            .to_numpy(dtype=int)
            .T,
            "ic3": df["read_count_3_input_g1"].to_numpy(dtype=int),
            "lc": df[
                [
                    "read_count_1_low_g1",
                    "read_count_2_low_g1",
                    "read_count_3_low_g1",
                ]
            ]
            .to_numpy(dtype=int)
            .T,
            "hc1": df["read_count_1_high_g1"].to_numpy(dtype=int),
            "hc2": df[["read_count_2_high_g1", "read_count_2_high_reseq_g1"]]
            .to_numpy(dtype=int)
            .T,
            "hc3": df["read_count_3_high_g1"].to_numpy(dtype=int),
            "lt": np.array([0.1, 0.115, 0.1]),
            "ht": np.array([0.501, 0.6, 0.370]),
        }

        # Package the
        return dataset

    def load_g2(filepath: str, negative_filepath: str) -> dict:

        # Load raw dataset
        df = combine_dset_elements(filepath, negative_filepath)

        # Process the dataset
        dataset = {
            "variants": df["mutations"].tolist(),
            "ic1": df[["read_count_1_input_deep_g2", "read_count_1_input_g2"]]
            .to_numpy(dtype=int)
            .T,
            "ic2": df["read_count_2_input_g2"].to_numpy(dtype=int),
            "c86": df["read_count_1_86_g2"].to_numpy(dtype=int),
            "c93": df["read_count_2_93_g2"].to_numpy(dtype=int),
            "c975": df["read_count_1_97.5_g2"].to_numpy(dtype=int),
            "ft": np.array([0.201, 0.318, 0.385]),
        }

        return dataset

    def load_g3(filepath: str, negative_filepath: str) -> dict:
        """
        Load the G3 dataset.
        """
        # Load dataset
        df = combine_dset_elements(filepath, negative_filepath)

        # Compile the data into a dictionary
        dataset = {
            "variants": df["mutations"].tolist(),
            "ic1": df["read_count_0_input_g3"].to_numpy(dtype=int),
            "oc1": df[
                [
                    "read_count_1_59_g3",
                    "read_count_1_80_g3",
                    "read_count_2_95_g3",
                    "read_count_2_99_g3",
                ]
            ]
            .to_numpy(dtype=int)
            .T,
            "ft": np.array([0.118, 0.217, 0.285, 0.346]),
        }

        return dataset

    def load_g4(filepath: str, negative_filepath: str) -> dict:

        # Load the G4 datasets
        df = combine_dset_elements(filepath, negative_filepath)

        # Compile the data into a dictionary
        dataset = {
            "variants": df["mutations"].tolist(),
            "ic1": df["read_count_0_input_g4"].to_numpy(dtype=int),
            "oc1": df[
                [
                    "read_count_1_70_g4",
                    "read_count_2_90_g4",
                    "read_count_3_98_g4",
                    "read_count_4_99.5_g4",
                ]
            ]
            .to_numpy(dtype=int)
            .T,
            "ft": np.array([0.124, 0.199, 0.306, 0.384]),
        }

        return dataset

    # Different load function depending on generation
    if gen == "G1":
        return load_g1(
            filepath=f"{processed_data_dir}/g1.csv",
            negative_filepath=f"{processed_fiducial_data_dir}/g1_neg_control.csv",
        )
    elif gen == "G2":
        return load_g2(
            filepath=f"{processed_data_dir}/g2.csv",
            negative_filepath=f"{processed_fiducial_data_dir}/g2_neg_control.csv",
        )
    elif gen == "G3":
        return load_g3(
            filepath=f"{processed_data_dir}/g3.csv",
            negative_filepath=f"{processed_fiducial_data_dir}/g3_neg_control.csv",
        )
    elif gen == "G4":
        return load_g4(
            filepath=f"{processed_data_dir}/g4.csv",
            negative_filepath=f"{processed_fiducial_data_dir}/g4_neg_control.csv",
        )

    raise ValueError(f"Unknown generation: {gen}")


def load_k50_data(
    count_filedir: str, scramble_filepath: str, qpcr_raw_filepath: str
) -> dict:
    """Loads the data from Tsuboyama et al."""

    def load_raw() -> tuple[
        list[pd.DataFrame],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        dict[str, int],
    ]:
        """Loads the raw datasets"""

        # Read in the count dataframes, remove duplicate sequences, and assign unique
        # sequence identities
        seqids = {}
        seqid = 0
        dfs = []
        ordinals = []
        for df_ind in range(1, 5):

            # Load the dataframe, remove unnecessary columns, and combine duplicates
            df = (
                pd.read_csv(os.path.join(count_filedir, f"NGS_count_lib{df_ind}.csv"))
                .drop(columns=["name", "dna_seq"])
                .groupby("aa_seq", as_index=False)
                .sum()
            )

            # Remove dummy seqs
            df = df[df.aa_seq != "-"]

            # Assign unique sequence IDs and ordinally encode with padding
            ids = [None] * len(df)
            for df_ind, seq in enumerate(df.aa_seq.tolist()):

                # Only process if we haven't seen before
                if seq not in seqids:

                    # Set ID
                    seqids[seq] = seqid
                    seqid += 1

                    # Determine how much padding we need
                    total_pad = 80 - len(seq)
                    front_pad = total_pad // 2
                    back_pad = total_pad - front_pad

                    # Apply padding and ordinally encode
                    ordinals.append(
                        [
                            AA_TO_ORDINAL[aa]
                            for aa in (
                                "X" * front_pad + "GGG" + seq + "GGG" + "X" * back_pad
                            )
                        ]
                    )

                # Note the sequence ID
                ids[df_ind] = seqids[seq]

            # Record sequence IDs as part of the dataframe
            df["seq_id"] = ids

            # Record
            dfs.append(df)

        # Get the scrambled sequence indices
        scrambled_inds = np.unique(
            pd.read_csv(scramble_filepath).aa_seq_experimental.map(seqids).to_numpy()
        )
        assert not np.isnan(scrambled_inds).any()

        return dfs, np.stack(ordinals), scrambled_inds, seqids

    def gather_counts() -> dict[str, npt.NDArray]:
        """Gathers count data for each library in the dataset"""
        # Convert the dataframes to numpy arrays of counts
        libdata = {}
        for lib_ind, df in enumerate(dfs, 1):

            # Get the columns for each replicate
            stringdices1 = [str(j).rjust(2, "0") for j in range(1, 13)]
            stringdices2 = [str(j) for j in range(13, 25)]

            # Get counts for each protease
            protease_counts = []
            for protease in ("C", "T"):
                protease_counts.append(
                    np.stack(
                        [
                            df[
                                [
                                    f"v{lib_ind}_{protease}{stringdex}"
                                    for stringdex in stringdices
                                ]
                            ]
                            .to_numpy(dtype=int)
                            .T
                            for stringdices in (stringdices1, stringdices2)
                        ]
                    )
                )

            # Record for the libraries
            key = f"v{lib_ind}"
            protease_counts = np.stack(protease_counts)
            libdata[f"{key}_counts_cg0"] = protease_counts[:, :, 1:]
            libdata[f"{key}_counts_c0"] = protease_counts[:, :, 0]
            libdata[f"{key}_seqids"] = df["seq_id"].to_numpy()

        return libdata

    def load_qpcr() -> dict[str, npt.NDArray]:
        """Loads raw qpcr data"""

        # Load qpcr data and separate trypsin from chymotrypsin
        qpcr = pd.read_csv(qpcr_raw_filepath)
        tryp = qpcr[qpcr.protease == "trypsin"]
        chymo = qpcr[qpcr.protease == "chymotrypsin"]

        # Get protease concentrations
        protease_conc = chymo.protease_con.to_numpy()
        assert np.all(protease_conc == tryp.protease_con.to_numpy())

        # Get survival data for each protein
        tryp_survival = tryp.iloc[1:, 1:9].to_numpy()
        chymo_survival = chymo.iloc[1:, 1:9].to_numpy()

        return {
            "qpcr_log_protease_conc": np.log(protease_conc[1:]),
            "qpcr_log2_survival": np.stack([chymo_survival, tryp_survival]),
        }

    # Get count data and ordinally encoded sequences
    dfs, ordinals, scrambled_inds, seqids = load_raw()
    libdata = gather_counts()
    libdata.update(
        {"ordinal_seqs": ordinals, "scrambled_inds": scrambled_inds, "seq_ids": seqids}
    )

    # Define the expected protease concentrations
    conc1 = np.log(25) - np.arange(10, -1, -1) * np.log(3)
    conc2 = 0.5 * np.log(3) + conc1
    libdata["log_expected_protease_conc"] = np.stack([conc1, conc2])

    # Add qpcr data
    libdata.update(load_qpcr())

    return libdata
