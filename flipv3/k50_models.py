"""Holds model for the dataset presented by Tsuboyama et al."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from scistanpy import Constant, Model, operations, parameters
from scistanpy.model.components.transformations.transformed_parameters import (
    TransformedParameter,
)

from flipv3.flip_dsets import AA_TO_ORDINAL

if TYPE_CHECKING:
    from scistanpy import custom_types

# pylint: disable=invalid-name

# Define indices of amino acids that inhibit and activate the different proteases
INHIBITOR_INDS = np.array([AA_TO_ORDINAL[aa] for aa in "DEP"])
C_ACTIVATOR_INDS = np.array([AA_TO_ORDINAL[aa] for aa in "FWY"])
T_ACTIVATOR_INDS = np.array([AA_TO_ORDINAL[aa] for aa in "KR"])


class K50Model(Model):
    """Model of the K50 dataset"""

    def __init__(
        self,
        *,
        qpcr_log_protease_conc: npt.NDArray[np.float64],
        qpcr_log2_survival: npt.NDArray[np.float64],
        log_expected_protease_conc: npt.NDArray[np.float64],
        v1_seqids: npt.NDArray[np.int64],
        v1_counts_c0: npt.NDArray[np.int64],
        v1_counts_cg0: npt.NDArray[np.int64],
        v2_seqids: npt.NDArray[np.int64],
        v2_counts_c0: npt.NDArray[np.int64],
        v2_counts_cg0: npt.NDArray[np.int64],
        v3_seqids: npt.NDArray[np.int64],
        v3_counts_c0: npt.NDArray[np.int64],
        v3_counts_cg0: npt.NDArray[np.int64],
        v4_seqids: npt.NDArray[np.int64],
        v4_counts_c0: npt.NDArray[np.int64],
        v4_counts_cg0: npt.NDArray[np.int64],
        ordinal_seqs: npt.NDArray[np.int64],
        scrambled_inds: npt.NDArray[np.int64],
        alpha_alpha: "custom_types.Float" = 2.0,
        alpha_beta: "custom_types.Float" = 2.0,
        log_kmax_t_mu: "custom_types.Float" = 0.0,
        log_kmax_t_sigma: "custom_types.Float" = 1.0,
        log_K50_mu_qpcr: "custom_types.Float" = -7.5,
        log_K50_sigma_qpcr: "custom_types.Float" = 1.5,
        protease_dg_sigma_sigma: "custom_types.Float" = 0.1,
        protease_conc_sigma_sigma: "custom_types.Float" = 0.1,
        expected_dg_unfolding: "custom_types.Float" = 5.0,  # Assume stable for non-scrambled
        expected_dg_unfolding_scrambled: "custom_types.Float" = -5.0,  # Assume unstable for scrambled
        dg_unfolding_sigma: "custom_types.Float" = 5.0,
        pssm_mus: npt.NDArray[np.float64] | None = None,
        pssm_sigmas: npt.NDArray[np.float64] | None = None,
        log_maxK50u_mu: "custom_types.Float" = 2.0,
        log_maxK50u_sigma: "custom_types.Float" = 1.0,
        log_minK50u_mu: "custom_types.Float" = -4.0,
        log_minK50u_sigma: "custom_types.Float" = 1.0,
        K50u_thresh_mus: npt.NDArray[np.float64] | None = None,
        K50u_thresh_sigma: "custom_types.Float" = 2.0,
        log_K50f_mu: npt.NDArray[np.float64] | None = None,
        log_K50f_sigma: "custom_types.Float" = 0.5,
    ):
        """Initializes all parameters and defines model graph"""
        # Set default values if they are not set
        pssm_mus = self._set_pssm_mus(pssm_mus)
        pssm_sigmas = self._set_pssm_sigmas(pssm_sigmas)
        if K50u_thresh_mus is None:
            K50u_thresh_mus = np.linspace(0, 20, 10)
        if log_K50f_mu is None:
            log_K50f_mu = np.array([1.75, 2.25])

        # Check shapes
        assert qpcr_log_protease_conc.shape == (11,)
        assert qpcr_log2_survival.shape == (2, 11, 8)
        assert log_expected_protease_conc.shape == (2, 11)
        assert v1_seqids.ndim == v2_seqids.ndim == v3_seqids.ndim == v4_seqids.ndim == 1
        assert (
            v1_counts_c0.ndim
            == v2_counts_c0.ndim
            == v3_counts_c0.ndim
            == v4_counts_c0.ndim
            == 3
        )
        assert (
            v1_counts_c0.shape[:2]
            == v2_counts_c0.shape[:2]
            == v3_counts_c0.shape[:2]
            == v4_counts_c0.shape[:2]
        )
        assert (
            v1_counts_cg0.ndim
            == v2_counts_cg0.ndim
            == v3_counts_cg0.ndim
            == v4_counts_cg0.ndim
            == 4
        )
        assert (
            v1_counts_cg0.shape[:3]
            == v2_counts_cg0.shape[:3]
            == v3_counts_cg0.shape[:3]
            == v4_counts_cg0.shape[:3]
        )
        assert v1_counts_c0.shape[-1] == v1_counts_cg0.shape[-1]
        assert v2_counts_c0.shape[-1] == v2_counts_cg0.shape[-1]
        assert v3_counts_c0.shape[-1] == v3_counts_cg0.shape[-1]
        assert v4_counts_c0.shape[-1] == v4_counts_cg0.shape[-1]
        assert ordinal_seqs.ndim == 2
        assert ordinal_seqs.shape[-1] == 86
        assert scrambled_inds.shape == (64238,)
        assert pssm_mus.shape == (2, 9, 21)
        assert pssm_sigmas.shape == (2, 9, 21)
        assert K50u_thresh_mus.shape == (10,)
        assert log_K50f_mu.shape == (2,)

        # Get the number of unique sequence IDs.
        self.n_seqids = (
            max(v1_seqids.max(), v2_seqids.max(), v3_seqids.max(), v4_seqids.max()) + 1
        )
        assert len(ordinal_seqs) == self.n_seqids

        # Record constants
        self.qpcr_log_protease_conc = Constant(
            qpcr_log_protease_conc[None, :, None], togglable=False
        )
        self.ordinal_seqs = Constant(ordinal_seqs, togglable=False)

        # Record default data
        super().__init__(
            default_data={
                "qpcr_log2_survival": qpcr_log2_survival,
                "v1_counts_c0": v1_counts_c0[..., None, :],  # Add conc dim
                "v1_counts_cg0": v1_counts_cg0,
                "v2_counts_c0": v2_counts_c0[..., None, :],
                "v2_counts_cg0": v2_counts_cg0,
                "v3_counts_c0": v3_counts_c0[..., None, :],
                "v3_counts_cg0": v3_counts_cg0,
                "v4_counts_c0": v4_counts_c0[..., None, :],
                "v4_counts_cg0": v4_counts_cg0,
            }
        )

        # Define the protease concentrations. We assume some degree of noise across
        # the two replicates, two different proteases, and four libraries.
        self.protease_conc_noise = parameters.HalfNormal(
            sigma=protease_conc_sigma_sigma
        )
        self.log_protease_conc = parameters.Normal(
            mu=Constant(log_expected_protease_conc[None, :, None, :], togglable=False),
            sigma=self.protease_conc_noise,
            shape=(
                4,
                2,
                2,
                11,
            ),  # 4 libraries x 2 replicates x 2 proteases x 11 concentrations
        )

        # Each sequence has a mean dG that combines information from both proteases.
        # We give different priors to scrambled vs non-scrambled sequences. Scrambled
        # sequences should have a lower dG unfolding, indicating they are less stable.
        mean_dg_unfolding = np.full((self.n_seqids,), expected_dg_unfolding)
        mean_dg_unfolding[scrambled_inds] = expected_dg_unfolding_scrambled
        self.mean_dg = parameters.Normal(
            mu=mean_dg_unfolding, sigma=dg_unfolding_sigma, shape=(self.n_seqids,)
        )

        # There are different dGs for each protein treated by each protease.
        self.protease_dg_noise = parameters.HalfNormal(sigma=protease_dg_sigma_sigma)
        self.protease_dg = parameters.Normal(
            mu=self.mean_dg, sigma=self.protease_dg_noise, shape=(2, self.n_seqids)
        )

        # We expect a universal kmaxt for all proteins for a given protease. We
        # have 2 proteases.
        self.log_kmax_t = parameters.Normal(
            mu=log_kmax_t_mu, sigma=log_kmax_t_sigma, shape=(2, 1, 1)
        )

        # Get K50u for all sequences. shape = (2, self.n_seqids)
        self.log_K50u = self._def_K50u(
            pssm_mus=pssm_mus,
            pssm_sigmas=pssm_sigmas,
            log_maxK50u_mu=log_maxK50u_mu,
            log_maxK50u_sigma=log_maxK50u_sigma,
            log_minK50u_mu=log_minK50u_mu,
            log_minK50u_sigma=log_minK50u_sigma,
            K50u_thresh_mus=K50u_thresh_mus,
            K50u_thresh_sigma=K50u_thresh_sigma,
        )  # n proteases x n_seqs

        # Get K50f. This is assumed to be a constant function of the two proteases
        self.log_K50f = parameters.Normal(
            mu=log_K50f_mu, sigma=log_K50f_sigma, shape=(2, 1)
        )

        # Get K50. This is calculated using the values for dG, K50u, and K50f
        self.log_K50 = self._def_K50()  # n protease x n seq

        # Using K50, kmaxt, and the protease concentrations, we can calculate survival
        self._model_counts(
            alpha_alpha=alpha_alpha,
            alpha_beta=alpha_beta,
            v1_seqids=v1_seqids,
            v2_seqids=v2_seqids,
            v3_seqids=v3_seqids,
            v4_seqids=v4_seqids,
        )

        # We also model the qpcr workflow, which helps us assign the value for log_kmax_t
        self._model_qpcr(
            log_K50_mu_qpcr=log_K50_mu_qpcr, log_K50_sigma_qpcr=log_K50_sigma_qpcr
        )

    def _set_pssm_mus(
        self, pssm_mus: npt.NDArray[np.float64] | None
    ) -> npt.NDArray[np.float64]:
        """Sets default values for the position-specific scoring matrix means."""
        # If provided, make sure it is the right shape, then return
        if pssm_mus is not None:
            assert pssm_mus.shape == (2, 9, 21)
            return pssm_mus

        # Otherwise, set
        pssm_mus = np.ones((2, 9, 21)) * -0.3

        # Inhibitory amino acids have lower influence
        pssm_mus[..., INHIBITOR_INDS] = -2.3  # 1 stdev from -0.3
        pssm_mus[:, 4, INHIBITOR_INDS] = -4.3  # 2 stdev from -0.3

        # Known cleavable amino acids have higher influence
        pssm_mus[0, :, C_ACTIVATOR_INDS] = 3.7  # 1 stdev from -0.3
        pssm_mus[0, 4, C_ACTIVATOR_INDS] = 7.7  # 2 stdev from -0.3
        pssm_mus[1, :, T_ACTIVATOR_INDS] = 3.7  # 1 stdev from -0.3
        pssm_mus[1, 4, T_ACTIVATOR_INDS] = 7.7  # 2 stdev from -0.3

        return pssm_mus

    def _set_pssm_sigmas(
        self, pssm_sigmas: npt.NDArray[np.float64] | None
    ) -> npt.NDArray[np.float64]:
        # If provided, make sure it is the right shape, then return
        if pssm_sigmas is not None:
            assert pssm_sigmas.shape == (2, 9, 21)
            return pssm_sigmas

        # Otherwise, set
        pssm_sigmas = np.ones((2, 9, 21)) * 0.1

        # Wider range for center, inhibitors, and cleavables alike
        pssm_sigmas[:, 4] = 2.0
        pssm_sigmas[..., INHIBITOR_INDS] = 2.0
        pssm_sigmas[0, :, C_ACTIVATOR_INDS] = 4.0
        pssm_sigmas[1, :, T_ACTIVATOR_INDS] = 4.0

        return pssm_sigmas

    def _def_K50u(
        self,
        pssm_mus: npt.NDArray[np.float64],
        pssm_sigmas: npt.NDArray[np.float64],
        log_maxK50u_mu: "custom_types.Float",
        log_maxK50u_sigma: "custom_types.Float",
        log_minK50u_mu: "custom_types.Float",
        log_minK50u_sigma: "custom_types.Float",
        K50u_thresh_mus: npt.NDArray[np.float64],
        K50u_thresh_sigma: "custom_types.Float",
    ) -> TransformedParameter:

        # Define the min and max K50u. We define one per protease.
        self.log_maxK50u = parameters.Normal(
            mu=log_maxK50u_mu, sigma=log_maxK50u_sigma, shape=(2, 1)  # one per protease
        )
        self.log_minK50u = parameters.Normal(
            mu=log_minK50u_mu, sigma=log_minK50u_sigma, shape=(2, 1)  # one per protease
        )

        # Initialize the position-specific scoring matrix
        self.pssm = parameters.Normal(
            mu=pssm_mus,
            sigma=pssm_sigmas,
            shape=(2, 9, 21),  # 2 proteases by 9 positions by 21 amino acids
        )

        # Calculate the sum of ssks
        self.sum_ssk = operations.sum_(
            operations.sigmoid(
                operations.convolve_sequence(
                    weights=self.pssm, ordinals=self.ordinal_seqs
                )
            ),  # n_seqs x padded length
            keepdims=True,
            shape=(2, self.n_seqids, 1),  # n proteases x n_seqs
        )

        # Now the activation thresholds. We use 10 of them and make sure that they
        # have different priors. If we do not give them different priors, we will
        # have identifiability issues
        self.thresholds = parameters.Normal(
            mu=K50u_thresh_mus, sigma=K50u_thresh_sigma, shape=(2, 1, 10)
        )

        # Apply thresholds and logistic functions, then sum over thresholds
        self.activation = operations.sum_(
            operations.sigmoid(self.sum_ssk - self.thresholds),
            shape=(2, self.n_seqids),  # n proteases x n_seqs
        )

        # Calculate K50u for each protease and protein
        return (
            self.log_maxK50u - (self.log_maxK50u - self.log_minK50u) * self.activation
        )

    def _def_K50(self):
        """Defines the K50 value as a transformed parameter"""
        # Calculate -dG/RT
        neg_dGRT = -self.protease_dg / (298 * 0.001987)  # n protease x n seq

        # Now calculate K50
        return (
            operations.log1p_exp(neg_dGRT)
            - self.log_K50f  # n protease x 1
            - operations.log1p_exp(self.log_K50f + neg_dGRT - self.log_K50u)
        )

    def _model_counts(
        self,
        *,
        alpha_alpha: "custom_types.Float",
        alpha_beta: "custom_types.Float",
        v1_seqids: npt.NDArray[np.int64],
        v2_seqids: npt.NDArray[np.int64],
        v3_seqids: npt.NDArray[np.int64],
        v4_seqids: npt.NDArray[np.int64],
    ) -> None:

        # Now we model the counts data
        for lib_ind, (lib, ind_array) in enumerate(
            (
                ("v1", v1_seqids),
                ("v2", v2_seqids),
                ("v3", v3_seqids),
                ("v4", v4_seqids),
            )
        ):

            # Names for the attributes to be assigned
            c0_alphas = f"{lib}_alphas"
            log_theta_c0 = f"{lib}_log_theta_c0"
            log_survival = f"{lib}_log_survival"
            log_theta_cg0 = f"{lib}_log_theta_cg0"
            log_theta_cg0_unnorm = f"{log_theta_cg0}_unnorm"
            counts_c0 = f"{lib}_counts_c0"
            counts_cg0 = f"{lib}_counts_cg0"
            total_counts_c0 = f"total_{counts_c0}"
            total_counts_cg0 = f"total_{counts_cg0}"

            # Define the prior on alphas. We assume that replicates start from similar
            # abundances
            seqs_in_lib = len(ind_array)
            setattr(
                self,
                c0_alphas,
                parameters.Gamma(
                    alpha=alpha_alpha, beta=alpha_beta, shape=(seqs_in_lib,)
                ),
            )

            # Model the starting proportions
            setattr(
                self,
                log_theta_c0,
                parameters.ExpDirichlet(
                    alpha=getattr(self, c0_alphas), shape=(2, 2, 1, seqs_in_lib)
                ),
            )  # nrep x nprot x 1 x nseq

            # Calculate survival
            setattr(
                self,
                log_survival,
                self.calculate_log_survival(
                    log_kmax_t=self.log_kmax_t,  # nprot x 1 x 1
                    log_K50=self.log_K50[:, None, ind_array],  # nprot x 1 x nseq
                    log_conc=self.log_protease_conc[
                        lib_ind, ..., None
                    ],  # nrep x nprot x nconc x 1
                ),
            )  # nrep x nprot x nconc x nseq

            # Using the log survival calculation, determine the proportions of sequences
            # we expect to remain after protease treatment
            setattr(
                self,
                log_theta_cg0_unnorm,
                (
                    getattr(self, log_survival)  # nrep x nprot x nconc x nseq
                    + getattr(self, log_theta_c0)  # nrep x nprot x 1 x nseq
                ),
            )
            setattr(
                self,
                log_theta_cg0,  # nrep x nprot x nconc x nseq
                operations.normalize_log(getattr(self, log_theta_cg0_unnorm)),
            )

            # Set the total counts as constants
            setattr(
                self,
                total_counts_c0,
                Constant(
                    self.default_data[counts_c0].sum(axis=-1, keepdim=True),
                    togglable=False,
                ),
            )  # nrep x nprot x 1 x 1
            setattr(
                self,
                total_counts_cg0,
                Constant(
                    self.default_data[counts_cg0].sum(axis=-1, keepdim=True),
                    togglable=False,
                ),
            )  # nrep x nprot x conc x 1

            # Model counts as a multinomial distribution
            setattr(
                self,
                counts_c0,
                parameters.MultinomialLogTheta(
                    log_theta=getattr(self, log_theta_c0),
                    N=getattr(self, total_counts_c0),
                    shape=(2, 2, 1, seqs_in_lib),
                ),
            )
            setattr(
                self,
                counts_cg0,
                parameters.MultinomialLogTheta(
                    log_theta=getattr(self, log_theta_cg0),
                    N=getattr(self, total_counts_cg0),
                    shape=(2, 2, 11, seqs_in_lib),
                ),
            )

    def _model_qpcr(self, *, log_K50_mu_qpcr: float, log_K50_sigma_qpcr: float) -> None:
        """Defines the piece of the generative model in charge of the qPCR data"""

        # Each protein will have its own K50 for each protease depending on how
        # well it's folded. 2 proteases x concentration dim x 8 qpcr proteins
        self.log_K50_qpcr = parameters.Normal(
            mu=log_K50_mu_qpcr, sigma=log_K50_sigma_qpcr, shape=(2, 1, 8)
        )

        # Calculate the log survival. The qPCR values are reported in log-2 space
        # while we are performing our calculations in natural log space, so we need
        # to convert between the two.
        self.qpcr_log2_survival = self.calculate_log_survival(
            log_kmax_t=self.log_kmax_t,  # nprot x 1 x 1
            log_K50=self.log_K50_qpcr,  # nprot x 1 x nseq
            log_conc=self.qpcr_log_protease_conc,  # 1 x nconc x 1
        )  # nprot x nconc x nseq

    @staticmethod
    def calculate_log_survival(
        log_kmax_t: parameters.Parameter,
        log_K50: parameters.Parameter,
        log_conc: Constant,
    ) -> TransformedParameter:
        """
        Calculates the log survival based on Eq. 2 and 3 from the source paper
        """
        # Log survival is the negative exponentiation of the log rate
        return -operations.exp(
            log_kmax_t + log_conc - log_K50 - operations.log1p_exp(log_conc - log_K50)
        )
