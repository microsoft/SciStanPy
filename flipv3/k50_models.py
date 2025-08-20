"""Holds model for the dataset presented by Tsuboyama et al."""

import numpy as np
import numpy.typing as npt

from scistanpy import Constant, Model, operations, parameters
from scistanpy.model.components.transformations.transformed_parameters import (
    TransformedParameter,
)

# pylint: disable=invalid-name


class K50Model(Model):
    """Model of the K50 dataset"""

    def __init__(
        self,
        *,
        qpcr_log_protease_conc: npt.NDArray[np.float64],
        qpcr_log2_survival: npt.NDArray[np.float64],
        log_expected_protease_conc: npt.NDArray[np.float64],
        v1_seq_ids: npt.NDArray[np.int64],
        v1_counts_c0: npt.NDArray[np.int64],
        v1_counts_cg0: npt.NDArray[np.int64],
        v2_seq_ids: npt.NDArray[np.int64],
        v2_counts_c0: npt.NDArray[np.int64],
        v2_counts_cg0: npt.NDArray[np.int64],
        v3_seq_ids: npt.NDArray[np.int64],
        v3_counts_c0: npt.NDArray[np.int64],
        v3_counts_cg0: npt.NDArray[np.int64],
        v4_seq_ids: npt.NDArray[np.int64],
        v4_counts_c0: npt.NDArray[np.int64],
        v4_counts_cg0: npt.NDArray[np.int64],
        ordinal_seqs: npt.NDArray[np.int64],
        alpha_alpha: float = 2.0,
        alpha_beta: float = 2.0,
        log_kmax_t_mu: float = 0.0,
        log_kmax_t_sigma: float = 1.0,
        log_K50_mu_qpcr: float = -7.5,
        log_K50_sigma_qpcr: float = 1.5,
        protease_dg_sigma_sigma: float = 0.1,
        protease_conc_sigma_sigma: float = 0.1,
        expected_dg_unfolding: float = -1.0,
        dg_unfolding_sigma: float = 3.0,
        pssm_mus: npt.NDArray[np.float64] | None = None,
        pssm_sigmas: npt.NDArray[np.float64] | None = None,
        log_maxK50u_mu: float = 2.0,
        log_maxK50u_sigma: float = 1.0,
        log_minK50u_mu: float = -4.0,
        log_minK50u_sigma: float = 1.0,
        K50u_thresh_mus: npt.NDArray[np.float64] | None = None,
        K50u_thresh_sigma: float = 2.0,
        log_K50f_mu: npt.NDArray[np.float64] | None = None,
        log_K50f_sigma: float = 0.5,
    ):
        """Initializes all parameters and defines model graph"""
        # Set default values if they are not set
        if pssm_mus is None:
            pssm_mus = np.ones((2, 9, 21)) * -0.3
        if pssm_sigmas is None:
            # TODO: Note that these vary by amino acid
            raise NotImplementedError
        if K50u_thresh_mus is None:
            K50u_thresh_mus = np.linspace(0, 20, 10)
        if log_K50f_mu is None:
            log_K50f_mu = np.array([1.75, 2.25])

        # Check shapes
        assert qpcr_log_protease_conc.shape == (12,)
        assert qpcr_log2_survival.shape == (2, 12, 8)
        assert log_expected_protease_conc.shape == (2, 11)
        assert (
            v1_seq_ids.ndim
            == v2_seq_ids.ndim
            == v3_seq_ids.ndim
            == v4_seq_ids.ndim
            == 1
        )
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
        assert ordinal_seqs.shape[-1] == 21
        assert pssm_mus.shape == (2, 9, 21)
        assert pssm_sigmas.shape == (2, 9, 21)
        assert K50u_thresh_mus.shape == (10,)
        assert log_K50f_mu.shape == (2,)

        # Get the number of unique sequence IDs.
        self.n_seq_ids = (
            max(v1_seq_ids.max(), v2_seq_ids.max(), v3_seq_ids.max(), v4_seq_ids.max())
            + 1
        )
        assert len(ordinal_seqs) == self.n_seq_ids

        # Record constants
        self.qpcr_log_protease_conc = Constant(
            qpcr_log_protease_conc[None, :, None], togglable=False
        )
        self.log_expected_protease_conc = Constant(
            log_expected_protease_conc, togglable=False
        )
        self.ordinal_seqs = Constant(ordinal_seqs, togglable=False)

        # Record default data
        super().__init__(
            default_data={
                "qpcr_log2_survival": qpcr_log2_survival,
                "v1_counts_c0": v1_counts_c0,
                "v1_counts_cg0": v1_counts_cg0,
                "v2_counts_c0": v2_counts_c0,
                "v2_counts_cg0": v2_counts_cg0,
                "v3_counts_c0": v3_counts_c0,
                "v3_counts_cg0": v3_counts_cg0,
                "v4_counts_c0": v4_counts_c0,
                "v4_counts_cg0": v4_counts_cg0,
            }
        )

        # Define the distributions that model the input data at concentration 0
        self.c0_alphas = parameters.Gamma(
            alpha=alpha_alpha, beta=alpha_beta, shape=(self.n_seq_ids)
        )
        self.log_theta_c0 = parameters.ExpDirichlet(
            alpha=self.c0_alphas, shape=(2, self.n_seq_ids)
        )

        # Define the protease concentrations. We assume some degree of noise across
        # the two replicates, two different proteases, and four libraries.
        self.protease_conc_noise = parameters.HalfNormal(
            sigma=protease_conc_sigma_sigma
        )
        self.log_protease_conc = parameters.Normal(
            mean=log_expected_protease_conc[:, None, None, :],
            sigma=self.protease_conc_noise,
            shape=(
                2,
                4,
                2,
                11,
            ),  # 2 replicates x 4 libraries x 2 proteases x 11 concentrations
        )

        # Each sequence has a mean dG that combines information from both proteases.
        # We're modeling dG of *unfolding*, and we expect proteins to be less stable
        # than otherwise, so we use a Normal distribution with a negative mean
        # TODO: We need a stronger prior on this. Scrambled sequences should have
        # higher dG than folded sequences.
        self.mean_dg = parameters.Normal(
            mu=expected_dg_unfolding, sigma=dg_unfolding_sigma, shape=(self.n_seq_ids,)
        )

        # There are different dGs for each protein treated by each protease.
        self.protease_dg_noise = parameters.HalfNormal(sigma=protease_dg_sigma_sigma)
        self.protease_dg = parameters.Normal(
            mu=self.mean_dg, sigma=self.protease_dg_noise, shape=(2, self.n_seq_ids)
        )

        # We expect a universal kmaxt for all proteins for a given protease. We
        # have 2 proteases.
        self.log_kmax_t = parameters.Normal(
            mu=log_kmax_t_mu, sigma=log_kmax_t_sigma, shape=(2, 1, 1)
        )

        # Get K50u for all sequences. shape = (2, self.n_seq_ids)
        self.log_K50u = self._def_K50u(
            pssm_mus=pssm_mus,
            pssm_sigmas=pssm_sigmas,
            log_maxK50u_mu=log_maxK50u_mu,
            log_maxK50u_sigma=log_maxK50u_sigma,
            log_minK50u_mu=log_minK50u_mu,
            log_minK50u_sigma=log_minK50u_sigma,
            K50u_thresh_mus=K50u_thresh_mus,
            K50u_thresh_sigma=K50u_thresh_sigma,
        )

        # Get K50f. This is assumed to be a constant function of the two proteases
        self.log_K50f = parameters.Normal(
            mu=log_K50f_mu, sigma=log_K50f_sigma, shape=(2, 1)
        )

        # Get K50. This is calculated using the values for dG, K50u, and K50f
        self.log_K50 = self._def_K50()

        # Using K50, kmaxt, and the protease concentrations, we can calculate survival
        self.log_survival = self.calculate_log_survival(
            log_kmax_t=self.log_kmax_t,
            log_K50=self.log_K50,
            log_conc=self.log_protease_conc,
        )

        # Using the log survival calculation, determine the proportions of sequences
        # we expect to remain after protease treatment
        self.log_theta_cg0_unnorm = self.log_survival + self.log_theta_c0
        self.log_theta_cg0 = operations.logsumexp(self.log_theta_cg0_unnorm)

        # Now onto modeling the observables. First, we model the qpcr workflow,
        # which helps us assign the value for log_kmax_t.
        self._model_qpcr(
            log_K50_mu_qpcr=log_K50_mu_qpcr, log_K50_sigma_qpcr=log_K50_sigma_qpcr
        )

        # Now we model the counts data
        self._model_counts(
            v1_seq_ids=v1_seq_ids,
            v2_seq_ids=v2_seq_ids,
            v3_seq_ids=v3_seq_ids,
            v4_seq_ids=v4_seq_ids,
        )

    def _def_K50u(
        self,
        pssm_mus: npt.NDArray[np.float64],
        pssm_sigmas: npt.NDArray[np.float64],
        log_maxK50u_mu: float,
        log_maxK50u_sigma: float,
        log_minK50u_mu: float,
        log_minK50u_sigma: float,
        K50u_thresh_mus: npt.NDArray[np.float64],
        K50u_thresh_sigma: float,
    ) -> TransformedParameter:

        # Define the min and max K50u. We define one per protease.
        self.log_maxK50u = parameters.Normal(
            mu=log_maxK50u_mu, sigma=log_maxK50u_sigma, shape=(2, 1)
        )
        self.log_minK50u = parameters.Normal(
            mu=log_minK50u_mu, sigma=log_minK50u_sigma, shape=(2, 1)
        )

        # Initialize the position-specific scoring matrix
        self.pssm = parameters.Normal(
            mu=pssm_mus,
            sigma=pssm_sigmas,
            shape=(2, 9, 21),  # 2 proteases by 9 positions by 21 amino acids
        )

        # Calculate the sum of ssks
        self.sum_ssk = operations.sum(
            operations.sigmoid(
                operations.convseq(weights=self.pssm),
                ordinals=self.ordinal_seqs,
            ),
            keepdims=True,
            shape=(2, self.n_seq_ids, 1),
        )

        # Now the activation thresholds. We use 10 of them and make sure that they
        # have different priors. If we do not give them different priors, we will
        # have identifiability issues
        self.thresholds = parameters.Normal(
            mu=K50u_thresh_mus, sigma=K50u_thresh_sigma, shape=(2, 1, 10)
        )

        # Apply thresholds and logistic functions, then sum over thresholds
        self.activations = operations.sum(
            operations.sigmoid(self.sum_ssk - self.thresholds),
            shape=(2, self.n_seq_ids),
        )

        # Calculate K50u
        return (
            self.log_maxK50u - (self.log_maxK50u - self.log_minK50u) * self.activations
        )

    def _def_K50(self):
        """Defines the K50 value as a transformed parameter"""
        # Calculate -dG/RT
        neg_dGRT = -self.protease_dg / (298 * 0.001987)

        # Now calculate K50
        return (
            operations.log1p_exp(neg_dGRT)
            - self.log_K50f
            - operations.log1p_exp(self.log_K50f + neg_dGRT - self.log_K50u)
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
            log_kmax_t=self.log_kmax_t,
            log_K50=self.log_K50_qpcr,
            log_conc=self.qpcr_log_protease_conc,
        )

    def _model_counts(
        self,
        *,
        v1_seq_ids: npt.NDArray[np.int64],
        v2_seq_ids: npt.NDArray[np.int64],
        v3_seq_ids: npt.NDArray[np.int64],
        v4_seq_ids: npt.NDArray[np.int64],
    ) -> None:

        # Now we model the counts data
        for lib, ind_array in (
            ("v1", v1_seq_ids),
            ("v2", v2_seq_ids),
            ("v3", v3_seq_ids),
            ("v4", v4_seq_ids),
        ):

            # Names for the attributes
            counts_c0 = f"{lib}_counts_c0"
            counts_cg0 = f"{lib}_counts_cg0"
            total_counts_c0 = f"total_{counts_c0}"
            total_counts_cg0 = f"total_{counts_cg0}"

            # Set the total counts as constants
            setattr(
                self,
                total_counts_c0,
                Constant(self.default_data[counts_c0].sum(axis=-1), togglable=False),
            )
            setattr(
                self,
                total_counts_cg0,
                Constant(self.default_data[counts_cg0].sum(axis=-1), togglable=False),
            )

            # Model counts as a multinomial distribution
            setattr(
                self,
                counts_c0,
                parameters.MultinomialLogTheta(
                    theta=self.log_theta_c0[..., ind_array],
                    N=getattr(self, total_counts_c0),
                ),
            )
            setattr(
                self,
                counts_cg0,
                parameters.MultinomialLogTheta(
                    theta=self.log_theta_cg0[..., ind_array],
                    N=getattr(self, total_counts_cg0),
                ),
            )

    @staticmethod
    def calculate_log_survival(
        log_kmax_t: parameters.Parameter,
        log_K50: parameters.Parameter,
        log_conc: Constant,
    ) -> TransformedParameter:
        """
        Calculates the log survival based on Eq. 2 and 3 from the source paper
        """
        # Calculate rate in log space
        log_rate = log_kmax_t + log_conc - operations.logsumexp(log_K50, log_conc)

        # Log survival is the negative exponentiation of the log rate
        return -operations.exp(log_rate)
