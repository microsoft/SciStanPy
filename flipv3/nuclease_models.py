"""Holds models for the nuclease dataset"""

import numpy as np
import numpy.typing as npt

from scistanpy import Constant, Model, operations, parameters
from .constants import DEFAULT_HYPERPARAMS


# These have biological replicates, so they should have a shared gamma distribution
# for the Dirichlet prior for the starting counts (i.e., alpha needs a prior)
class G1Template(Model):
    """Template model for G1"""

    def __init__(
        self,
        *,
        ic1: npt.NDArray[np.int64],
        ic2: npt.NDArray[np.int64],
        ic3: npt.NDArray[np.int64],
        lc: npt.NDArray[np.int64],
        hc1: npt.NDArray[np.int64],
        hc2: npt.NDArray[np.int64],
        hc3: npt.NDArray[np.int64],
        lt: npt.NDArray[np.float64],
        ht: npt.NDArray[np.float64],
        alpha_alpha: float = DEFAULT_HYPERPARAMS["alpha_alpha"],
        alpha_beta: float = DEFAULT_HYPERPARAMS["alpha_beta"],
        codon_noise_sigma: float = DEFAULT_HYPERPARAMS["codon_noise_sigma"],
        experimental_noise_sigma: float = DEFAULT_HYPERPARAMS[
            "experimental_noise_sigma"
        ],
        **kwargs,
    ):
        # Check shapes
        assert ic1.shape == ic2.shape
        assert ic1.ndim == 2
        assert ic3.ndim == 1
        assert hc1.ndim == 1
        assert hc2.ndim == 2
        assert hc3.ndim == 1
        assert lc.ndim == 2
        assert all(
            count_array.shape[-1] == ic1.shape[-1]
            for count_array in (ic2, ic3, lc, hc1, hc2, hc3)
        )
        assert lt.shape == ht.shape == (3,)

        # Record the number of variants at the DNA level and protein level
        self.n_variants = ic1.shape[-1]

        # The fluorescence gates are constants. We need to add a dimension to make
        # them compatible with the other arrays
        self.log_lt = Constant(np.log(lt[:, None]), togglable=False)
        self.log_ht = Constant(np.log(ht[:, None]), togglable=False)

        # Run inherited init to assign default data
        count_arrays = {
            "ic1": ic1,
            "ic2": ic2,
            "ic3": ic3,
            "lc": lc,
            "hc1": hc1,
            "hc2": hc2,
            "hc3": hc3,
        }
        super().__init__(default_data=count_arrays)

        # Record total counts of each input and output
        for name, array in count_arrays.items():
            setattr(
                self,
                f"total_{name}",
                Constant(np.sum(array, axis=-1, keepdims=True), togglable=False),
            )

        # We need a shared alpha as a hyperparameter on the starting counts
        self.alpha = parameters.Gamma(
            alpha=alpha_alpha, beta=alpha_beta, shape=(self.n_variants,)
        )

        # Starting proportions are described by a Dirichlet distribution
        self.log_theta_t0 = parameters.ExpDirichlet(
            alpha=self.alpha, shape=(3, self.n_variants)
        )

        # We have two sources of noise in fluorescence: One from differing experimental
        # conditions and another that results naturally from varying levels of
        # protein expression due to differing codon usage
        self.experimental_noise = parameters.HalfNormal(sigma=experimental_noise_sigma)
        self.codon_noise = parameters.HalfNormal(
            sigma=codon_noise_sigma, shape=(self.n_variants,)
        )

        # All variants are each described by a mean log fluorescence
        # pylint: disable=no-member
        self.absolute_mean_log_fluorescence = self._set_base_log_fluorescence(**kwargs)
        # pylint: enable=no-member

        # The distributions of fluorescence values will vary from experiment to
        # experiment due to random noise.
        self.experimental_mean_log_fluorescence = parameters.Normal(
            mu=self.absolute_mean_log_fluorescence,
            sigma=self.experimental_noise,
            shape=(3, self.n_variants),
        )

        # We assume that any noise from the detector is caught in the experimental
        # noise term. However, for any experimental mean fluorescence value, we
        # expect a distribution of values resulting from varying expression levels
        # affected by codon usage. Thus, the proportion of variants that will have
        # fluorescence greater than a given threshold is given by evaluating the
        # survival function of the log normal distribution that describes the distribution
        # at the fluorescence value of the threshold
        self.log_theta_low = operations.normalize_log(
            parameters.Normal.log_ccdf(
                x=self.log_lt,
                mu=self.experimental_mean_log_fluorescence,
                sigma=self.codon_noise,
                shape=self.experimental_mean_log_fluorescence.shape,
            )
            + self.log_theta_t0
        )
        self.log_theta_high = operations.normalize_log(
            parameters.Normal.log_ccdf(
                x=self.log_ht,
                mu=self.experimental_mean_log_fluorescence,
                sigma=self.codon_noise,
                shape=self.experimental_mean_log_fluorescence.shape,
            )
            + self.log_theta_t0
        )

        # Now model input counts
        for i, name in enumerate(("ic1", "ic2", "ic3")):
            setattr(
                self,
                name,
                parameters.MultinomialLogTheta(
                    log_theta=self.log_theta_t0[i],
                    N=getattr(self, f"total_{name}"),
                    shape=self.default_data[name].shape,
                ),
            )

        # Model low gate counts
        self.lc = parameters.MultinomialLogTheta(
            log_theta=self.log_theta_low,
            N=self.total_lc,  # pylint: disable=no-member
            shape=self.default_data["lc"].shape,
        )

        # Model high gate counts
        for i, name in enumerate(("hc1", "hc2", "hc3")):
            setattr(
                self,
                name,
                parameters.MultinomialLogTheta(
                    log_theta=self.log_theta_high[i],
                    N=getattr(self, f"total_{name}"),
                    shape=self.default_data[name].shape,
                ),
            )


class G2Template(Model):
    """Models Generation 2 of the nuclease dataset"""

    def __init__(
        self,
        *,
        ic1: npt.NDArray[np.int64],
        ic2: npt.NDArray[np.int64],
        c86: npt.NDArray[np.int64],
        c93: npt.NDArray[np.int64],
        c975: npt.NDArray[np.int64],
        ft: npt.NDArray[np.floating],
        alpha_alpha: float = DEFAULT_HYPERPARAMS["alpha_alpha"],
        alpha_beta: float = DEFAULT_HYPERPARAMS["alpha_beta"],
        codon_noise_sigma: float = DEFAULT_HYPERPARAMS["codon_noise_sigma"],
        **kwargs,
    ):
        # Check shapes
        assert ic1.ndim == 2
        assert ic2.ndim == c86.ndim == c975.ndim == c93.ndim == ft.ndim == 1
        assert (
            ic1.shape[-1]
            == ic2.shape[-1]
            == c86.shape[-1]
            == c975.shape[-1]
            == c93.shape[-1]
        )

        # Get number of variants
        self.n_variants = ic1.shape[-1]

        # Record default data values
        count_arrays = {
            "ic1": ic1,
            "ic2": ic2,
            "c86": c86,
            "c93": c93,
            "c975": c975,
        }
        super().__init__(default_data=count_arrays)

        # Add total counts as constants
        for name, val in count_arrays.items():
            setattr(
                self,
                f"total_{name}",
                Constant(np.sum(val, axis=-1, keepdims=True), togglable=False),
            )

        # Fluorescence gates are constant. Record their logs.
        self.log_thresholds = Constant(np.log(ft), togglable=False)

        # The input libraries should be correlated, but might have slightly different
        # proportions
        self.alpha = parameters.Gamma(
            alpha=alpha_alpha, beta=alpha_beta, shape=(self.n_variants,)
        )
        self.log_theta_t0 = parameters.ExpDirichlet(
            alpha=self.alpha, shape=(2, self.n_variants)
        )

        # Model input counts
        for i, name in enumerate(("ic1", "ic2")):
            setattr(
                self,
                name,
                parameters.MultinomialLogTheta(
                    log_theta=self.log_theta_t0[i],
                    N=getattr(self, f"total_{name}"),
                    shape=count_arrays[name].shape,
                ),
            )

        # We assume a single fluorescence distribution shared across all gates
        # pylint: disable=no-member
        self.log_fluorescence = self._set_base_log_fluorescence(**kwargs)
        # pylint: enable=no-member

        # We have noise in the system due to varying codon expression levels
        self.codon_noise = parameters.HalfNormal(
            sigma=codon_noise_sigma, shape=(self.n_variants,)
        )

        # Update the initial distributions based on a normal survival function,
        # then model counts
        for i, (gate, source) in enumerate((("c86", 0), ("c93", 1), ("c975", 0))):

            # Update proportions
            log_theta_name = f"log_theta_{gate}"
            setattr(
                self,
                log_theta_name,
                operations.normalize_log(
                    parameters.Normal.log_ccdf(
                        x=self.log_thresholds[i],
                        mu=self.log_fluorescence,
                        sigma=self.codon_noise,
                        shape=(self.n_variants,),
                    )
                    + self.log_theta_t0[source]
                ),
            )

            # Model counts
            setattr(
                self,
                gate,
                parameters.MultinomialLogTheta(
                    log_theta=getattr(self, log_theta_name),
                    N=getattr(self, f"total_{gate}"),
                    shape=count_arrays[gate].shape,
                ),
            )


class G3G4Template(Model):
    """Template model for G3 and G4"""

    def __init__(
        self,
        ic1: npt.NDArray[np.int64],
        oc1: npt.NDArray[np.int64],
        ft: npt.NDArray[np.floating],
        alpha: float = DEFAULT_HYPERPARAMS["alpha"],
        codon_noise_sigma: float = DEFAULT_HYPERPARAMS["codon_noise_sigma"],
        **kwargs,
    ):

        # Check shapes
        assert ic1.ndim == ft.ndim == 1
        assert oc1.ndim == 2
        assert ic1.shape[-1] == oc1.shape[-1]

        # Get number of variants
        self.n_variants = ic1.shape[-1]

        # Record default data values
        count_arrays = {"ic1": ic1, "oc1": oc1}
        super().__init__(default_data=count_arrays)

        # Add total counts and log-fluorescence
        for name, val in count_arrays.items():
            setattr(
                self,
                f"total_{name}",
                Constant(np.sum(val, axis=-1, keepdims=True), togglable=False),
            )
        self.log_thresholds = Constant(np.log(ft[:, None]), togglable=False)

        # No prior on the alpha parameter for this model as there is only one input
        # population
        self.log_theta_t0 = parameters.ExpDirichlet(
            alpha=alpha, shape=(self.n_variants,)
        )

        # Set log-fluorescence
        # pylint: disable=no-member
        self.log_fluorescence = self._set_base_log_fluorescence(**kwargs)
        # pylint: enable=no-member

        # Set codon noise
        self.codon_noise = parameters.HalfNormal(
            sigma=codon_noise_sigma, shape=(self.n_variants,)
        )

        # Pass through the survival function
        self.log_theta_filtered = operations.normalize_log(
            parameters.Normal.log_ccdf(
                x=self.log_thresholds,
                mu=self.log_fluorescence,
                sigma=self.codon_noise,
                shape=(
                    self.log_thresholds.shape[0],
                    self.n_variants,
                ),
            )
            + self.log_theta_t0
        )

        # Model counts
        self.ic1 = parameters.MultinomialLogTheta(
            log_theta=self.log_theta_t0,
            N=self.total_ic1,  # pylint: disable=no-member
            shape=ic1.shape,
        )
        self.oc1 = parameters.MultinomialLogTheta(
            log_theta=self.log_theta_filtered,
            N=self.total_oc1,  # pylint: disable=no-member
            shape=oc1.shape,
        )


class LomaxFluorescenceMixIn:
    """Mix in class defining the log-fluorescence as ExpLomax distributed"""

    def _set_base_log_fluorescence(  # pylint: disable=unused-argument
        self,
        lambda_: float = DEFAULT_HYPERPARAMS["lambda_"],
        lomax_alpha: float = DEFAULT_HYPERPARAMS["lomax_alpha_nuclease"],
        **kwargs,
    ):
        # pylint: disable = no-member
        return parameters.ExpLomax(
            lambda_=lambda_, alpha=lomax_alpha, shape=self.n_variants
        )


class ExpFluorescenceMixIn:
    """Mix in class defining the log-fluorescence as Exponentially distributed"""

    def _set_base_log_fluorescence(  # pylint: disable=unused-argument
        self,
        beta: float = DEFAULT_HYPERPARAMS["exp_beta_nuclease"],
        **kwargs,
    ):
        # pylint: disable = no-member
        return parameters.ExpExponential(beta=beta, shape=self.n_variants)


class G1Lomax(G1Template, LomaxFluorescenceMixIn):
    """Describes the G1 phase with lomax-distributed fluorescence"""


class G1Exponential(G1Template, ExpFluorescenceMixIn):
    """Describes the G1 phase with exponentially-distributed fluorescence"""


class G2Lomax(G2Template, LomaxFluorescenceMixIn):
    """Describes the G2 phase with lomax-distributed fluorescence"""


class G2Exponential(G2Template, ExpFluorescenceMixIn):
    """Describes the G2 phase with exponentially-distributed fluorescence"""


class G3G4Lomax(G3G4Template, LomaxFluorescenceMixIn):
    """Describes the G3/G4 phase with lomax-distributed fluorescence"""


class G3G4Exponential(G3G4Template, ExpFluorescenceMixIn):
    """Describes the G3/G4 phase with exponentially-distributed fluorescence"""
