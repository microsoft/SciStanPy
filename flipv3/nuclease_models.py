"""Holds models for the nuclease dataset"""

from abc import abstractmethod

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
        absolute_noise_sigma: float = DEFAULT_HYPERPARAMS["absolute_noise_sigma"],
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
        self.lt = Constant(lt[:, None], togglable=False)
        self.ht = Constant(ht[:, None], togglable=False)

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
        self.log_theta_t0 = parameters.Dirichlet(
            alpha=self.alpha, shape=(3, self.n_variants)
        )

        # We have two sources of noise in fluorescence: One from differing expression
        # levels from varying codon usage and another capturing everything else.
        # Codon noise operates on the log scale while experimental noise operates
        # on the absolute scale
        self.codon_noise = parameters.HalfNormal(sigma=codon_noise_sigma)
        self.absolute_noise = parameters.HalfNormal(sigma=absolute_noise_sigma)

        # All variants are each described by a mean log fluorescence
        self.mean_log_fluorescence = self._set_mean_log_fluorescence(**kwargs)

        # The distributions of fluorescence values will vary from experiment to
        # experiment due to random noise
        self.log_fluorescence = parameters.ExpNormal(
            mu=self.mean_log_fluorescence,
            sigma=self.absolute_noise,
            shape=(3, self.n_variants),
        )

        # Each variant population contains multiple alternate DNA sequences, leading
        # to variability in expression levels. We thus expect the distribution of
        # fluorescences to be log normal. The proportion of variants that will have
        # a fluorescence greater than the threshold is thus given by the survival
        # function of the log normal distribution
        self.updated_low_unnorm = (
            parameters.LogNormal.log_ccdf(
                x=self.lt,
                mu=self.log_fluorescence,
                sigma=self.codon_noise,
                shape=self.log_fluorescence.shape,
            )
            + self.log_theta_t0
        )
        self.updated_high_unnorm = (
            parameters.LogNormal.log_ccdf(
                x=self.ht,
                mu=self.log_fluorescence,
                sigma=self.codon_noise,
                shape=self.log_fluorescence.shape,
            )
            + self.log_theta_t0
        )

        # Renormalize the updated proportions
        self.log_theta_low = operations.normalize_log(self.updated_low_unnorm)
        self.log_theta_high = operations.normalize_log(self.updated_high_unnorm)

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


class LomaxFluorescenceMixIn:
    """Mix in class defining the log-fluorescence as ExpLomax distributed"""

    def _set_mean_log_fluorescence(  # pylint: disable=unused-argument
        self,
        lambda_: float = DEFAULT_HYPERPARAMS["lambda_"],
        lomax_alpha: float = DEFAULT_HYPERPARAMS["lomax_alpha"],
        **kwargs,
    ):
        # pylint: disable = no-member
        return parameters.ExpLomax(
            lambda_=lambda_, alpha=lomax_alpha, shape=self.n_variants
        )


class ExpFluorescenceMixIn:
    """Mix in class defining the log-fluorescence as Exponentially distributed"""

    def _set_mean_log_fluorescence(  # pylint: disable=unused-argument
        self,
        beta: float = DEFAULT_HYPERPARAMS["exp_beta"],
        **kwargs,
    ):
        # pylint: disable = no-member
        return parameters.ExpExponential(scale=beta, shape=self.n_variants)


class G1Lomax(G1Template, LomaxFluorescenceMixIn):
    """Describes the G1 phase with lomax-distributed fluorescence"""


class G1Exponential(G1Template, ExpFluorescenceMixIn):
    """Describes the G1 phase with exponentially-distributed fluorescence"""
