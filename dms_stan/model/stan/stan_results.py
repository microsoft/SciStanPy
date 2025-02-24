import cmdstanpy
import numpy.typing as npt


class SampleResults:
    def __init__(
        self, fit: cmdstanpy.CmdStanMCMC, observed_data: dict[str, npt.NDArray]
    ):
        self.fit = fit
        self.observed_data = observed_data

    # Missing attributes are pulled from the CmdStanMCMC object
    def __getattr__(self, name):
        return getattr(self.fit, name)

    @classmethod
    def from_csv(cls, path: str, observed_data: dict[str, npt.NDArray]):
        """
        Functions exactly as `cmdstanpy.from_csv`, but returns a StanResults object.
        """
        return cls(fit=cmdstanpy.from_csv(path), observed_data=observed_data)
