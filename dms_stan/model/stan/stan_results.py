import arviz as az
import cmdstanpy
import numpy.typing as npt


class SampleResults:
    def __init__(self, fit: cmdstanpy.CmdStanMCMC, data: dict[str, npt.NDArray]):
        # Store the CmdStanMCMC object and the observed data
        self.fit = fit
        self.data = data

        # Build the arviz object
        self.az = az.from_cmdstanpy(
            posterior=fit,
            posterior_predictive=self._get_ppc(),
            observed_data={k + "_ppc": v for k, v in data.items()},
        )

    def _get_ppc(self) -> list[str]:

        # Note the difference between the provided observed data and the known
        # observed data
        expected_observations = {
            name.removesuffix("_ppc")
            for name in self.fit.stan_variables().keys()
            if name.endswith("_ppc")
        }
        actual_observations = set(self.data.keys())
        if additional_observations := actual_observations - expected_observations:
            raise ValueError(
                "The following observations were provided as data, but there were "
                "no samples generated for them by the Stan model: "
                + ", ".join(additional_observations)
            )
        if missing_observations := expected_observations - actual_observations:
            raise ValueError(
                "The following observations were expected to be provided as data, "
                "but were not: " + ", ".join(missing_observations)
            )

        return list(expected_observations)

    # Missing attributes are pulled from the CmdStanMCMC object
    def __getattr__(self, name):
        return getattr(self.fit, name)

    @classmethod
    def from_csv(cls, path: str, data: dict[str, npt.NDArray]):
        """
        Functions exactly as `cmdstanpy.from_csv`, but returns a StanResults object.
        """
        return cls(fit=cmdstanpy.from_csv(path), data=data)
