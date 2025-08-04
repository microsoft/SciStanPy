"""Holds classes for working with results from fit models"""

from dms_stan import utils

MLEInferenceRes = utils.lazy_import_from(
    "dms_stan.model.results.mle", "MLEInferenceRes"
)
SampleResults = utils.lazy_import_from("dms_stan.model.results.hmc", "SampleResults")
