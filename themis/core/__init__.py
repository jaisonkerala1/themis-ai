"""Core mathematical primitives for Active Inference."""

from themis.core.distributions import (
    GaussianDist,
    CategoricalDist,
    DirichletDist
)
from themis.core.math_utils import (
    logsumexp,
    softmax_with_temp,
    precision_weighted_error,
    precision_weighted_mse,
    kl_categorical,
    kl_gaussian,
    entropy_categorical,
    entropy_gaussian
)
from themis.core.free_energy import (
    variational_free_energy,
    free_energy_gaussian,
    free_energy_predictive_coding
)
from themis.core.expected_free_energy import (
    epistemic_value_gaussian,
    extrinsic_value_gaussian,
    expected_free_energy_gaussian,
    expected_free_energy_categorical
)
