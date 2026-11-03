import numpy as np
from motpy.rfs.poisson import Poisson
from motpy.distributions import Gaussian
from motpy.distributions.gaussian import merge_gaussians
from typing import *
from scipy.special import erf

def norm_cdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
  """Compute the CDF of a normal distribution."""
  return 0.5 * (1 + erf((x - mean) / np.sqrt(2 * var)))


def symlog(x):
  return np.sign(x) * np.log1p(np.abs(x))

def symexp(x):
    return np.sign(x) * (np.expm1(np.abs(x)))

def merge_poisson(
    distribution: Poisson,
    metadata: List[Dict[str, Any]],
    source_inds: np.ndarray,
    target_inds: np.ndarray
) -> Poisson:
  mu = distribution.state.mean[[source_inds, target_inds]].swapaxes(0, 1)
  P = distribution.state.covar[[source_inds, target_inds]].swapaxes(0, 1)
  w = distribution.state.weight[[source_inds, target_inds]].swapaxes(0, 1)
  merged_mu, merged_P, merged_w = merge_gaussians(
      means=mu, covars=P, weights=w
  )
  merged_state = Gaussian(
      mean=merged_mu,
      covar=merged_P,
      weight=merged_w
  )

  merged_metadata = [metadata[i] for i in target_inds]

  return Poisson(state=merged_state), merged_metadata
