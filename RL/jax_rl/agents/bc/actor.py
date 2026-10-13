from typing import Tuple

import jax.numpy as jnp

from jax_rl.datasets import Batch
from jax_rl.networks.common import InfoDict, Model, Params


def update(actor: Model, batch: Batch) -> Tuple[Model, InfoDict]:
    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params}, batch.observations)
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -log_probs.mean()
        return actor_loss, {'actor_loss': actor_loss}

    return actor.apply_gradient(loss_fn)


def _taylor_expm(A: jnp.ndarray, n_terms: int = 30) -> jnp.ndarray:
    """Differentiable, vmap-friendly matrix exponential via truncated Taylor
    series -- avoids relying on jax.scipy.linalg.expm's vmap support. A has
    shape (..., d, d); for the eps ranges and generator norms used in the
    equivariance-penalty dynamics check (||eps * G|| well under ~7), 30 terms
    is comfortably converged."""
    d = A.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(d, dtype=A.dtype), A.shape)
    term = eye
    total = eye
    for n in range(1, n_terms):
        term = jnp.einsum('...ij,...jk->...ik', term, A) / n
        total = total + term
    return total
