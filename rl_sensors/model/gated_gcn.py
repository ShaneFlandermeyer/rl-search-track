import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Optional


class GatedGCN(nn.Module):
  embed_dim: int
  share_kv: bool = False
  normalize_eta: bool = True
  kernel_init: Callable = nn.initializers.xavier_normal()
  dtype: jnp.dtype = jnp.float32
  force_float32_reductions: bool = True

  @nn.compact
  def __call__(self,
               node_features: jax.Array,
               senders: jax.Array,
               receivers: jax.Array,
               edge_features: Optional[jax.Array] = None,
               ) -> jax.Array:
    batch_dims = node_features.shape[:-2]
    num_nodes = node_features.shape[-2]

    segment_sum = jax.ops.segment_sum
    for _ in range(len(batch_dims)):
      segment_sum = jax.vmap(segment_sum, in_axes=(0, 0, None))

    ####################################
    # Node/edge update
    ####################################
    if self.share_kv:
      W = nn.Dense(
          3*self.embed_dim,
          name='W',
          kernel_init=self.kernel_init,
          dtype=self.dtype
      )
      h, Q, KV = jnp.split(W(node_features), 3, axis=-1)
      Q_edges = jnp.take_along_axis(Q, receivers[..., None], axis=-2)
      K_edges = V_edges = jnp.take_along_axis(KV, senders[..., None], axis=-2)
    else:
      W = nn.Dense(
          4*self.embed_dim,
          name='W',
          kernel_init=self.kernel_init,
          dtype=self.dtype
      )
      h, Q, K, V = jnp.split(W(node_features), 4, axis=-1)
      Q_edges = jnp.take_along_axis(Q, receivers[..., None], axis=-2)
      K_edges = jnp.take_along_axis(K, senders[..., None], axis=-2)
      V_edges = jnp.take_along_axis(V, senders[..., None], axis=-2)

    edges = Q_edges + K_edges
    if edge_features is not None:
      W_e = nn.Dense(
          self.embed_dim,
          name='W_e',
          kernel_init=self.kernel_init,
          dtype=self.dtype
      )
      edges = edges + W_e(edge_features)

    #####################################
    # Aggregate edges
    #####################################
    eta = jax.nn.sigmoid(edges)
    if self.force_float32_reductions:
      eta = eta.astype(jnp.float32)
    if self.normalize_eta:
      eta /= jnp.take_along_axis(
          segment_sum(eta, receivers, num_nodes),
          receivers[..., None],
          axis=-2
      ) + 1e-6
      
    nodes = h + segment_sum(
        eta * V_edges, receivers, num_nodes
    ).astype(self.dtype)

    return dict(
        node_features=nodes,
        edge_features=edges,
        senders=senders,
        receivers=receivers,
    )
