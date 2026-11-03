from typing import Dict, Any, Callable

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
from einops import rearrange

from gated_gcn import GatedGCN


class GraphEncoder(nn.Module):
  embed_dim: int
  num_layers: int
  kernel_init: Callable = nn.initializers.xavier_normal()
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, input: Dict[str, Any]):
    ######################
    # Pre-processing
    ######################
    current_agent_ind = jnp.asarray(input['current_agent_node_ind'])[..., None]
    edge_features = input['edge_features']
    edge_list = input['edge_list']
    edge_mask = input['edge_mask']
    global_features = input['global_features']
    node_features = input['node_features']
    node_mask = input['node_mask']
    senders, receivers = edge_list[..., 0], edge_list[..., 1]
    senders = jnp.where(edge_mask, senders, -1)
    receivers = jnp.where(edge_mask, receivers, -1)

    node_features = nn.Dense(
        self.embed_dim, kernel_init=self.kernel_init, dtype=self.dtype
    )(node_features)
    edge_features = nn.Dense(
        self.embed_dim, kernel_init=self.kernel_init, dtype=self.dtype
    )(edge_features)
    global_features = nn.Dense(
        self.embed_dim, kernel_init=self.kernel_init, dtype=self.dtype
    )(global_features)

    node_features = node_features + global_features
    ######################
    # Graph Processing
    ######################
    graph = dict(
        node_features=node_features,
        edge_features=edge_features,
        senders=senders,
        receivers=receivers,
    )

    for i in range(self.num_layers):
      nodes = graph['node_features']
      edges = graph['edge_features']

      # Graph processing
      graph['node_features'] = nn.RMSNorm(dtype=self.dtype)(
          graph['node_features']
      )
      graph['edge_features'] = nn.RMSNorm(dtype=self.dtype)(
          graph['edge_features']
      )
      gnn = GatedGCN(
          embed_dim=self.embed_dim,
          share_kv=False,
          normalize_eta=True,
          kernel_init=self.kernel_init,
          dtype=self.dtype,
          force_float32_reductions=True,
      )
      graph = gnn(**graph)
      graph['node_features'] = nodes + nn.elu(graph['node_features'])
      graph['edge_features'] = edges + nn.elu(graph['edge_features'])

    ######################
    # Decode
    ######################
    current_agent = jnp.take_along_axis(
        graph['node_features'], current_agent_ind[..., None], axis=-2
    )
    global_pool = jnp.max(
        graph['node_features'],
        axis=-2,
        keepdims=True,
        where=node_mask[..., None],
        initial=-jnp.inf,
    )
    x = jnp.concatenate([current_agent, global_pool], axis=-1)
    x = nn.Dense(
        self.embed_dim, kernel_init=self.kernel_init, dtype=self.dtype
    )(x)

    x = nn.RMSNorm(dtype=self.dtype)(x)
    x = rearrange(x, '... n d -> ... (n d)')
    return x


if __name__ == '__main__':
  import rl_sensors.envs
  import gymnasium as gym
  env = gym.make('MobileSurveillance-v0')

  obs, _ = env.reset()
  obs, reward, term, trunc, info = env.step(env.action_space.sample())
  model = GraphEncoder(
      embed_dim=128,
      num_layers=3,
      dtype=jnp.bfloat16,
  )
  params = model.init(jax.random.PRNGKey(0), obs)
  print(model.tabulate(jax.random.PRNGKey(0), obs))
  out = model.apply(params, obs)
  print(out.shape)
