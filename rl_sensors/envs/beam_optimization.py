import copy
import functools
from typing import Optional, Tuple, Dict, Union, Any, List

import gymnasium as gym
import igraph
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from motpy.distributions.gaussian import Gaussian
from motpy.estimators.kalman.ukf import UnscentedKalmanFilter
from motpy.models.transition import ConstantVelocity
from motpy.models.measurement.radar import Radar2D
from motpy.rfs.bernoulli import MultiBernoulli
from motpy.rfs.poisson import Poisson
from motpy.rfs.tomb import TOMBP
from motpy.distributions.gaussian import uniform_sample_ellipse
from motpy.estimators.kalman.sigma_points import merwe_scaled_sigma_points, merwe_sigma_weights


from rl_sensors.envs.util import merge_poisson, symlog, symexp


class BeamOptimizationEnv(gym.Env):
  def __init__(self):
    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    self.action_dim = np.prod(self.action_space.shape)
    self.predict_edges = True
    self.randomized_reset = False
    self.initialized = False

    # Search config
    self.nx_grid = 7
    self.ny_grid = 7
    self.max_search_nodes = self.nx_grid * self.ny_grid
    self.top_k_search_update = 5
    self.top_k_search_predict = int(1*self.max_search_nodes)

    # Agent config
    self.max_agent_nodes = 10

    # Track config
    self.max_track_history = 3
    self.max_active_tracks = 10
    self.max_track_nodes = self.max_active_tracks * self.max_track_history

    self.max_nodes = self.max_search_nodes + \
        self.max_agent_nodes + self.max_track_nodes
    self.max_edges = (
        # Agent edges
        # Transition
        + (self.max_agent_nodes - 1)
        # Search edges
        # Update
        + 2 * (self.top_k_search_update * self.max_agent_nodes)
        # Predict
        + (2 * self.top_k_search_predict if self.predict_edges else 0)
        # Track edges
        # Transition
        + max((self.max_track_nodes - 1), 0)
        # Update/predict
        + 2 * self.max_track_nodes
    )
    self.observation_space = gym.spaces.Dict(
        current_agent_node_ind=gym.spaces.Discrete(self.max_nodes),
        edge_features=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_edges, 11),
            dtype=np.float32,
        ),
        edge_list=gym.spaces.MultiDiscrete(
            np.full((self.max_edges, 2), self.max_nodes+1),
            start=np.full((self.max_edges, 2), -1)
        ),
        edge_mask=gym.spaces.MultiBinary(self.max_edges),
        global_features=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, 2),
            dtype=np.float32,
        ),
        node_features=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_nodes, 23),
            dtype=np.float32,
        ),
    )

  def reset(
      self,
      seed: Optional[int] = None,
      **kwargs
  ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ###########################
    # Initialize environment
    ###########################
    if seed is not None or not self.initialized:  # New seed resets the scenario
      self.initialized = True
      self.np_random, seed = gym.utils.seeding.np_random(seed)
      self.action_space.seed(seed)
      self.observation_space.seed(seed)

      self.timestep = 0
      self.scenario = dict(
          extents=np.array([
              [-1000, 1000],
              [-1000, 1000]
          ]),
          max_velocity=10,
          birth_rate=1/50,
          clutter_rate=0.0,
          dt=1.0,
          max_trace=50**2,
          num_confirm_detections=3,
          search_priority=1.0,  # alpha
          track_priority=1.0,  # beta
      )
      ###########################
      # Initialize tracker
      ###########################
      self.pos_inds = [0, 2]
      self.vel_inds = [1, 3]
      self.transition_model = ConstantVelocity(
          state_dim=4,
          w=1e-1,
          position_inds=self.pos_inds,
          velocity_inds=self.vel_inds,
          noise_type='continuous',
      )
      self.measurement_model = Radar2D(
          covar=np.diag([5, 1*np.pi/180, 1])**2,
          pos_inds=self.pos_inds,
          vel_inds=self.vel_inds,
      )
      self.state_estimator = UnscentedKalmanFilter(
          transition_model=self.transition_model,
          measurement_model=self.measurement_model,
          state_subtract_fn=np.subtract,
          state_average_fn=np.average,
          measurement_subtract_fn=Radar2D.subtract_fn,
          measurement_average_fn=Radar2D.average_fn,
          sigma_params=dict(alpha=1e-3, beta=2, kappa=0),
      )

      # Birth distribution
      xmin, xmax = self.scenario['extents'][0]
      ymin, ymax = self.scenario['extents'][1]
      dx = 0.5*(xmax - xmin) / self.nx_grid
      dy = 0.5*(ymax - ymin) / self.ny_grid
      x = np.linspace(xmin+dx, xmax-dx, self.nx_grid)
      y = np.linspace(ymin+dy, ymax-dy, self.ny_grid)

      grid_x, grid_y = np.meshgrid(x, y, indexing='ij')
      grid_x, grid_y = grid_x.ravel(), grid_y.ravel()
      n_grid = len(grid_x)
      dvx = dvy = self.scenario['max_velocity']
      grid_vx = grid_vy = np.zeros_like(grid_x)
      self.birth_grid = np.stack([
          grid_x, grid_vx, grid_y, grid_vy,
      ], axis=-1)

      birth_covars = np.diag(
          np.array([dx, dvx, dy, dvy])**2
      )[None, ...].repeat(n_grid, axis=0)
      birth_distribution = Gaussian(
          mean=self.birth_grid,
          covar=birth_covars,
          weight=np.full(n_grid, self.scenario['birth_rate'] / n_grid)
      )

      # Initial undetected distribution
      init_num_objects = 1
      undetected_state = Gaussian(
          mean=self.birth_grid,
          covar=birth_covars,
          weight=np.full(n_grid, init_num_objects / n_grid)
      )
      init_ground_truth_inds = self.np_random.choice(
          a=np.arange(birth_distribution.shape[0]),
          size=init_num_objects,
          p=birth_distribution.weight / np.sum(birth_distribution.weight)
      )
      init_ground_truth = uniform_sample_ellipse(
          n=1,
          mean=birth_distribution.mean[init_ground_truth_inds],
          covar=birth_distribution.covar[init_ground_truth_inds],
          rng=self.np_random,
      )
      self.ground_truth = [list(state) for state in init_ground_truth]

      self.tracker = TOMBP(
          poisson=Poisson(state=undetected_state),
          mb=None,
          birth_distribution=birth_distribution
      )
      self.global_track_counter = 0

      ############################
      # Initialize sensor
      ############################
      self.sensor_feature_keys = ['steering_angle', 'beamwidth']
      self.sensor = dict(
          # Static parameters
          max_range=1500,
          min_beamwidth=20*np.pi/180,
          max_beamwidth=40*np.pi/180,
          # Sensor state
          state_vector=np.zeros(self.transition_model.state_dim),
          steering_angle=0.0,
          beamwidth=0.0,
      )

      ############################
      # Graph setup
      ############################
      self.graph = igraph.Graph(directed=True)
      self.update_graph()
    elif self.randomized_reset:
      # Randomize birth grid
      xmin, xmax = self.scenario['extents'][0]
      ymin, ymax = self.scenario['extents'][1]
      dx = 0.5*(xmax - xmin) / self.nx_grid
      dy = 0.5*(ymax - ymin) / self.ny_grid
      birth_pos_noise = np.zeros_like(self.birth_grid)
      birth_pos_noise[:, self.pos_inds] = self.np_random.uniform(
          low=[-dx/2, -dy/2],
          high=[dx/2, dy/2],
          size=self.birth_grid[:, self.pos_inds].shape
      )
      self.tracker.birth_distribution.mean = self.birth_grid + birth_pos_noise

    obs = self.get_obs()
    info = {}

    return obs, info

  def step(
      self,
      action: np.ndarray
  ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
    self.timestep += 1

    # Update simulation
    self.sensor = self.update_sensor_state(state=self.sensor, action=action)
    self.update_ground_truth(dt=self.scenario['dt'])
    measurements = self.measure(
        states=np.array([path[-1] for path in self.ground_truth])
    )

    # Predict step
    self.tracker = self.tracker.predict(
        state_estimator=self.state_estimator,
        dt=self.scenario['dt'],
        ps_model=functools.partial(
            self.ps,
            scenario=self.scenario,
            sensor=self.sensor,
            pos_inds=self.pos_inds,
        ),
    )
    self.tracker.poisson, self.tracker.poisson_metadata = merge_poisson(
        distribution=self.tracker.poisson,
        metadata=self.tracker.poisson_metadata,
        source_inds=np.arange(len(self.tracker.poisson)//2),
        target_inds=np.arange(
            len(self.tracker.poisson)//2, len(self.tracker.poisson)
        )
    )

    # Update step
    volume = (self.sensor['beamwidth'] / (2*np.pi)) * \
        (np.pi * self.sensor['max_range']**2)
    self.tracker = self.tracker.update(
        measurements=measurements,
        state_estimator=self.state_estimator,
        pd_model=functools.partial(
            self.pd,
            sensor=self.sensor,
            scenario=self.scenario,
            pos_inds=self.pos_inds,
        ),
        lambda_fa=self.scenario['clutter_rate'] / volume,
        pg=0.999,
        sensor_pos=self.sensor['state_vector'][self.pos_inds],
        sensor_vel=self.sensor['state_vector'][self.vel_inds],
        min_poisson_pd=0.1,
    )
    if len(self.tracker.mb) > 0:
      self.tracker.mb, self.tracker.mb_metadata = self.tracker.mb.prune(
          valid_fn=lambda mb: np.logical_and(
              mb.r > 1e-4,
              np.linalg.trace(
                  mb.state.covar[
                      np.ix_(
                          np.arange(len(mb)), self.pos_inds, self.pos_inds
                      )
                  ]
              ) < 5*self.scenario['max_trace']
          ),
          meta=self.tracker.mb_metadata,
      )
    if len(self.tracker.mb) > 0:
      # Update track metadata
      track_pd = self.pd(
          state=self.tracker.mb.state,
          sensor=self.sensor,
          scenario=self.scenario,
          pos_inds=self.pos_inds,
      )
      track_ps = self.ps(
          state=self.tracker.mb.state,
          scenario=self.scenario,
          sensor=self.sensor,
          pos_inds=self.pos_inds,
      )
      track_measurement = self.measurement_model(
          self.tracker.mb.state.mean,
          sensor_pos=self.sensor['state_vector'][self.pos_inds],
          sensor_vel=self.sensor['state_vector'][self.vel_inds],
          noise=False,
          rng=self.np_random,
      )
      track_quality = self.track_quality(self.tracker.mb)
      track_active = self.tracker.mb.r > 1e-2
      for i, meta in enumerate(self.tracker.mb_metadata):
        if meta['new']:
          meta['id'] = f'track_{self.global_track_counter}'
          self.global_track_counter += 1
        new = meta.get('new', False)
        updated = meta.get('updated', False)
        missed = track_pd[i] > 0 and not (updated or new)
        measurement_type = (
            'update' if (updated or new) else
            'miss' if missed else
            'predict'
        )
        num_detections = meta.get('num_detections', 0)
        if measurement_type == 'update':
          num_detections += 1
        confirmed = meta.get('confirmed', False) or \
            num_detections >= self.scenario['num_confirm_detections']
        self.tracker.mb_metadata[i].update(
            pd=track_pd[i],
            ps=track_ps[i],
            quality=track_quality[i] if confirmed else 0.0,
            measurement=track_measurement[i],
            measurement_type=measurement_type,
            num_detections=num_detections,
            confirmed=confirmed,
            active=track_active[i],
        )

    # Env update
    self.update_graph()
    obs = self.get_obs()
    reward = self.get_reward()
    terminated = truncated = False
    info = {}

    return obs, reward, terminated, truncated, info

  def update_graph(self) -> None:
    I = np.eye(3)
    node_label_map = dict(
        agent=I[0],
        search=I[1],
        track=I[2],
    )
    edge_label_map = dict(
        transition=I[0],
        predict=I[1],
        update=I[2],
    )

    # Delete predict nodes
    if len(self.graph.es) > 0:
      self.graph.delete_vertices(self.graph.vs(measurement_type_eq='predict'))
      self.graph.delete_edges(self.graph.es(type_eq='predict'))

    #############################
    # Search nodes
    #############################
    search_pd = self.pd(
        state=self.tracker.poisson.state,
        sensor=self.sensor,
        scenario=self.scenario,
        pos_inds=self.pos_inds,
    )
    search_ps = self.ps(
        state=self.tracker.poisson.state,
        scenario=self.scenario,
        sensor=self.sensor,
        pos_inds=self.pos_inds,
    )
    search_measurement = self.measurement_model(
        self.tracker.poisson.state.mean,
        sensor_pos=self.sensor['state_vector'][self.pos_inds],
        sensor_vel=self.sensor['state_vector'][self.vel_inds],
        noise=False,
        rng=self.np_random,
    )
    if self.timestep == 0:
      search_ids = [
          f'search_{i}' for i in range(len(self.tracker.poisson))
      ]
      self.graph.add_vertices(
          n=len(self.tracker.poisson),
          attributes=dict(
              type='search',
              label=[node_label_map['search']],
              name=search_ids,
              id=search_ids,
              state_vector=self.tracker.poisson.state.mean,
              # Agent features
              sensor_state=np.zeros((1, len(self.sensor_feature_keys))),
              is_current_agent=False,
              total_search_weight=0.0,
              num_confirmed_tracks=0,
              total_track_quality=0.0,
              # Track features
              measurement_type='none',
              track_quality=0.0,
              existence_probability=0.0,
              confirmed=False,
              active=True,
          )
      )
    search_nodes = self.graph.vs(type_eq='search')
    search_nodes['timestep'] = self.timestep
    search_nodes['state_vector'] = self.tracker.poisson.state.mean
    search_nodes['covar_diag'] = np.sqrt(
        np.diagonal(self.tracker.poisson.state.covar, axis1=-1, axis2=-2)
    )
    search_nodes['survival_probability'] = search_ps
    w = self.tracker.poisson.state.weight
    search_nodes['weight'] = w / (w.mean() + 1e-8)

    ##############################
    # Agent nodes
    ##############################
    sensor_state = np.concatenate(
        [np.atleast_1d(self.sensor[k]) for k in self.sensor_feature_keys]
    )
    num_confirmed_tracks = np.count_nonzero([
        m['confirmed'] and m['active'] for m in self.tracker.mb_metadata
    ])
    total_track_quality = np.sum([
        m['quality'] for m in self.tracker.mb_metadata
        if m['confirmed'] and m['active']
    ])

    current_agent_node = dict(
        type='agent',
        label=node_label_map['agent'],
        name=f'agent_t{self.timestep}',
        id='agent',
        timestep=self.timestep,
        state_vector=self.sensor['state_vector'],
        covar_diag=np.zeros(self.transition_model.state_dim),
        survival_probability=0.0,
        # Search features
        weight=0.0,
        # Agent features
        sensor_state=sensor_state,
        is_current_agent=True,
        total_search_weight=self.tracker.poisson.state.weight.sum(),
        num_confirmed_tracks=num_confirmed_tracks,
        total_track_quality=total_track_quality,
        # Track features
        measurement_type='none',
        track_quality=0.0,
        existence_probability=0.0,
        confirmed=False,
        active=True,
    )
    agent_edges = []
    agent_edge_attributes = dict(
        type=[],
        label=[],
        direction=[],
        relative_pose=[],
        pd=[],
        measurement=[],
    )
    # Agent transition edge
    if self.timestep > 0:
      last_agent = self.graph.vs.find(
          type_eq='agent', timestep_eq=self.timestep-1
      )
      last_agent['is_current_agent'] = False

      rel_pos = (
          current_agent_node['state_vector'][self.pos_inds]
          - last_agent['state_vector'][self.pos_inds]
      )
      rel_orientation = np.arctan2(
          current_agent_node['state_vector'][self.vel_inds[1]],
          current_agent_node['state_vector'][self.vel_inds[0]]
      ) - np.arctan2(
          last_agent['state_vector'][self.vel_inds[1]],
          last_agent['state_vector'][self.vel_inds[0]]
      )
      rel_orientation = (rel_orientation + np.pi) % (2 * np.pi) - np.pi
      relative_agent_pose = np.array([rel_pos[0], rel_pos[1], rel_orientation])

      agent_edges.extend([
          (last_agent['name'], current_agent_node['name']),
          #   (current_agent_node['name'], last_agent['name']),
      ])
      agent_edge_attributes.update(
          type=agent_edge_attributes['type'] +
          ['transition'],
          label=agent_edge_attributes['label'] +
          [edge_label_map['transition']],
          direction=agent_edge_attributes['direction'] +
          [1],
          relative_pose=agent_edge_attributes['relative_pose'] +
          [relative_agent_pose],
          pd=agent_edge_attributes['pd'] +
          [0.0],
          measurement=agent_edge_attributes['measurement'] +
          [np.zeros(self.measurement_model.measurement_dim)],
      )
    # Search update edges
    if self.timestep > 0:
      search_detect_inds = np.where(search_pd > 0)[0]
      if len(search_detect_inds) > self.top_k_search_update:
        search_detect_inds = self.np_random.choice(
            a=search_detect_inds,
            size=self.top_k_search_update,
            replace=False,
            p=search_pd[search_detect_inds] /
            np.sum(search_pd[search_detect_inds])
        )
      agent_edges.extend(
          [
              (current_agent_node['name'], search_nodes[i]['name'])
              for i in search_detect_inds
          ] +
          [
              (search_nodes[i]['name'], current_agent_node['name'])
              for i in search_detect_inds
          ]
      )
      agent_edge_attributes.update(
          type=agent_edge_attributes['type'] +
          2*len(search_detect_inds)*['update'],
          label=agent_edge_attributes['label'] +
          2*len(search_detect_inds)*[edge_label_map['update']],
          direction=agent_edge_attributes['direction'] +
          len(search_detect_inds)*[1] + len(search_detect_inds)*[-1],
          pd=agent_edge_attributes['pd'] +
          2*search_pd[search_detect_inds].tolist(),
          relative_pose=agent_edge_attributes['relative_pose'] +
          2*len(search_detect_inds)*[np.zeros(3)],
          measurement=agent_edge_attributes['measurement'] +
          2*search_measurement[search_detect_inds].tolist(),
      )
    else:
      search_detect_inds = []

    if self.predict_edges and self.top_k_search_predict > 0:
      search_predict_inds = np.setdiff1d(
          np.arange(len(search_nodes)), search_detect_inds
      )
      if len(search_predict_inds) > self.top_k_search_predict:
        search_predict_inds = search_predict_inds[
            np.argpartition(
                search_nodes[search_predict_inds]['weight'],
                -self.top_k_search_predict
            )[-self.top_k_search_predict:]
        ]
      agent_edges.extend(
          [
              (current_agent_node['name'], search_nodes[i]['name'])
              for i in search_predict_inds
          ] +
          [
              (search_nodes[i]['name'], current_agent_node['name'])
              for i in search_predict_inds
          ]
      )
      agent_edge_attributes.update(
          type=agent_edge_attributes['type'] +
          2*len(search_predict_inds)*['predict'],
          label=agent_edge_attributes['label'] +
          2*len(search_predict_inds)*[edge_label_map['predict']],
          direction=agent_edge_attributes['direction'] +
          len(search_predict_inds)*[1] + len(search_predict_inds)*[-1],
          pd=agent_edge_attributes['pd'] +
          2*search_pd[search_predict_inds].tolist(),
          relative_pose=agent_edge_attributes['relative_pose'] +
          2*len(search_predict_inds)*[np.zeros(3)],
          measurement=agent_edge_attributes['measurement'] +
          2*search_measurement[search_predict_inds].tolist(),
      )

    self.graph.add_vertex(**current_agent_node)
    self.graph.add_edges(es=agent_edges, attributes=agent_edge_attributes)

    # Remove old agent nodes
    agent_nodes = self.graph.vs(type_eq='agent')
    if len(agent_nodes) > self.max_agent_nodes:
      num_to_delete = len(agent_nodes) - self.max_agent_nodes
      agent_nodes[:num_to_delete]['delete'] = True
    else:
      agent_nodes['delete'] = False

    #################################
    # Track nodes
    #################################
    if len(self.tracker.mb) > 0 and self.max_active_tracks > 0:
      # Delete stale track nodes
      track_ids = [meta['id'] for meta in self.tracker.mb_metadata]
      self.graph.vs(type_eq='track', id_notin=track_ids)['delete'] = True

      # Select active tracks
      active_indices = [
        i for i, m in enumerate(self.tracker.mb_metadata)
        if m["active"]
      ]
      active_indices.sort(
        key=lambda i: self.tracker.mb.r[i],
        reverse=True,
      )
      active_indices = active_indices[:self.max_active_tracks]
      active_mb = self.tracker.mb[active_indices]
      active_mb_metadata = [self.tracker.mb_metadata[i] for i in active_indices]

      # Collect track info for this timestep
      track_node_attributes = dict(
          type=len(active_mb)*['track'],
          label=len(active_mb)*[node_label_map['track']],
          timestep=len(active_mb)*[self.timestep],
          name=[],
          id=[],
          state_vector=active_mb.state.mean,
          covar_diag=np.sqrt(np.diagonal(
              active_mb.state.covar, axis1=-1, axis2=-2
          )),
          survival_probability=[m['ps'] for m in active_mb_metadata],
          # Search features
          weight=np.zeros(len(active_mb)),
          # Agent features
          sensor_state=len(active_mb) *
          [np.zeros(len(self.sensor_feature_keys))],
          is_current_agent=np.zeros(len(active_mb), dtype=bool),
          total_search_weight=np.zeros(len(active_mb)),
          num_confirmed_tracks=np.zeros(len(active_mb)),
          total_track_quality=np.zeros(len(active_mb)),
          # Track features
          existence_probability=active_mb.r,
          track_quality=[m['quality'] for m in active_mb_metadata],
          measurement_type=[],
          confirmed=[],
          active=[],
      )
      track_edge_attributes = dict(
          type=[],
          label=[],
          direction=[],
          relative_pose=[],
          pd=[],
          measurement=[],
      )
      track_edges = []
      track_nodes = self.graph.vs(type_eq='track')
      track_nodes['active'] = False
      for i, meta in enumerate(active_mb_metadata):
        # Update node attributes
        track_id = meta['id']
        track_node_name = f"{track_id}_t{self.timestep}"
        measurement_type = meta['measurement_type']
        confirmed = meta['confirmed']
        track_node_attributes.update(
            id=track_node_attributes['id'] + [track_id],
            name=track_node_attributes['name'] + [track_node_name],
            measurement_type=track_node_attributes['measurement_type'] +
            [measurement_type],
            confirmed=track_node_attributes['confirmed'] + [confirmed],
            active=track_node_attributes['active'] + [True],
        )

        # Transition edge
        track_history = track_nodes(id_eq=track_id)
        track_history['active'] = True
        if len(track_history) > 0:
          update_history = track_history(
              measurement_type_in=['update', 'miss']
          )
          if len(update_history) > 0:
            last_update = update_history[-1]

            rel_pos = (
                active_mb.state.mean[i][self.pos_inds]
                - last_update['state_vector'][self.pos_inds]
            )
            rel_orientation = np.arctan2(
                active_mb.state.mean[i][self.vel_inds[1]],
                active_mb.state.mean[i][self.vel_inds[0]]
            ) - np.arctan2(
                last_update['state_vector'][self.vel_inds[1]],
                last_update['state_vector'][self.vel_inds[0]]
            )
            rel_orientation = (rel_orientation + np.pi) % (2 * np.pi) - np.pi
            relative_track_pose = np.array([
                rel_pos[0], rel_pos[1], rel_orientation
            ])

            track_edges.extend([
                (last_update['name'], track_node_name),
                # (track_node_name, last_update['name'])
            ])
            track_edge_attributes.update(
                type=track_edge_attributes['type'] +
                ['transition'],
                label=track_edge_attributes['label'] +
                [edge_label_map['transition']],
                relative_pose=track_edge_attributes['relative_pose'] +
                [relative_track_pose],
                pd=track_edge_attributes['pd'] +
                [0.0],
                measurement=track_edge_attributes['measurement'] +
                [np.zeros(self.measurement_model.measurement_dim)],
                direction=track_edge_attributes['direction'] +
                [1],
            )

        # Track update edge
        if measurement_type in ['update', 'miss']:
          edge_type = 'update'
        else:
          edge_type = 'predict' if self.predict_edges else None

        if edge_type is not None:
          track_edges.extend([
              (current_agent_node['name'], track_node_name),
              (track_node_name, current_agent_node['name']),
          ])
          track_edge_attributes.update(
              type=track_edge_attributes['type'] +
              2*[edge_type],
              label=track_edge_attributes['label'] +
              2*[edge_label_map[edge_type]],
              relative_pose=track_edge_attributes['relative_pose'] +
              2*[np.zeros(3)],
              pd=track_edge_attributes['pd'] + 2*[meta['pd']],
              measurement=track_edge_attributes['measurement'] +
              2*[meta['measurement']],
              direction=track_edge_attributes['direction'] +
              [1, -1],
          )

        # Track graph pruning
        if len(track_history) >= self.max_track_history:
          track_history[:-(self.max_track_history-1)]['delete'] = True

      # Add track nodes and edges
      self.graph.add_vertices(
          n=len(active_mb), attributes=track_node_attributes
      )
      self.graph.add_edges(es=track_edges, attributes=track_edge_attributes)
    else:
      self.graph.vs(type_eq='track')['delete'] = True

    ###############################
    # Global graph update
    ###############################
    deleted_nodes = self.graph.vs(delete_eq=True)
    if len(deleted_nodes) > 0:
      self.graph.delete_vertices(deleted_nodes)
    self.graph.vs['age'] = self.timestep - np.array(self.graph.vs['timestep'])

  def get_obs(self) -> Dict[str, Any]:
    ###########################
    # Nodes
    ###########################
    graph = self.graph.subgraph(self.graph.vs(active_eq=True))
    nodes = graph.vs
    node_keys = [
        # Shared features
        'label',
        'age',
        'state_vector',
        'covar_diag',
        'survival_probability',
        # Agent features
        'sensor_state',
        'is_current_agent',
        'total_search_weight',
        'num_confirmed_tracks',
        'total_track_quality',
        # Search features
        'weight',
        # Track features
        'track_quality',
        'existence_probability',
        'confirmed',
    ]
    node_features = np.concatenate(
        [
            np.array(nodes[k]).reshape((len(nodes), -1)) for k in node_keys
        ], axis=-1
    )
    node_features = np.pad(
        node_features,
        ((0, self.max_nodes - len(nodes)), (0, 0)),
        mode='constant', constant_values=0,
    )
    current_agent_node_ind = nodes.find(
        type_eq='agent', timestep_eq=self.timestep
    ).index

    ###########################
    # Edges
    ###########################
    edges = graph.es
    if len(edges) > 0:
      edge_keys = [
          'label',
          'direction',
          # Transition features
          'relative_pose',
          # Measurement features
          'measurement',
          'pd',
      ]
      edge_features = np.concatenate(
          [
              np.array(edges[k]).reshape((len(edges), -1)) for k in edge_keys
          ], axis=-1
      )

      edge_list = np.array(graph.get_edgelist())
      # Pad edges
      edge_features = np.pad(
          edge_features,
          ((0, self.max_edges - len(edges)), (0, 0)),
          mode='constant', constant_values=0,
      )
      edge_list = np.pad(
          edge_list,
          ((0, self.max_edges - len(edges)), (0, 0)),
          mode='constant', constant_values=-1,
      )
      edge_mask = np.arange(self.max_edges) < len(edges)
    else:
      edge_features = np.zeros(
          self.observation_space['edge_features'].shape,
          dtype=self.observation_space['edge_features'].dtype
      )
      edge_list = np.zeros(
          self.observation_space['edge_list'].shape,
          dtype=self.observation_space['edge_list'].dtype
      )
      edge_mask = np.zeros(
          self.observation_space['edge_mask'].shape,
          dtype=self.observation_space['edge_mask'].dtype
      )

    ###########################
    # Global features
    ###########################
    global_features = np.stack([
        np.array([self.scenario['search_priority']]),
        np.array([self.scenario['track_priority']]),
    ], axis=-1)

    obs = dict(
        current_agent_node_ind=current_agent_node_ind,
        edge_features=symlog(edge_features).astype(np.float32),
        edge_list=edge_list,
        edge_mask=edge_mask,
        global_features=symlog(global_features).astype(np.float32),
        node_features=symlog(node_features).astype(np.float32),
    )
    return obs

  def get_reward(self) -> float:
    w = self.tracker.poisson.state.weight
    search_reward = -w.sum()
    alpha = self.scenario['search_priority']
    beta = self.scenario['track_priority']

    confirmed = np.array([
        meta['confirmed'] and meta['active']
        for meta in self.tracker.mb_metadata
    ])
    num_confirmed = np.count_nonzero(confirmed)

    # Search-only case
    if num_confirmed == 0:
      return alpha * search_reward

    # Search-and-track case
    track_qualities = self.track_quality(self.tracker.mb[confirmed])
    track_reward = track_qualities.sum()
    total_reward = alpha * search_reward + beta * track_reward
    return total_reward

  def update_ground_truth(self, dt: float) -> None:
    if len(self.ground_truth) > 0:
      states = np.atleast_2d([x[-1] for x in self.ground_truth])
      next_states = self.transition_model(
          states, dt=dt, noise=True, rng=self.np_random
      )

      # Object survival
      ps = self.ps(
          state=next_states,
          scenario=self.scenario,
          sensor=self.sensor,
          pos_inds=self.pos_inds,
      )
      survived = ps > 0
      for i, path in enumerate(self.ground_truth.copy()):
        if survived[i]:
          path.append(next_states[i])
        else:
          self.ground_truth.remove(path)

    # New object birth
    if len(self.ground_truth) >= self.max_active_tracks:
      return
    num_birth = self.np_random.poisson(lam=self.scenario['birth_rate'] * dt)
    if num_birth > 0:
      birth_distribution = self.tracker.birth_distribution
      inds = self.np_random.choice(
          a=np.arange(birth_distribution.shape[0]),
          size=num_birth,
          p=birth_distribution.weight / np.sum(birth_distribution.weight)
      )
      new_states = uniform_sample_ellipse(
          n=1,
          mean=birth_distribution.mean[inds],
          covar=birth_distribution.covar[inds],
          rng=self.np_random,
      )
      self.ground_truth.extend([list(state) for state in new_states])

  def measure(self, states: np.ndarray, noise: bool = True) -> np.ndarray:
    Z = np.zeros((0, self.measurement_model.measurement_dim))

    # Object measurements
    if len(states) > 0:
      pd = self.pd(
          states,
          sensor=self.sensor,
          scenario=self.scenario,
          pos_inds=self.pos_inds,
      )
      detected = self.np_random.uniform(size=len(self.ground_truth)) < pd
      if np.any(detected):
        Z = self.measurement_model(
            states[detected],
            sensor_pos=self.sensor['state_vector'][self.pos_inds],
            sensor_vel=self.sensor['state_vector'][self.vel_inds],
            noise=noise,
            rng=self.np_random,
        )

    # Clutter measurements
    n_clutter = self.np_random.poisson(
        lam=self.scenario['clutter_rate'] * self.scenario['dt']
    )
    if n_clutter > 0:
      Z_clutter = self.measure_clutter(n_clutter)
      Z = np.concatenate([Z, Z_clutter], axis=0)

    return Z

  def measure_clutter(self, n: int) -> np.ndarray:
    r = self.np_random.uniform(0, self.sensor['max_range'], n)
    az = self.np_random.uniform(
        self.sensor['steering_angle'] - self.sensor['beamwidth'], 
        self.sensor['steering_angle'] + self.sensor['beamwidth'],
        n
    )
    v = self.np_random.uniform(
        -self.scenario['max_velocity'], self.scenario['max_velocity'], n
    )
    return np.stack([r, az, v], axis=-1)


  def track_quality(self, tracks: MultiBernoulli) -> np.ndarray:
    """Compute track quality based on trace of covariance."""
    covars = tracks.state.covar[
        np.ix_(np.arange(len(tracks)), self.pos_inds, self.pos_inds)
    ]
    traces = np.linalg.trace(covars)
    track_quality = 1 - (traces / self.scenario['max_trace']).clip(0, 1)
    return track_quality

  def render(self, graph: igraph.Graph = None):
    if graph is None:
      graph = self.graph

    graph = self.graph.subgraph(self.graph.vs(active_eq=True))
    nodes = graph.vs

    plt.clf()

    ##################################
    # Sensor visualization
    ##################################
    beamwidth = self.sensor['beamwidth']
    steering_angle = self.sensor['steering_angle']
    plt.gca().add_patch(matplotlib.patches.Wedge(
        center=self.sensor['state_vector'][self.pos_inds],
        r=np.max(np.linalg.norm(self.scenario['extents'], axis=-1)),
        theta1=np.degrees(steering_angle - beamwidth/2),
        theta2=np.degrees(steering_angle + beamwidth/2),
        color='r',
        alpha=0.5,
    ))

    ##################################
    # Scenario visualization
    ##################################
    # Agent nodes
    agent_nodes = nodes(type_eq='agent')
    if len(agent_nodes) > 0:
      agent_pos = np.array(agent_nodes['state_vector'])[:, self.pos_inds]
      plt.scatter(agent_pos[:, 0], agent_pos[:, 1], c='orange')


    # Search nodes
    search_nodes = nodes(type_eq='search')
    nx, ny = 100, 100
    search_grid = np.meshgrid(
        np.linspace(*self.scenario['extents'][0], nx),
        np.linspace(*self.scenario['extents'][1], ny)
    )
    search_grid = np.stack([search_grid[0].ravel(), search_grid[1].ravel()]).T
    # Plot gaussian mixture (likelihood) as an image
    mixture = np.zeros((nx, ny))

    w = search_nodes['weight']
    P = self.tracker.birth_distribution.covar
    for i in range(len(search_nodes)):
      mixture += w[i] * scipy.stats.multivariate_normal.pdf(
          search_grid,
          mean=search_nodes[i]['state_vector'][self.pos_inds],
          cov=P[i][np.ix_(self.pos_inds, self.pos_inds)],
      ).reshape((nx, ny))
    plt.imshow(
        mixture,
        extent=self.scenario['extents'].ravel(),
        origin='lower',
        aspect='auto'
    )

    # Edges
    if len(graph.es) > 0:
      source_pos = np.array(
          [nodes[e.source]['state_vector'][self.pos_inds] for e in graph.es]
      )
      target_pos = np.array(
          [nodes[e.target]['state_vector'][self.pos_inds] for e in graph.es]
      )
      edge_x = np.vstack([source_pos[:, 0], target_pos[:, 0]])
      edge_y = np.vstack([source_pos[:, 1], target_pos[:, 1]])

      # Plot predict edges
      predict = np.array([e['type'] == 'predict' for e in graph.es])
      transition = np.array([e['type'] == 'transition' for e in graph.es])
      update = np.array([e['type'] == 'update' for e in graph.es])
      plt.plot(edge_x[:, transition], edge_y[:, transition], 
               'red', linewidth=0.75, label='Transition edges', zorder=0),
      plt.plot(edge_x[:, update], edge_y[:, update],
               'black', linewidth=0.75, label='Update edges', zorder=0)

    # Ground truth
    for path in self.ground_truth:
      path = np.array(path)
      plt.plot(path[:, 0], path[:, 2], 'r--')
      p = np.array(path[-1])
      plt.plot(p[0], p[2], '*', color='red', label='Ground Truth')

    # Track nodes
    track_nodes = nodes(type_eq='track')
    if len(track_nodes) > 0:
      track_pos = np.array(track_nodes['state_vector'])[:, self.pos_inds]
      color = [
          'green' if track_nodes[i]['confirmed'] else 'blue'
          for i in range(len(track_nodes))
      ]
      plt.scatter(
          track_pos[:, 0], track_pos[:, 1], c=color, s=50, label='Track'
      )
      for i, pos in enumerate(track_pos):
        if track_nodes[i]['age'] > 0:
            continue
        plt.text(
            pos[0], pos[1],
            f"q={track_nodes[i]['track_quality']:.2f}\n r={track_nodes[i]['existence_probability']:.2f}",
            fontsize=8,
            color='white',
            ha='center',
            va='center',
        )
        
    # Current agent
    current_agent_node = nodes.find(type_eq='agent', is_current_agent_eq=True)
    plt.scatter(
        current_agent_node['state_vector'][self.pos_inds][0],
        current_agent_node['state_vector'][self.pos_inds][1],
        c='orange',
        label='Agent',
        zorder=1,
    )

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')

    plt.xlim(self.scenario['extents'][0])
    plt.ylim(self.scenario['extents'][1])
    plt.pause(0.01)
    plt.draw()

  def update_sensor_state(
      self, state: Dict[str, Any], action: np.ndarray
  ) -> Dict[str, Any]:
    steering_angle = np.interp(action[0], [-1, 1], [-np.pi, np.pi])
    beamwidth = np.interp(
        action[1],
        [-1, 1],
        [self.sensor['min_beamwidth'], self.sensor['max_beamwidth']]
    )

    new_state = state.copy()
    new_state.update(
        steering_angle=steering_angle,
        beamwidth=beamwidth,
        action=action,
    )
    return new_state

  @staticmethod
  def pd(
      state: Union[np.ndarray, Gaussian],
      sensor: Dict[str, Any],
      scenario: Dict[str, Any],
      pos_inds: List[int],
  ) -> np.ndarray:
    if isinstance(state, Gaussian):
      mu = state.mean[..., pos_inds]

      sensor_pos = sensor['state_vector'][pos_inds]
      beamwidth = sensor['beamwidth']
      steering_angle = sensor['steering_angle']
      beam_pd = np.interp(
          sensor['beamwidth'],
          xp=[sensor['min_beamwidth'], sensor['max_beamwidth']],
          fp=[0.9, 0.6],
      )

      az = np.mod(
          np.arctan2(
              mu[..., 1] - sensor_pos[1], mu[..., 0] - sensor_pos[0]
          ) - steering_angle + np.pi, 2*np.pi
      ) - np.pi
      ranges = np.linalg.norm(mu - sensor_pos, axis=-1)
      pd = np.where(
          np.logical_and(
              abs(az) <= beamwidth/2,
              ranges <= sensor['max_range']
          ),
          beam_pd, 0.0
      )

    else:
      x = state[..., pos_inds]

      sensor_pos = sensor['state_vector'][pos_inds]
      beamwidth = sensor['beamwidth']
      steering_angle = sensor['steering_angle']
      beam_pd = np.interp(
          sensor['beamwidth'],
          xp=[sensor['min_beamwidth'], sensor['max_beamwidth']],
          fp=[0.9, 0.6],
      )

      az = np.mod(
          np.arctan2(
              x[..., 1] - sensor_pos[1], x[..., 0] - sensor_pos[0]
          ) - steering_angle + np.pi, 2*np.pi
      ) - np.pi
      ranges = np.linalg.norm(x - sensor_pos, axis=-1)
      pd = np.where(
          np.logical_and(
              abs(az) <= beamwidth/2,
              ranges <= sensor['max_range']
          ),
          beam_pd, 0.0
      )
    return pd

  @staticmethod
  def ps(
      state: Union[np.ndarray, Gaussian],
      scenario: Dict[str, Any],
      sensor: Dict[str, Any],
      pos_inds: List[int],
  ) -> np.ndarray:
    extents = scenario['extents']
    if isinstance(state, Gaussian):
      x = state.mean[..., pos_inds]

      ps = np.where(
          np.logical_and.reduce([
              x[..., 0] >= extents[0, 0],
              x[..., 0] <= extents[0, 1],
              x[..., 1] >= extents[1, 0],
              x[..., 1] <= extents[1, 1],
          ]), 0.997, 0
      )

    else:
      x = state[..., pos_inds]
      ps = np.where(
          np.logical_and.reduce([
              x[..., 0] >= extents[0, 0],
              x[..., 0] <= extents[0, 1],
              x[..., 1] >= extents[1, 0],
              x[..., 1] <= extents[1, 1],
          ]), 0.997, 0
      )

    return ps
