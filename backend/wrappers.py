import numpy as np
import gym
import dm_env
from dm_env import specs
from bsuite.utils.gym_wrapper import DMEnvFromGym, space2spec
from collections import deque
from .timestep import ExtendedTimeStep
from gym.spaces import Box, Dict

_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY = (
    "`env.action_spec()` must return a single `BoundedArray`, got: {}.")
_MUST_BE_FINITE = "All values in `{name}` must be finite, got: {bounds}."
_MUST_BROADCAST = (
    "`{name}` must be broadcastable to shape {shape}, got: {bounds}.")


class DMEnvFromGymWrapper(DMEnvFromGym):
    def __init__(self, gym_env: gym.Env, height=84, width=84):
        self.gym_env = gym_env
        # Convert gym action and observation spaces to dm_env specs.
        obs_box = self.gym_env.observation_space
        obs_box = Box(
            np.full((height, width, 3), 0),
            np.full((height, width, 3), 255)
        )

        self._observation_spec = space2spec(obs_box,
                                            name='observation')
        self._action_spec = space2spec(self.gym_env.action_space, name='action')
        self._reset_next_step = True

    def is_successful(self, obs=None):
        if hasattr(self.gym_env, 'is_successful') and self.gym_env.is_successful(obs=obs):
            return True
        else:
            return False

    def compute_reward(self, obs):
        reward = self.gym_env.compute_reward(obs=obs)
        # meta world environments return a list of rewards, the first one is the env reward
        if isinstance(reward, list):
            reward = reward[0]
        return reward

    def render(self, height=84, width=84, mode="rgb_array"):
        return self.gym_env.render(mode=mode, height=height, width=width)

    @property
    def sim(self):
        return self.gym_env.sim

    def __getattr__(self, name):
        return getattr(self.gym_env, name)

class FrankaDMEnvFromGymWrapper(DMEnvFromGym):
    def __init__(self, gym_env: gym.Env, state='qpos', height=100, width=100):
        self.gym_env = gym_env
        # Convert gym action and observation spaces to dm_env specs.
        obs_box = self.gym_env.observation_space
        self._state = state
        if self._state == 'qpos':
            state_spec = self.gym_env.observation_space['lowdim_qpos']
        elif self._state == 'ee':
            state_spec = self.gym_env.observation_space['lowdim_ee']

        obs_box = Dict({'images': Box(np.full((6, height, width), 0),
                                 np.full((6, height, width), 255)),
                        'state': state_spec})
        self._observation_spec = space2spec(obs_box,
                                            name='observation')
        self._action_spec = space2spec(self.gym_env.action_space, name='action')
        self._reset_next_step = False

    def _franka_transform_observation(self, obs):
        fp_image = obs['hand_img_obs'].transpose(2, 0, 1)
        tp_image = obs['third_person_img_obs'].transpose(2, 0, 1)
        state = obs['lowdim_qpos'] if self._state == 'qpos' else obs['lowdim_ee']

        new_obs = {'images': np.concatenate([fp_image, tp_image], axis=0),
                   'state': state}
        return new_obs

    def reset(self):
        time_step = super().reset()
        new_obs = self._franka_transform_observation(time_step.observation)
        return time_step._replace(observation=new_obs)

    def step(self, action):
        time_step = super().step(action)
        new_obs = self._franka_transform_observation(time_step.observation)
        return time_step._replace(observation=new_obs)

    def __getattr__(self, name):
        return getattr(self.gym_env, name)

class FrankaObsActionDTypeWrapper(dm_env.Environment):
    '''
    obs (env.observation_spec().dtype) -> obs (obs_dtype)
    actions (action_dtype) -> (env.action_spec().dtype)

    This wrapper mediates whenever the action / observation 
    dtypes are different between environments and algorithm.
    '''

    def __init__(self, env, action_dtype, obs_dtype):
        self._env = env
        self._action_dtype = action_dtype
        self._obs_dtype = obs_dtype

        # rewrite action spec
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            action_dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

        # rewrite observation spec
        self._observation_spec = {
            key: self._get_spec(spec, key) for key, spec in self._env.observation_spec().items()
        }
        self._observation_spec['name'] = 'observation'

    def _get_spec(self, old_spec, name):
        return specs.BoundedArray(
            old_spec.shape,
            self._obs_dtype[name],
            old_spec.minimum,
            old_spec.maximum,
            name,
        )

    def _modify_obs_dtype(self, time_step):
        updated_obs = {key: time_step.observation[key].astype(self._obs_dtype[key]) for key in self._observation_spec.keys() if key != 'name'}
        return time_step._replace(observation=updated_obs)

    def step(self, action):
        action = action.astype(self._action_dtype)
        return self._modify_obs_dtype(self._env.step(action))

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._modify_obs_dtype(self._env.reset())

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ObsActionDTypeWrapper(dm_env.Environment):
    '''
    obs (env.observation_spec().dtype) -> obs (obs_dtype)
    actions (action_dtype) -> (env.action_spec().dtype)

    Environments operate in float64 (provide float64 obs, 
    expect float64 actions), while the algorithms i/o float32.
    This wrapper mediates whenever the action / observation 
    dtypes are different between environments and algorithm.
    '''

    def __init__(self, env, action_dtype, obs_dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            action_dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )
        wrapped_obs_spec = env.observation_spec()
        self._observation_spec = specs.BoundedArray(
            wrapped_obs_spec.shape,
            obs_dtype,
            wrapped_obs_spec.minimum,
            wrapped_obs_spec.maximum,
            "observation",
        )

    def _modify_obs_dtype(self, time_step):
        updated_obs = time_step.observation.astype(self._observation_spec.dtype)
        return time_step._replace(observation=updated_obs)

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._modify_obs_dtype(self._env.step(action))

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._modify_obs_dtype(self._env.reset())

    def __getattr__(self, name):
        return getattr(self._env, name)

class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionScaleWrapper(dm_env.Environment):
  """Wraps a control environment to rescale actions to a specific range."""
  __slots__ = ("_action_spec", "_env", "_transform")

  def __init__(self, env, minimum, maximum):
    """Initializes a new action scale Wrapper.
    Args:
      env: Instance of `dm_env.Environment` to wrap. Its `action_spec` must
        consist of a single `BoundedArray` with all-finite bounds.
      minimum: Scalar or array-like specifying element-wise lower bounds
        (inclusive) for the `action_spec` of the wrapped environment. Must be
        finite and broadcastable to the shape of the `action_spec`.
      maximum: Scalar or array-like specifying element-wise upper bounds
        (inclusive) for the `action_spec` of the wrapped environment. Must be
        finite and broadcastable to the shape of the `action_spec`.
    Raises:
      ValueError: If `env.action_spec()` is not a single `BoundedArray`.
      ValueError: If `env.action_spec()` has non-finite bounds.
      ValueError: If `minimum` or `maximum` contain non-finite values.
      ValueError: If `minimum` or `maximum` are not broadcastable to
        `env.action_spec().shape`.
    """
    action_spec = env.action_spec()
    if not isinstance(action_spec, specs.BoundedArray):
      raise ValueError(_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY.format(action_spec))

    minimum = np.array(minimum)
    maximum = np.array(maximum)
    shape = action_spec.shape
    orig_minimum = action_spec.minimum
    orig_maximum = action_spec.maximum
    orig_dtype = action_spec.dtype

    def validate(bounds, name):
      if not np.all(np.isfinite(bounds)):
        raise ValueError(_MUST_BE_FINITE.format(name=name, bounds=bounds))
      try:
        np.broadcast_to(bounds, shape)
      except ValueError:
        raise ValueError(_MUST_BROADCAST.format(
            name=name, bounds=bounds, shape=shape))

    validate(minimum, "minimum")
    validate(maximum, "maximum")
    validate(orig_minimum, "env.action_spec().minimum")
    validate(orig_maximum, "env.action_spec().maximum")

    scale = (orig_maximum - orig_minimum) / (maximum - minimum)

    def transform(action):
      new_action = orig_minimum + scale * (action - minimum)
      return new_action.astype(orig_dtype, copy=False)

    dtype = np.result_type(minimum, maximum, orig_dtype)
    self._action_spec = action_spec.replace(
        minimum=minimum, maximum=maximum, dtype=dtype)
    self._env = env
    self._transform = transform

  def step(self, action):
    return self._env.step(self._transform(action))

  def reset(self):
    return self._env.reset()

  def observation_spec(self):
    return self._env.observation_spec()

  def action_spec(self):
    return self._action_spec

  def __getattr__(self, name):
    return getattr(self._env, name)

class GoalVisionWrapper(dm_env.Environment):
    '''
    Use for frame stacking and attaching goal frames for *simulation* envs.
    Also, transposes the image (2, 0, 1) to comply with the codebase.
    (In general, only use this wrapper for multi-goal environments,
    and preferably provide goal_states and vision_states)
    When using a simulation environment:
        (a) if the environment has multiple goals:
            (i) provide both goal_states and vision_goal_states (aligned along axis=0)
            (ii) provide only goal_states, assuming vision_goal_states can be set
        (b) if the environment has no goals:
            keep goal_states = None and the wrapper will just render the current state
            and return (along with frame stacking)
    '''
    def __init__(self, env, num_frames, goal_states=None, height=84, width=84, vision_goal_states=None):
        self._env = env

        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self.height = height
        self.width = width
        self.goal_states = goal_states
        # assume that the goal states and vision goal states are aligned if available
        self.vision_goal_states = vision_goal_states
        if self.goal_states is not None:
            self._initialize_goal_images()
            self.obs_dim = goal_states.shape[1]
            self.cur_goal = None

        wrapped_obs_spec = env.observation_spec()
        pixels_shape = wrapped_obs_spec.shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        # add depth for goal image if conditioning on goals
        goal_image = 3 if self.goal_states is not None else 0
        self._obs_spec = specs.BoundedArray(shape=np.concatenate([[pixels_shape[2] * num_frames + goal_image], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _initialize_goal_images(self):
        ''' render the goal images if not already available.
            this works only when state = qpos and can be set in the environment.
            likely only works for tabletop_manipulation. '''
        if self.vision_goal_states is None:
            self.vision_goal_states = np.zeros([self.goal_states.shape[0], 3, self.height, self.width], dtype=np.uint8)
            for idx in range(self.goal_states.shape[0]):
                self._env.set_state(self.goal_states[idx])
                self._env.sim.forward()
                self.vision_goal_states[idx] = np.expand_dims(self._env.render(self.height, self.width).transpose(2, 0, 1), 0).astype(np.uint8)

    def goal_idx(self, goal):
        return np.argmin(np.linalg.norm(self.goal_states - goal, axis=1))

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        if self.goal_states is not None:
            cur_goal = self.vision_goal_states[self.goal_idx(time_step.observation[self.obs_dim:])]
            obs = np.concatenate([obs, cur_goal], axis=0)            
            return time_step._replace(observation=obs)
        else:
            return time_step._replace(observation=obs)

    def get_goal_images(self):
        return self.vision_goal_states

    def _extract_pixels(self):
        pixels = self._env.render(self.height, self.width)
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]

        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels()
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels()
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
