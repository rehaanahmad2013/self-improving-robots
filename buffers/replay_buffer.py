import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from backend.timestep import ExtendedTimeStep, StepType

def episode_len(episode):
    # subtract -1 because the dummy first transition
    return episode['action'].shape[0] - 1

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())

def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f, allow_pickle=True)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def render_state(self, state, env, height, width):
        env.set_state(state)
        env.sim.forward()
        return env.render(height, width).transpose(2,0,1)

    def add_state_to_frame(self, arr, state, env, height, width, first=False):
        if first:
            for idx in range(0, arr.shape[0], 3):
                arr[idx:idx+3] = self.render_state(state, env, height, width)
        else:
            arr = np.concatenate([arr[3:], self.render_state(state, env, height, width)], axis=0)
        return arr

    def add_offline_data_franka(self, demos):
        '''Expects demos arranged as [obs_i, a_i, next_obs_i, r_i, done_i].'''

        obs = demos['observations']
        next_obs = demos['next_observations']
        acts = demos['actions']
        term = demos['dones']
        rewards = demos['rewards']
        default_action = np.zeros((self._data_specs[1].shape[0],), dtype=np.float32)

        # clip off actions to avoid NaN with BC regularization
        acts[acts == 1] = 1. - 1e-7
        acts[acts == -1] = -1. + 1e-7

        for idx in range(acts.shape[0]):
            if idx == 0 or term[idx-1][0]:
                time_step = ExtendedTimeStep(observation=obs[idx],
                                             step_type=StepType.FIRST,
                                             action=default_action,
                                             reward=0.0,
                                             discount=1.0)
                self.add(time_step)

            time_step = ExtendedTimeStep(observation=next_obs[idx],
                                         step_type=StepType.MID, # always use MID step type
                                         action=acts[idx],
                                         reward=rewards[idx][0],
                                         discount=1.0)
            self.add(time_step, end_episode=term[idx][0])

    def add_offline_vision_data(self, demos, default_action, env, num_frames, height, width, inf_bootstrap=False, clip_off_actions=False):
        obs = demos['observations']
        acts = demos['actions']
        if clip_off_actions:
            eps = 1e-6
            acts[acts == 1] = 1 - eps
            acts[acts == -1] = -1 + eps

        rew = demos['rewards']
        term = demos['terminals']
        next_o = demos['next_observations']
        
        stacked_frames = np.zeros([3 * num_frames, height, width])
        success_states = []

        # demos may already be from vision
        if len(obs.shape) == 4:
            already_vision = True
        else:
            already_vision = False

        for idx in range(obs.shape[0]):
            obs_dim = obs[idx].shape[0] // 2
            if idx == 0 or term[idx - 1][0]:
                if not already_vision:
                    stacked_frames = self.add_state_to_frame(stacked_frames, obs[idx][:obs_dim], env, height, width, first=True)
                    goal_state = env.get_goal_images()[env.goal_idx(obs[idx][obs_dim:])]
                else:
                    stacked_frames = np.repeat(obs[idx,:,:,:3].transpose(2,0,1), num_frames, axis=0)
                    goal_state = obs[idx,:,:,3:].transpose(2,0,1)

                time_step = ExtendedTimeStep(observation=np.concatenate([stacked_frames, goal_state], axis=0),
                                             step_type=StepType.FIRST,
                                             action=default_action,
                                             reward=0.0,
                                             discount=1.0)
                self.add(time_step)

            if not already_vision:
                stacked_frames = self.add_state_to_frame(stacked_frames, next_o[idx][:obs_dim], env, height, width, first=False)
                goal_state = env.get_goal_images()[env.goal_idx(next_o[idx][obs_dim:])]
            else:
                stacked_frames = np.concatenate([stacked_frames[3:], next_o[idx,:,:,:3].transpose(2,0,1)], axis=0)
                goal_state = next_o[idx,:,:,3:].transpose(2,0,1)
            new_vision_obs = np.concatenate([stacked_frames, goal_state], axis=0)

            # check this line out: you have a lot of demo examples where rew > 0
            if rew[idx] > 0.0:
                if not already_vision:
                    success_states.append(new_vision_obs[-6:])
                else:
                    success_states.append(next_o[idx].transpose(2,0,1))

            '''
            Case A: dense reward function. Assume terminal states don't have a reward == 1.0
            If inf_bootstrap => Add with StepType.MID (except first step).
            Case B: sparse reward function. Assume r = {0, 1}.
            If inf_bootstrap => Add with StepType.LAST if terminal state + (reward = 1.0).
            Assumes a goal reaching setup where MDP transitions into an absorbing state with 0 reward.
            '''
            if term[idx][0]:
                if inf_bootstrap and rew[idx][0] != 1.0:
                    time_step = ExtendedTimeStep(observation=new_vision_obs, step_type=StepType.MID, action=acts[idx], reward=rew[idx][0], discount=1.0)
                else:
                    time_step = ExtendedTimeStep(observation=new_vision_obs, step_type=StepType.LAST, action=acts[idx], reward=rew[idx][0], discount=1.0)
                self.add(time_step, end_episode=True)
            else:
                time_step = ExtendedTimeStep(observation=new_vision_obs, step_type=StepType.MID, action=acts[idx], reward=rew[idx][0], discount=1.0)
                self.add(time_step)

        return np.stack(success_states, axis=0)

    def add_offline_data(self, demos, default_action, inf_bootstrap=False, clip_off_actions=False):
        obs = demos['observations']
        acts = demos['actions']
        if clip_off_actions:
            eps = 1e-7
            acts[acts == 1] = 1. - eps
            acts[acts == -1] = -1. + eps

        rew = demos['rewards']
        term = demos['terminals']
        next_o = demos['next_observations']
        
        success_states = []
        for idx in range(obs.shape[0]):            
            if idx == 0 or term[idx - 1][0]:
                time_step = ExtendedTimeStep(observation=obs[idx], step_type=StepType.FIRST, action=default_action, reward=0.0, discount=1.0)
                self.add(time_step)

            if rew[idx] > 0.0:
                success_states.append(next_o[idx])

            '''
            Case A: dense reward function. Assume terminal states don't have a reward == 1.0
            If inf_bootstrap => Add with StepType.MID (except first step).
            Case B: sparse reward function. Assume r = {0, 1}.
            If inf_bootstrap => Add with StepType.LAST if terminal state + (reward = 1.0).
            Assumes a goal reaching setup where MDP transitions into an absorbing state with 0 reward.
            '''
            if term[idx][0]:
                if inf_bootstrap and rew[idx][0] != 1.0:
                    time_step = ExtendedTimeStep(observation=next_o[idx], step_type=StepType.MID, action=acts[idx], reward=rew[idx][0], discount=1.0)
                else:
                    time_step = ExtendedTimeStep(observation=next_o[idx], step_type=StepType.LAST, action=acts[idx], reward=rew[idx][0], discount=1.0)
                self.add(time_step, end_episode=True)
            else:
                time_step = ExtendedTimeStep(observation=next_o[idx], step_type=StepType.MID, action=acts[idx], reward=rew[idx][0], discount=1.0)
                self.add(time_step)

        return np.stack(success_states, axis=0)

    def _check_shapes(self, spec, val):
        if isinstance(spec, dict):
            for key in spec.keys():
                if key == 'name':
                    continue
                if spec[key].shape != val[key].shape or spec[key].dtype != val[key].dtype:
                    return False
            return True
        return spec.shape == val.shape and spec.dtype == val.dtype

    def add(self, time_step, end_episode=False):
        '''to allow infinite bootstrapping, the step type needs to be MID.
        To indicate the end of episode to the buffer, use `end_episode`.'''
        for spec in self._data_specs:
            if isinstance(spec, dict):
                spec_name = spec['name']
            else:
                spec_name = spec.name
            value = time_step[spec_name]
            if spec_name == 'discount':
                value = np.expand_dims(time_step.discount, 0).astype(np.float32)
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert self._check_shapes(spec, value)
            self._current_episode[spec_name].append(value)

        # TODO: add step_type to specs, instead of manually adding here
        self._current_episode['step_type'].append(time_step.step_type)

        if time_step.last() or end_episode:
            episode = dict()
            for spec in self._data_specs:
                if isinstance(spec, dict):
                    spec_name = spec['name']
                else:
                    spec_name = spec.name
                value = self._current_episode[spec_name]
                # if the spec is a dictionary, just keep the entries as a list
                if isinstance(spec, dict):
                    episode[spec_name] = value
                else:
                    episode[spec_name] = np.array(value, spec.dtype)
            episode['step_type'] = np.array(self._current_episode['step_type'])

            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot, custom_reward_callable=None):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._custom_reward = custom_reward_callable

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if not worker_info else worker_info.id
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        cur_nstep = self._nstep

        if episode_len(episode) - cur_nstep + 1 < 1:
            cur_nstep = 1

        idx = np.random.randint(0, episode_len(episode) - cur_nstep + 1) + 1

        obs = episode['observation'][idx - 1]
        step_type = episode['step_type'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + cur_nstep - 1]
        next_step_type = episode['step_type'][idx + cur_nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(cur_nstep):
            if self._custom_reward is not None:
                with torch.no_grad():
                    step_reward = self._custom_reward(episode['observation'][idx + i])
            else:
                step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount

        # return latents too if stored in the episode
        if 'latent' in episode.keys():
            latent = episode['latent'][idx - 1]
            next_latent = episode['latent'][idx + cur_nstep - 1]
            return (obs, latent, action, reward, discount, next_obs, next_latent, step_type, next_step_type)
        else:
            return (obs, action, reward, discount, next_obs, step_type, next_step_type)

    def __iter__(self):
        while True:
            yield self._sample()

def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)

def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep=1, discount=0.99, custom_reward_callable=None):
    max_size_per_worker = max_size // max(1, num_workers)
    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot,
                            custom_reward_callable=custom_reward_callable)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader

class EARLExpertRB(IterableDataset):
    def __init__(self, demos):
        self._episodes = []
        obses, actions = [], []
        for idx in range(demos['observations'].shape[0]):
            obses.append(demos['observations'][idx].transpose(2, 0, 1))
            actions.append(demos['actions'][idx])
            if demos['terminals'][idx]:
                episode = dict(observation=np.array(obses), action=np.array(actions))
                self._episodes.append(episode)
                obses, actions = [], []

    def _sample_episode(self):
        episode = random.choice(self._episodes)
        return episode

    def _sample(self):
        episode = self._sample_episode()
        idx = np.random.randint(0, episode_len(episode)) + 1
        obs = episode['observation'][idx]
        action = episode['action'][idx]

        return (obs, action)

    def __iter__(self):
        while True:
            yield self._sample()

def make_expert_replay_loader(replay_dir, batch_size, num_demos, obs_type, earl_demos):
    iterable = EARLExpertRB(earl_demos)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=2,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader
