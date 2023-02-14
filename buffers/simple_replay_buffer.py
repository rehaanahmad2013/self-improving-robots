import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from backend.timestep import ExtendedTimeStep

class SimpleReplayBuffer:
    
    def __init__(self, data_specs, max_size, batch_size=None, replay_dir=None, discount=0.99,
                 filter_transitions=True, with_replacement=True):
        self._data_specs = data_specs
        self._max_size = max_size # assume max_size >= total transitions
        self._batch_size = batch_size
        self._discount = discount
        self._replay_dir = replay_dir
        self._replay_buffer = {}
        self._filter_transitions = filter_transitions
        self._with_replacement = with_replacement
        for spec in self._data_specs:
            self._replay_buffer[spec.name] = np.empty((max_size, *spec.shape), dtype=spec.dtype)
        self._replay_buffer['step_type'] = np.empty((max_size, 1), dtype=np.int32)
        self._num_transitions = 0

    def __len__(self):
        return self._num_transitions
    
    def add_offline_data(self, demos, default_action):
        obs = demos['observations']
        acts = demos['actions']
        rew = demos['rewards']
        term = demos['terminals']
        next_o = demos['next_observations']

        for idx in range(obs.shape[0]):            
            if idx == 0 or term[idx - 1][0]:
                time_step = ExtendedTimeStep(observation=obs[idx], step_type=0, action=default_action, reward=0.0, discount=1.0)
                self.add(time_step)

            if term[idx][0]:
                time_step = ExtendedTimeStep(observation=next_o[idx], step_type=2, action=acts[idx], reward=rew[idx][0], discount=1.0)
            else:
                time_step = ExtendedTimeStep(observation=next_o[idx], step_type=1, action=acts[idx], reward=rew[idx][0], discount=1.0)
            self.add(time_step)

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if spec.name == 'discount':
                value = np.expand_dims(time_step.discount * self._discount, 0).astype('float32')
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)

            assert spec.shape == value.shape and spec.dtype == value.dtype
            np.copyto(self._replay_buffer[spec.name][self._num_transitions], value)

        np.copyto(self._replay_buffer['step_type'][self._num_transitions], time_step.step_type)
        self._num_transitions += 1

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self, batch_size=None, filter_transitions=None, with_replacement=None):
        batch_size = self._batch_size if batch_size is None else batch_size
        filter_transitions = self._filter_transitions if filter_transitions is None else filter_transitions
        with_replacement = self._with_replacement if with_replacement is None else with_replacement

        if with_replacement:
            idxs = np.random.randint(0, len(self), size=batch_size)
        else:
            # do not use np.random.choice, it gets much slower as the size increases
            idxs = np.array(random.sample(range(len(self)), batch_size), dtype=np.int64)

        if filter_transitions:
            filtered_idxs = []
            for idx in idxs:
                if self._replay_buffer['step_type'][idx]:
                    filtered_idxs.append(idx)
            idxs = np.array(filtered_idxs, dtype=np.int64)

        return (self._replay_buffer['observation'][idxs - 1],
                self._replay_buffer['action'][idxs],
                self._replay_buffer['reward'][idxs],
                self._replay_buffer['discount'][idxs],
                self._replay_buffer['observation'][idxs],
                self._replay_buffer['step_type'][idxs - 1].squeeze(1),
                self._replay_buffer['step_type'][idxs].squeeze(1),)

    def save_buffer(self):
        with open(self._replay_dir / 'episodic_replay_buffer.buf', 'wb') as f:
            pickle.dump(self._current_episode, f)
        np.save(self._replay_dir / 'num_transitions.npy', self._num_transitions)

    def load(self):
        try:
            self._replay_buffer = pickle.load(open(self._replay_dir / 'episodic_replay_buffer.buf'), 'rb')
            self._num_transitions = np.save(self._replay_dir / 'num_transitions.npy').tolist()
        except:
            print('no replay buffer to be restored')

class UnionBuffer:

    def __init__(self,
                 replay_buffer_1, replay_buffer_2,
                 batch_size=None,
                 filter_transitions=True,
                 with_replacement=True,):

        self._rb1 = replay_buffer_1
        self._rb2 = replay_buffer_2
        self._batch_size = batch_size
        self._filter_transitions = filter_transitions
        self._with_replacement = with_replacement

    def __iter__(self):
        return self

    def __next__(self):
        batch_size_1 = np.random.binomial(self._batch_size,
                                          len(self._rb1) / (len(self._rb1) + len(self._rb2)))
        batch_size_2 = self._batch_size - batch_size_1

        # batch size being 0 should be handled correctly
        batch_1 = self._rb1.next(batch_size=batch_size_1,
                                 filter_transitions=self._filter_transitions,
                                 with_replacement=self._with_replacement)
        batch_2 = self._rb2.next(batch_size=batch_size_2,
                                 filter_transitions=self._filter_transitions,
                                 with_replacement=self._with_replacement)
        batch = ()
        for el1, el2 in zip(batch_1, batch_2):
            batch += (np.concatenate([el1, el2], axis=0),)

        return batch