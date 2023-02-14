import random
import re
import time

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)

def weight_init(m):
    if isinstance(m, nn.Linear):
        # TODO: Changed initialization to xavier_uniform_
        nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def split_demos(demo, num_val_trajectories=1):
    # assume the demo is structured according to EARL specification
    final_val_demo_idx = np.where(demo['terminals'])[0][num_val_trajectories - 1] + 1

    train_demos, val_demos = {}, {}
    for key in ['observations', 'actions', 'rewards', 'next_observations', 'terminals']:
        val_demos[key] = demo[key][:final_val_demo_idx]
        train_demos[key] =  demo[key][final_val_demo_idx:]

    return train_demos, val_demos

def split_demos_franka_shuf(demo, split_val=0.10):
    train_demos, val_demos = {}, {}
    len_demos = demo['actions'].shape[0]

    # randomly shuffle indices
    randperm = np.random.permutation(len_demos)
    split_idx = int(split_val*len_demos)
    train_indx = randperm[split_idx:]
    val_indx = randperm[:split_idx]

    for key in demo.keys():
        val_demos[key] = demo[key][val_indx]
        train_demos[key] =  demo[key][train_indx]

    # to ensure uniform sampling, put all state-action pairs in one episode
    train_demos['dones'] = np.zeros(train_demos['dones'].shape)
    train_demos['dones'][-1] = 1
    return train_demos, val_demos            

def refactor_demos_franka(demos, obs_tranform):

    def _get_obs_at_idx(idx, demos, obs_transform, next=False):
        if next:
            prefix = 'next_'
        else:
            prefix = ''

        return obs_transform({
            'hand_img_obs': demos[prefix + 'hand_img_obs'][idx],
            'third_person_img_obs': demos[prefix + 'third_person_img_obs'][idx],
            'lowdim_qpos': demos[prefix + 'lowdim_qpos'][idx],
            'lowdim_ee': demos[prefix + 'lowdim_ee'][idx],
        })

    new_demos = dict()
    for key in ['actions', 'dones', 'rewards']:
        new_demos[key] = demos[key].copy()

    new_demos['observations'] = np.array([
        _get_obs_at_idx(idx, demos, obs_tranform) for idx in range(demos['actions'].shape[0])
    ])
    new_demos['next_observations'] = np.array([
        _get_obs_at_idx(idx, demos, obs_tranform, next=True) for idx in range(demos['actions'].shape[0])
    ])
    return new_demos

class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = None if every == 'None' else every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
        match = re.match(r'linear_offset\((.+),(.+),(.+),(.+)\)', schdl)
        if match:
            start_val, end_val, start_step, end_step= [
                float(g) for g in match.groups()
            ]
            if step <= start_step:
                mix = 0.0
            else:
                mix = np.clip((step - start_step) / (end_step - start_step), 0.0, 1.0)
            return (1.0 - mix) * start_val + mix * end_val
    raise NotImplementedError(schdl)

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))
        
class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

# ROT related utilities
import ot

def optimal_transport_plan(X,
                           Y,
                           cost_matrix,
                           method='sinkhorn_gpu',
                           niter=500,
                           epsilon=0.01):
    X_pot = np.ones(X.shape[0]) * (1 / X.shape[0])
    Y_pot = np.ones(Y.shape[0]) * (1 / Y.shape[0])
    c_m = cost_matrix.data.detach().cpu().numpy()
    transport_plan = ot.sinkhorn(X_pot, Y_pot, c_m, epsilon, numItermax=niter)
    transport_plan = torch.from_numpy(transport_plan).to(X.device)
    transport_plan.requires_grad = False
    return transport_plan


def cosine_distance(x, y):
    C = torch.mm(x, y.T)
    x_norm = torch.norm(x, p=2, dim=1)
    y_norm = torch.norm(y, p=2, dim=1)
    x_n = x_norm.unsqueeze(1)
    y_n = y_norm.unsqueeze(1)
    norms = torch.mm(x_n, y_n.T)
    C = (1 - C / norms)
    return C


def euclidean_distance(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sqrt(torch.sum((torch.abs(x_col - y_lin)) ** 2, 2))
    return c
