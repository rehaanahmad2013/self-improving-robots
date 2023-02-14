import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ColorJitter
import utils

from torch.nn.utils.parametrizations import spectral_norm

class RandomShiftsAug(nn.Module):
    def __init__(self, pad, color_jitter=False):
        super().__init__()
        self.pad = pad
        self.color_jitter = color_jitter
        if self.color_jitter:
            self.cj = ColorJitter((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        if self.color_jitter:
            # color jitter only works on 3 channels at a time
            if c > 3:
                # split along channels to get each image
                # for frame stacking / multiple views
                split_imgs = torch.split(x, 3, dim=1)
                cj_imgs = [self.cj(img) for img in split_imgs]
                x = torch.cat(cj_imgs, dim=1)
            else:
                x = self.cj(x)
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        if len(obs.shape) == 3:
            return h.reshape(-1,)
        else:
            return h.contiguous().view(h.shape[0], -1)

class DiscrimVisionAction(nn.Module):
    def __init__(self, obs_shape, action_dim, feature_dim, repr_dim=39200,
                 create_inp_encoder=True, use_trunk=True, hidden_dim=256, use_spectral_norm=False):
        super().__init__()
        assert len(obs_shape) == 3
        self.repr_dim = repr_dim
        self.create_inp_encoder = create_inp_encoder
        self.use_trunk = use_trunk

        # assume that user will provided an encoded observation if false
        if self.create_inp_encoder:
            if use_spectral_norm:
                self.convnet = nn.Sequential(spectral_norm(nn.Conv2d(obs_shape[0], 32, 3, stride=2)),
                                                nn.ReLU(), spectral_norm(nn.Conv2d(32, 32, 3, stride=2)),
                                                nn.ReLU())
            else:
                self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                            nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2),
                                            nn.ReLU())
            # output from the smaller net   
            self.repr_dim = 12800

        if use_trunk:
            if use_spectral_norm:
                self.trunk = nn.Sequential(spectral_norm(nn.Linear(self.repr_dim, feature_dim)),
                                           nn.LayerNorm(feature_dim),
                                           nn.Tanh())
            else:
                self.trunk = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                                           nn.LayerNorm(feature_dim),
                                           nn.Tanh())
            out_dim = feature_dim
        else:
            out_dim = self.repr_dim

        if use_spectral_norm:
            self.linlayer = nn.Sequential(spectral_norm(nn.Linear(out_dim + action_dim, hidden_dim)),
                                          nn.ReLU(), spectral_norm(nn.Linear(hidden_dim, 1)),)
        else:
            self.linlayer = nn.Sequential(nn.Linear(out_dim + action_dim, hidden_dim), 
                                          nn.ReLU(), nn.Linear(hidden_dim, 1),)
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        # encode the observation
        if self.create_inp_encoder:
            obs = obs / 255.0 - 0.5
            h = self.convnet(obs)
            h = h.contiguous().view(h.shape[0], -1)
        else:
            h = obs

        if self.use_trunk:
            return self.linlayer(torch.cat([self.trunk(h), action], axis=1))
        else:
            return self.linlayer(torch.cat([h, action], axis=1))

    def encode(self, obs):
        if self.create_inp_encoder:
            obs = obs / 255.0 - 0.5
            h = self.convnet(obs)
            h = h.contiguous().view(h.shape[0], -1)
            return h
        else:
            print('use shared encoder to encode images')
            exit()

    def trunk_pass(self, encoded_obs):
        if self.use_trunk:
            return self.trunk(encoded_obs)
        else:
            print('trunk disabled')
            exit()
    
    def final_out(self, obs_action):
        return self.linlayer(obs_action)

class DiscrimVisionActionFranka(nn.Module):
    def __init__(self, obs_shape, action_dim, feature_dim, repr_dim=39200,
                 create_inp_encoder=True, hidden_dim=256):
        super().__init__()
        assert len(obs_shape) == 3
        self.repr_dim = repr_dim
        self.create_inp_encoder = create_inp_encoder

        # assume that user will provided an encoded observation if false
        if self.create_inp_encoder:
            self.convnet = nn.Sequential(spectral_norm(nn.Conv2d(obs_shape[0], 32, 3, stride=2)),
                                         nn.ReLU(), spectral_norm(nn.Conv2d(32, 32, 3, stride=2)),
                                         nn.ReLU())

            # output from the smaller net
            self.repr_dim = 12800

        self.trunk = nn.Sequential(spectral_norm(nn.Linear(self.repr_dim, feature_dim)),
                                   nn.LayerNorm(feature_dim),
                                   nn.Tanh())
        # state dim is 4, [x, y, z, gripper] of end effector
        out_dim = feature_dim + 4

        self.linlayer = nn.Sequential(spectral_norm(nn.Linear(out_dim + action_dim, hidden_dim)),
                                      nn.ReLU(), spectral_norm(nn.Linear(hidden_dim, 1)),)
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        # encode the observation
        if self.create_inp_encoder:
            obs_img = obs['imgs']
            obs_img = obs_img / 255.0 - 0.5
            h = self.convnet(obs_img)
            h = h.contiguous().view(h.shape[0], -1)
            return self.linlayer(torch.cat([self.trunk(h), obs['state'], action], axis=1))
        else:
            # assume obs = [trunk(img), state]
            return self.linlayer(torch.cat([obs, action], axis=1))

    def encode(self, obs):
        if self.create_inp_encoder:
            obs_img = obs['imgs'] / 255.0 - 0.5
            h = self.convnet(obs_img)
            return h.contiguous().view(h.shape[0], -1)
        else:
            print('use shared encoder to encode images')
            exit()

    def trunk_pass(self, encoded_obs):
        return self.trunk(encoded_obs)
    
    def final_out(self, obs_state_action):
        return self.linlayer(obs_state_action)

class DiscrimVision(nn.Module):
    def __init__(self, obs_shape, feature_dim, repr_dim=39200,
                 create_inp_encoder=True, use_trunk=True, hidden_dim=256, use_spectral_norm=False):
        super().__init__()
        assert len(obs_shape) == 3
        self.repr_dim = repr_dim
        self.hidden_dim = hidden_dim
        self.create_inp_encoder = create_inp_encoder
        self.use_trunk = use_trunk
        self.use_spectral_norm = use_spectral_norm

        # assume that user will provided an encoded observation if false
        if self.create_inp_encoder:
            if self.use_spectral_norm:
                self.convnet = nn.Sequential(spectral_norm(nn.Conv2d(obs_shape[0], 32, 3, stride=2)),
                                             nn.ReLU(), spectral_norm(nn.Conv2d(32, 32, 3, stride=2)),
                                             nn.ReLU())
            else:
                self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                            nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2),
                                            nn.ReLU())
            # output from the smaller net   
            self.repr_dim = 12800

        if use_trunk:
            if self.use_spectral_norm:
                self.trunk = nn.Sequential(spectral_norm(nn.Linear(self.repr_dim, feature_dim)),
                                           nn.LayerNorm(feature_dim),
                                           nn.Tanh())
            else:
                self.trunk = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                                           nn.LayerNorm(feature_dim),
                                           nn.Tanh())
            self.out_dim = feature_dim
        else:
            self.out_dim = self.repr_dim

        if self.use_spectral_norm:
            self.linlayer = nn.Sequential(spectral_norm(nn.Linear(self.out_dim, self.hidden_dim)),
                                          nn.ReLU(), spectral_norm(nn.Linear(self.hidden_dim, 1)),)
        else:
            self.linlayer = nn.Sequential(nn.Linear(self.out_dim, self.hidden_dim),
                                          nn.ReLU(), nn.Linear(self.hidden_dim, 1),)
        self.apply(utils.weight_init)

    def forward(self, obs):
        # encode the observation
        if self.create_inp_encoder:
            obs = obs / 255.0 - 0.5
            h = self.convnet(obs)
            h = h.contiguous().view(h.shape[0], -1)
        else:
            h = obs

        return self.linlayer(self.trunk(h) if self.use_trunk else h)

    def encode(self, obs):
        if self.create_inp_encoder:
            obs = obs / 255.0 - 0.5
            h = self.convnet(obs)
            h = h.contiguous().view(h.shape[0], -1)
            return h
        else:
            print('use shared encoder to encode images')
            exit()

    def trunk_pass(self, encoded_obs):
        if self.use_trunk:
            return self.trunk(encoded_obs)
        else:
            print('trunk disabled')
            exit()
    
    def final_out(self, obs_action):
        return self.linlayer(obs_action)

class DiscrimVisionFranka(nn.Module):
    def __init__(self, obs_shape, feature_dim, repr_dim=39200,
                 create_inp_encoder=True, use_trunk=True, hidden_dim=256,
                 state_dim=0, ignore_view=None,):
        '''
        state_dim: 0 if no state, 4 if state is [x, y, z, gripper], 8 if joint state.
        NOTE: 0 allows for state to be ignored.

        ignore_view: [None, 'first', 'third']. If None, then all views are used. If 'first', 
        then only the third person view is used. If 'third', then only the first person view is used.
        '''
        super().__init__()
        assert len(obs_shape) == 3
        self.repr_dim = repr_dim
        self.create_inp_encoder = create_inp_encoder
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.ignore_view = ignore_view

        # assume that user will provided an encoded observation if false
        if self.create_inp_encoder:
            if self.ignore_view in ['first', 'third']:
                obs_shape[0] = 3
            self.convnet = nn.Sequential(spectral_norm(nn.Conv2d(obs_shape[0], 32, 3, stride=2)),
                                         nn.ReLU(), spectral_norm(nn.Conv2d(32, 32, 3, stride=2)),
                                         nn.ReLU())

            # output from the smaller net for 100x100 images
            self.repr_dim = 18432

        self.trunk = nn.Sequential(spectral_norm(nn.Linear(self.repr_dim, self.feature_dim)),
                                   nn.LayerNorm(self.feature_dim),
                                   nn.Tanh())
        # add additional state information before final layers (if any)
        self.out_dim = self.feature_dim + self.state_dim
        self.linlayer = nn.Sequential(spectral_norm(nn.Linear(self.out_dim, hidden_dim)),
                                      nn.ReLU(), spectral_norm(nn.Linear(hidden_dim, 1)),)
        self.apply(utils.weight_init)

    def _transform_image(self, obs):
        if self.ignore_view == 'first':
            obs = obs[:, 3:] if len(obs.shape) == 4 else obs[3:]
        elif self.ignore_view == 'third':
            obs = obs[:, :3] if len(obs.shape) == 4 else obs[:3]
        return obs

    def forward(self, obs):
        # encode the observation
        if self.create_inp_encoder:
            obs_img = self._transform_image(obs['images'])
            obs_img = obs_img / 255.0 - 0.5
            h = self.convnet(obs_img)
            h = h.contiguous().view(h.shape[0], -1)
            obs = torch.cat([self.trunk(h), obs['state']], axis=1)

        # assume obs = [trunk(img), state]
        return self.final_out(obs)

    def encode(self, obs):
        if self.create_inp_encoder:
            obs_img = self._transform_image(obs) / 255.0 - 0.5
            h = self.convnet(obs_img)
            return h.contiguous().view(h.shape[0], -1)
        else:
            print('use shared encoder to encode images')
            exit()

    def trunk_pass(self, encoded_obs):
        return self.trunk(encoded_obs)

    def final_out(self, obs_state):
        '''Discriminator may still receive obs_state = [trunk(img), state].
        To ensure this, remove the last few dimensions. For example,
            if state_dim = 4, feature_dim = 50, out_dim = 54. We end up using the obs_state.
            if state_dim = 0, feature_dim = 50, out_dim = 50. trunk_pass will get obs_state of shape [batch, 54 or 57].
        Using out_dim, ensures that we get the correct shape.
        '''
        obs_state = obs_state[:, :self.out_dim]
        return self.linlayer(obs_state)

class LatentDiscriminator(DiscrimVision):
    def __init__(self, *args, num_outs, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outs = num_outs
        if self.use_spectral_norm:
            self.linlayer = nn.Sequential(spectral_norm(nn.Linear(self.out_dim, self.hidden_dim)),
                                          nn.ReLU(), spectral_norm(nn.Linear(self.hidden_dim, self.num_outs)),)
        else:
            self.linlayer = nn.Sequential(nn.Linear(self.out_dim, self.hidden_dim),
                                          nn.ReLU(), nn.Linear(self.hidden_dim, self.num_outs),)
        self.apply(utils.weight_init)

class DDPGActor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, franka=0):
        '''
        franka: extra dimensions for the franka state, concatenated after embedding the image
        '''
        super().__init__()
        action_dim = action_shape[0]
        self.franka = franka

        # convert image/state to a normalized vector 
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
								   nn.LayerNorm(feature_dim),
                                   nn.Tanh())

        # if using Franka, also concat state [x, y, z, gripper]
        policy_dim = feature_dim + self.franka
        self.policy = nn.Sequential( # policy layers
                                    nn.Linear(policy_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))
        self.apply(utils.weight_init)


    def forward(self, obs, std=None):
        if not self.franka:
            assert std != None
            h = self.trunk(obs)
            mu = self.policy(h)

            mu = torch.tanh(mu)
            std = torch.ones_like(mu) * std
            dist = utils.TruncatedNormal(mu, std)

            return dist
    
    def trunk_pass(self, obs_img):
        return self.trunk(obs_img)
    
    def final_out(self, obs, std=None):
        assert std != None
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)

        return dist

class SACActor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, log_std_bounds, franka=0):
        '''
        franka(uint): extra dimensions concatenated after trunking the image.
        '''
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.franka = franka
        action_dim = action_shape[0] * 2

        self.policy_trunk = nn.Sequential(# convert image/state to a normalized vector 
                                          nn.Linear(repr_dim, feature_dim),
                                          nn.LayerNorm(feature_dim),
                                          nn.Tanh())

        # if using franka, also concat state [x, y, z, gripper]
        policy_dim = feature_dim + self.franka
        self.policy_layer = nn.Sequential(# policy layers
                                          nn.Linear(policy_dim, hidden_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(hidden_dim, hidden_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(hidden_dim, action_dim))
        self.apply(utils.weight_init)

    def forward(self, obs):
        if not self.franka:
            return self.final_out(self.trunk_pass(obs))
        else:
            # assumes we are passed the concat(image_embedding, state)
            obs_h, obs_state = obs[:, :-self.franka], obs[:, -self.franka:]
            obs_trunk = self.trunk_pass(obs_h)
            obs = torch.cat([obs_trunk, obs_state], axis=-1)
            return self.final_out(obs)

    def trunk_pass(self, obs_img):
        return self.policy_trunk(obs_img)
    
    def final_out(self, obs):
        mu, log_std = self.policy_layer(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        # TODO: switched to simple clipping instead of going the tanh / rescaling route
        log_std_min, log_std_max = self.log_std_bounds
        # log_std = torch.tanh(log_std)
        # log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        log_std = torch.clip(log_std, log_std_min, log_std_max)
        std_pred = log_std.exp()

        return utils.SquashedNormal(mu, std_pred)

class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, franka=0):
        '''
        franka: extra dimensions for concatenation after trunking the image.
        '''
        super().__init__()
        self.franka = franka
        self.action_dim = action_shape[0] + self.franka

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim),
                                   nn.Tanh())
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + self.action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + self.action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)
        
    def forward(self, obs, action):
        ''' if self.franka > 0, assumes it is passed as concat(addn_state, action)'''
        h_action = torch.cat([self.trunk(obs), action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

class Qnet(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, franka=0):
        super().__init__()
        self.franka = franka
        self.action_dim = action_shape[0] + self.franka
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim),
                                   nn.Tanh())
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + self.action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))
        
        self.apply(utils.weight_init)
        
    def forward(self, obsaction):
        ''' if self.franka > 0, assumes it is passed as concat(addn_state, action)'''
        h_action = torch.cat([self.trunk(obsaction[:, :-self.action_dim]), # trunk the observation
                              obsaction[:, -self.action_dim:]], dim=-1)
        q1 = self.Q1(h_action)
        return q1