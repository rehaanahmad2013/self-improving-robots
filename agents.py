import copy
import math
import numpy as np
from functorch import combine_state_for_ensemble, vmap
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from networks import *
from utils import optimal_transport_plan, cosine_distance, euclidean_distance

class SACAgent:
	def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
				 hidden_dim, critic_target_tau, reward_scale_factor,    
				 use_tb, from_vision, bc_reg_lambda=0.0, repr_dim=None):

		self.obs_shape = obs_shape
		self.action_shape = action_shape
		self.lr = lr
		self.feature_dim = feature_dim
		self.hidden_dim = hidden_dim
		self.device = device
		self.critic_target_tau = critic_target_tau
		self.reward_scale_factor = reward_scale_factor
		self.use_tb = use_tb
		self.from_vision = from_vision

		# Changed log_std_bounds from [-10, 2] -> [-20, 2]
		self.log_std_bounds = [-20, 2]
		# Changed self.init_temperature to 1.0
		self.init_temperature = 1.0
		self.bc_reg_lambda = bc_reg_lambda
		self.repr_dim = repr_dim

		# models
		if self.from_vision:
			self.encoder = Encoder(obs_shape).to(device)
			# overwrite hard-coded representation dim for convnet
			self.encoder.repr_dim = repr_dim if repr_dim else self.encoder.repr_dim
			self.model_repr_dim = self.encoder.repr_dim
		else:
			self.model_repr_dim = self.obs_shape[0]

		self.actor = SACActor(self.model_repr_dim, action_shape, feature_dim,
							  hidden_dim, self.log_std_bounds).to(device)
		self.critic = Critic(self.model_repr_dim, action_shape, feature_dim,
							 hidden_dim).to(device)
		self.critic_target = Critic(self.model_repr_dim, action_shape,
									feature_dim, hidden_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(device)
		self.log_alpha.requires_grad = True
		self.target_entropy = -action_shape[0] / 2.0
		self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

		# optimizers
		if self.from_vision:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
			# data augmentation
			self.aug = RandomShiftsAug(pad=4)

		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.training = True
		if self.from_vision:
			self.encoder.train()
		self.actor.train()
		self.critic.train()
		self.critic_target.train()

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def train(self, training=True):
		self.training = training
		if self.from_vision:
			self.encoder.train(training)
		self.actor.train(training)
		self.critic.train(training)

	def act(self, obs, uniform_action=False, eval_mode=False):
		obs = torch.as_tensor(obs, device=self.device)
		if self.from_vision:
			obs = self.encoder(obs.unsqueeze(0))[0]

		dist = self.actor(obs)
		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample()

		if uniform_action:
			action.uniform_(-1.0, 1.0)

		return action.cpu().numpy()

	def update_critic(self, obs, action, reward, discount, next_obs, step, not_done=None):
		metrics = dict()

		with torch.no_grad():
			dist = self.actor(next_obs)
			next_action = dist.rsample()
			log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

			target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
			target_V = torch.min(target_Q1, target_Q2)
			target_V -= self.alpha.detach() * log_prob
			# TODO: figure out whether we want the not_done at the end or not
			target_Q = self.reward_scale_factor * reward + \
							(discount * target_V * not_done.unsqueeze(1))

		Q1, Q2 = self.critic(obs, action)
		# scaled the loss by 0.5, might have some effect initially
		critic_loss = 0.5 * (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))

		if self.use_tb:
			metrics['critic_target_q'] = target_Q.mean().item()
			metrics['critic_q1'] = Q1.mean().item()
			metrics['critic_q2'] = Q2.mean().item()
			metrics['critic_loss'] = critic_loss.item()

		# optimize encoder and critic
		if self.from_vision:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_opt.step()
		if self.from_vision:
			self.encoder_opt.step()

		return metrics

	def update_actor(self, obs, step, bc_reg=0.0, obs_d=None, action_d=None):
		metrics = dict()

		dist = self.actor(obs)
		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		Q1, Q2 = self.critic(obs, action)
		# TODO: shouldn't we set require grad to false here / detach?
		Q = torch.min(Q1, Q2)

		actor_loss = -Q + (self.alpha.detach() * log_prob)
		actor_loss = actor_loss.mean()

		if bc_reg > 0.0:
			dist_demo = self.actor(obs_d)
			log_prob_demo = dist_demo.log_prob(action_d).sum(-1, keepdim=True)
			actor_loss = (1. - bc_reg) * actor_loss - bc_reg * log_prob_demo.mean()

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()

		return metrics

	def update_alpha(self, obs, step):
		metrics = dict()

		dist = self.actor(obs)
		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		self.log_alpha_optimizer.zero_grad()
		alpha_loss = (self.alpha *
					(-log_prob - self.target_entropy).detach()).mean()
		alpha_loss.backward()
		self.log_alpha_optimizer.step()

		if self.use_tb:
			metrics['actor_logprob'] = log_prob.mean().item()
			metrics['alpha_loss'] = alpha_loss
			metrics['alpha_value'] = self.alpha

		return metrics

	def transition_tuple(self, replay_iter, demo_iter=None, oversample_count=None, online_buf_len=None, offline_buf_len=None):
		batch = utils.to_torch(next(replay_iter), self.device)
		batch_size = int(batch[0].shape[0])
		if demo_iter is not None:
			if oversample_count == 0:
				m = torch.distributions.binomial.Binomial(batch_size, torch.tensor([offline_buf_len/(offline_buf_len+online_buf_len)]))
				num_offline = int(m.sample().numpy()[0])
				num_online = batch_size - num_offline
			else:			
				num_offline = oversample_count
				num_online = batch[0].shape[0] - num_offline

			batch_d = utils.to_torch(next(demo_iter), self.device)

			new_batch = ()
			for x, y in zip(batch, batch_d):
				new_batch += (torch.cat((x[:num_online], y[:num_offline]), axis=0),)
			return new_batch

		return batch

	def update(self, trans_tuple_fn, step, trans_tuple_demo=None):
		metrics = dict()

		obs, action, reward, discount, next_obs, step_type, next_step_type = trans_tuple_fn()

		not_done = next_step_type.clone()
		not_done[not_done < 2] = 1
		not_done[not_done == 2] = 0

		# augment
		if self.from_vision:
			obs = self.aug(obs.float())
			next_obs = self.aug(next_obs.float())
			# encode
			obs = self.encoder(obs)
			with torch.no_grad():
				next_obs = self.encoder(next_obs)

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()

		# update critic
		metrics.update(
			self.update_critic(obs, action, reward, discount, next_obs, step, not_done))

		# update actor
		if trans_tuple_demo is not None:
			obs_d, action_d, reward_d, discount_d, next_obs_d, step_type_d, next_step_type_d = trans_tuple_demo
			obs_d = self.aug(obs_d.float())
			# encode
			obs_d = self.encoder(obs_d)
			bc_reg = utils.schedule(self.bc_reg_lambda, step)
			metrics.update(self.update_actor(obs.detach(),
											 step,
											 bc_reg=bc_reg,
											 obs_d=obs_d.detach(),
											 action_d=action_d.detach()))
		else:
			metrics.update(self.update_actor(obs.detach(), step))

		# update alpha
		metrics.update(self.update_alpha(obs.detach(), step))

		# update critic target
		utils.soft_update_params(self.critic, self.critic_target,
								 self.critic_target_tau)

		return metrics

class DDPGAgent:
	def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
				 hidden_dim, critic_target_tau, num_expl_steps,
				 stddev_schedule, stddev_clip, use_tb, from_vision):
		self.obs_shape = obs_shape
		self.action_shape = action_shape
		self.lr = lr
		self.feature_dim = feature_dim
		self.hidden_dim = hidden_dim
		self.device = device
		self.critic_target_tau = critic_target_tau
		self.use_tb = use_tb
		self.num_expl_steps = num_expl_steps
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.from_vision = from_vision

		# models
		if self.from_vision:
			self.encoder = Encoder(obs_shape).to(device)
			model_repr_dim = self.encoder.repr_dim
		else:
			model_repr_dim = obs_shape[0]

		self.actor = DDPGActor(model_repr_dim, action_shape, feature_dim,
							   hidden_dim).to(device)
		
		self.critic = Critic(model_repr_dim, action_shape, feature_dim,
							 hidden_dim).to(device)
		self.critic_target = Critic(model_repr_dim, action_shape,
									feature_dim, hidden_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		# optimizers
		if self.from_vision:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
		   # data augmentation
			self.aug = RandomShiftsAug(pad=4)

		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
 
		self.train()
		self.critic_target.train()

	def train(self, training=True):
		self.training = training
		if self.from_vision:
			self.encoder.train(training)
		self.actor.train(training)
		self.critic.train(training)

	def act(self, obs, uniform_action=False, eval_mode=False):
		obs = torch.as_tensor(obs, device=self.device)
		if self.from_vision:
			obs = self.encoder(obs.unsqueeze(0))

		stddev = utils.schedule(self.stddev_schedule, step)
		dist = self.actor(obs, stddev)

		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample()
		
		if uniform_action:
			action.uniform_(-1.0, 1.0)
			
		return action.cpu().numpy()

	def update_critic(self, obs, action, reward, discount, next_obs, step):
		metrics = dict()
		print("is this update critic fn called?")
		exit()
		with torch.no_grad():
			stddev = utils.schedule(self.stddev_schedule, step)
			dist = self.actor(next_obs, stddev)
			next_action = dist.sample(clip=self.stddev_clip)
			target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
			target_V = torch.min(target_Q1, target_Q2)
			target_Q = reward + discount * target_V


		Q1, Q2 = self.critic(obs, action)
		critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

		if self.use_tb:
			metrics['critic_target_q'] = target_Q.mean().item()
			metrics['critic_q1'] = Q1.mean().item()
			metrics['critic_q2'] = Q2.mean().item()
			metrics['critic_loss'] = critic_loss.item()

		# optimize encoder and critic
		if self.from_vision:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_opt.step()
	
		if self.from_vision:
			self.encoder_opt.step()

		return metrics

	def update_actor(self, obs, step):
		metrics = dict()
		print("is this update actor fn called?")
		exit()
		stddev = utils.schedule(self.stddev_schedule, step)
		dist = self.actor(obs, stddev)
		action = dist.sample(clip=self.stddev_clip)
		
		Q1, Q2 = self.critic(obs, action)
		Q = torch.min(Q1, Q2)

		actor_loss = -Q
		actor_loss = actor_loss.mean()

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['actor_logprob'] = log_prob.mean().item()

		return metrics

	def transition_tuple(self, replay_iter):
		batch = next(replay_iter)
		obs, action, reward, discount, next_obs, step_type, next_step_type = utils.to_torch(batch, self.device)

		return (obs, action, reward, discount, next_obs, step_type, next_step_type)

	def update(self, trans_tuple, step):
		metrics = dict()
		print("is this update fn called?")
		exit()
		obs, action, reward, discount, next_obs, step_type, next_step_type = trans_tuple

		# augment
		if self.from_vision:
			obs = self.aug(obs.float())
			next_obs = self.aug(next_obs.float())
			# encode
			obs = self.encoder(obs)
			with torch.no_grad():
				next_obs = self.encoder(next_obs)

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()

		# update critic
		metrics.update(
			self.update_critic(obs, action, reward, discount, next_obs, step))

		# update actor
		metrics.update(self.update_actor(obs.detach(), step))

		# update critic target
		utils.soft_update_params(self.critic, self.critic_target,
								 self.critic_target_tau)

		return metrics

class REDQAgent(SACAgent):
	def __init__(self, *agent_args,
				 num_Q=10,
				 utd_ratio=20,
				 num_min=2,
				 train_alpha=False,
				 target_entropy='default',
				 **agent_kwargs):

		super().__init__(*agent_args, **agent_kwargs)
		self.critic_list, self.critic_target_list = [], []
		self.num_Q = num_Q
		self.utd_ratio = utd_ratio
		self.num_min = num_min

		# delete critic class from baseclass since we maintain separate q-nets directly
		delattr(self, 'critic')
		delattr(self, 'critic_target')
		delattr(self, 'critic_opt')

		# init ensemble of Q functions
		for _ in range(num_Q):
			new_critic = Qnet(self.model_repr_dim, self.action_shape, self.feature_dim,
							  self.hidden_dim).to(self.device)
			self.critic_list.append(new_critic)
			new_critic_target = Qnet(self.model_repr_dim, self.action_shape, self.feature_dim,
									 self.hidden_dim).to(self.device)
			# initialize target critic network
			new_critic_target.load_state_dict(new_critic.state_dict())
			self.critic_target_list.append(new_critic_target)

		# setting up feedforward pass via vmap
		self.Qens, self.Qens_params, self.Qens_buffers = combine_state_for_ensemble(self.critic_list)
		[p.requires_grad_() for p in self.Qens_params];
		self.Qtens, self.Qtens_params, self.Qtens_buffers = combine_state_for_ensemble(self.critic_target_list)
		self.critic_opt = torch.optim.Adam(self.Qens_params, lr=self.lr)

		# allows entropy to be turned off
		self.train_alpha = train_alpha
		if target_entropy == 'auto_large':
			self.target_entropy = self.action_shape[0] / 2.0
		elif isinstance(target_entropy, float):
			self.target_entropy = target_entropy
		self.zero_alpha = torch.tensor(0.).to(self.device)

		self.train()

	@property
	def alpha(self):
		if self.train_alpha:
			return self.log_alpha.exp()
		else:
			return self.zero_alpha

	def update_alpha(self, *args):
		if self.train_alpha:
			return super().update_alpha(*args)
		else:
			return dict()

	def train(self, training=True):
		self.training = training
		if self.from_vision:
			self.encoder.train(training)
		self.actor.train(training)

	def ensemble_forward_pass(self, obs, action):
		 return vmap(self.Qens, in_dims=(0, 0, None))(self.Qens_params, self.Qens_buffers, torch.cat([obs, action], -1)).squeeze()

	def target_ensemble_forward_pass(self, obs, action):
		return vmap(self.Qtens, in_dims=(0, 0, None))(self.Qtens_params, self.Qtens_buffers, torch.cat([obs, action], -1)).squeeze()

	def redq_q_target_no_grad(self, next_obs, reward, discount, not_done):
		# Helper function for creating different instantiations of REDQ
		sample_idxs = np.random.choice(self.num_Q, self.num_min, replace=False)
		with torch.no_grad():
			dist = self.actor(next_obs)
			next_action = dist.rsample()
			log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

			# Q target is a min of a randomly sampled subset of Q values
			target_Q_pred = self.target_ensemble_forward_pass(next_obs, next_action)[sample_idxs].transpose(0, 1)
			target_V, _ = torch.min(target_Q_pred, dim=1, keepdim=True)
			target_V -= self.alpha.detach() * log_prob
			target_Q = self.reward_scale_factor * reward + \
							(discount * target_V * not_done.unsqueeze(1))

		return target_Q

	def update_critic(self, obs, action, reward, discount, next_obs, step, not_done=None):
		metrics = dict()

		target_Q = self.redq_q_target_no_grad(next_obs, reward, discount, not_done)
		target_Q = target_Q.transpose(0, 1).expand((self.num_Q, -1))

		Q_pred = self.ensemble_forward_pass(obs, action)
		critic_loss_total = F.mse_loss(Q_pred, target_Q) * self.num_Q

		if self.use_tb:
			metrics['critic_target_q'] = target_Q.mean().item()
			for idx in range(self.num_Q):
				metrics[f'critic_q{idx+1}'] = Q_pred[idx].mean().item()
			metrics['critic_loss'] = critic_loss_total.item()

		# optimize encoder and critic ensemble
		if self.from_vision:
			self.encoder_opt.zero_grad(set_to_none=True)

		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss_total.backward()
		self.critic_opt.step()

		if self.from_vision:
			self.encoder_opt.step()

		return metrics

	def update_actor(self, obs, step, bc_reg=0.0, obs_d=None, action_d=None):
		'''
		obs: update using the critic on these samples
		(obs_d, action_d): update actor using BC regularization on these samples
		'''
		metrics = dict()

		dist = self.actor(obs)
		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		# minor speedup: turn off gradient collection for ensemble parameter
		[p.requires_grad_(False) for p in self.Qens_params];
		Q_pred = self.ensemble_forward_pass(obs, action).transpose(0, 1)
		# REDQ takes average over ensemble of Q values
		Q_mean = torch.mean(Q_pred, dim=1, keepdim=True)

		actor_loss = -Q_mean + (self.alpha.detach() * log_prob)
		actor_loss = actor_loss.mean()

		if bc_reg > 0.0:
			dist_demo = self.actor(obs_d)
			log_prob_demo = dist_demo.log_prob(action_d).sum(-1, keepdim=True)
			actor_loss = (1. - bc_reg) * actor_loss - bc_reg * log_prob_demo.mean()

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()

		# turn back gradients for ensemble parameters
		[p.requires_grad_(True) for p in self.Qens_params];
		self.actor_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			if bc_reg > 0.0:
				metrics['bc_log_prob'] = log_prob_demo.mean().item()
		return metrics

	def update(self, trans_tuple_fn, step, trans_tuple_demo=None):
		metrics = dict()
		for _ in range(self.utd_ratio):
			obs, action, reward, discount, next_obs, step_type, next_step_type = trans_tuple_fn()
			
			not_done = next_step_type.clone()
			not_done[not_done < 2] = 1
			not_done[not_done == 2] = 0

			# augment
			if self.from_vision:
				obs = self.aug(obs.float())
				next_obs = self.aug(next_obs.float())
				# encode
				obs = self.encoder(obs)
				with torch.no_grad():
					next_obs = self.encoder(next_obs)

			# update critic
			metrics.update(
				self.update_critic(obs, action, reward, discount, next_obs, step, not_done))

			# soft update target parameters
			for param, target_param in zip(self.Qens_params, self.Qtens_params):
				target_param.data.copy_(self.critic_target_tau * param.data + 
									   (1. - self.critic_target_tau) * target_param.data)

		# update actor
		if trans_tuple_demo is not None:
			obs_d, action_d, _, _, _, _, _ = trans_tuple_demo
			
			if self.from_vision:
				obs_d = self.aug(obs_d.float())
				# TODO: pass the gradients from BC loss to the encoder
				with torch.no_grad():
					obs_d = self.encoder(obs_d)

			if self.bc_reg_lambda == 'soft_qfilter':
				with torch.no_grad():
					dist_d = self.actor(obs_d)
					Q_p = torch.mean(self.ensemble_forward_pass(obs_d, dist_d.rsample()).transpose(0, 1), dim=1)
					Q_bc = torch.mean(self.ensemble_forward_pass(obs_d, action_d).transpose(0, 1), dim=1)
					mask = Q_bc > Q_p
					bc_reg = mask.float().mean()

			elif self.bc_reg_lambda == 'hard_qfilter':
				with torch.no_grad():
					dist_d = self.actor(obs_d)
					Q_p = torch.mean(self.ensemble_forward_pass(obs_d, dist_d.rsample()).transpose(0, 1), dim=1)
					Q_bc = torch.mean(self.ensemble_forward_pass(obs_d, action_d).transpose(0, 1), dim=1)
					mask = Q_bc > Q_p
					# hard_qfilter: remove the state-action pairs that have lower Q-vals
					obs_d, action_d = obs_d[mask], action_d[mask]
					bc_reg = mask.float().mean()
			else:
				# either a fixed float or linear decay
				bc_reg = utils.schedule(self.bc_reg_lambda, step)

			metrics.update(self.update_actor(obs.detach(),
											 step,
											 bc_reg=bc_reg,
											 obs_d=obs_d,
											 action_d=action_d))
		else:
			metrics.update(self.update_actor(obs.detach(), step))

		# update alpha
		metrics.update(self.update_alpha(obs.detach(), step))

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()
			if self.bc_reg_lambda != 0.0:
				metrics['bc_reg_coef'] = bc_reg

		return metrics

class DACAgent(REDQAgent):
	def __init__(self,
				 *agent_args,
				 discrim_hidden_size=128,
				 discrim_lr=3e-4,
				 mixup=True,
				 reward_type='logd',
				 share_encoder=False,
				 gaussian_noise_coef=0.,
				 use_trunk=True,
				 spectral_norm=True,
				 discrim_val_data=None,
				 **agent_kwargs):

		super(DACAgent, self).__init__(**agent_kwargs)
		self.discrim_hidden_size = discrim_hidden_size
		self.discrim_lr = discrim_lr
		self.mixup = mixup
		self.eps = 1e-10
		self.reward_type = reward_type
		self.share_encoder = share_encoder
		self.gaussian_noise_coef = gaussian_noise_coef
		self.use_trunk = use_trunk
		self.spectral_norm = spectral_norm
		self.discrim_val_data = discrim_val_data

		if self.from_vision:
			self.discriminator = DiscrimVisionAction(obs_shape=self.obs_shape,
													 action_dim=self.action_shape[0],
													 feature_dim=self.feature_dim,
													 hidden_dim=discrim_hidden_size,
													 create_inp_encoder=not bool(self.share_encoder),
													 use_spectral_norm=self.spectral_norm,
													 use_trunk=self.use_trunk,).to(self.device)
		else:
			self.discriminator = nn.Sequential(nn.Linear(self.obs_shape[0] + self.action_shape[0],
														discrim_hidden_size),
											   nn.ReLU(inplace=True),
											   nn.Linear(discrim_hidden_size, discrim_hidden_size),
											   nn.ReLU(inplace=True),
											   nn.Linear(discrim_hidden_size, 1)).to(self.device)

		self.discrim_opt = torch.optim.Adam(self.discriminator.parameters(), lr=discrim_lr)

	def update_discriminator(self, pos_replay_iter, neg_replay_iter):
		metrics = dict()

		batch_pos = next(pos_replay_iter)
		obs_pos, action_pos, _, _, _, _, _ = utils.to_torch(batch_pos, self.device)
		num_pos = obs_pos.shape[0]

		batch_neg = next(neg_replay_iter)
		obs_neg, action_neg, _, _, _, _, _ = utils.to_torch(batch_neg, self.device)
		num_neg = obs_neg.shape[0]

		'''augment the images and encode + trunk images before mixup,
		the shared encoder has not seen mixed up images.'''
		if self.from_vision:
			# TODO: maybe remove augmentation from the goal image?
			obs_pos = self.aug(obs_pos.float())
			obs_neg = self.aug(obs_neg.float())

			# frozen shared encoder
			if self.share_encoder == 1:
				with torch.no_grad():
					obs_pos = self.encoder(obs_pos)
					obs_neg = self.encoder(obs_neg)
			# update shared encoder
			elif self.share_encoder == 2:
				obs_pos = self.encoder(obs_pos)
				obs_neg = self.encoder(obs_neg)
			# use and train discriminator's own encoder
			elif self.share_encoder == 0:
				obs_pos = self.discriminator.encode(obs_pos)
				obs_neg = self.discriminator.encode(obs_neg)

			# run mixup on low-dim / trunk representation
			# TODO: maybe do mixup before the trunk layer?
			if self.use_trunk:
				obs_pos = self.discriminator.trunk_pass(obs_pos)
				obs_neg = self.discriminator.trunk_pass(obs_neg)

		# TODO: add gaussian noise to just the actions independently?
		pos_input = torch.cat([obs_pos, action_pos], 1)
		neg_input = torch.cat([obs_neg, action_neg], 1)

		if self.mixup:
			alpha = 1.0
			beta_dist = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

			l = beta_dist.sample([num_pos + num_neg])
			mixup_coef = torch.reshape(l, (num_pos + num_neg, 1)).to(self.device)

			mixup_coef_lab = torch.reshape(l, (num_pos + num_neg, 1)).to(self.device)
			labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)

			disc_inputs = torch.cat((pos_input, neg_input), 0)
			ridxs = torch.randperm(num_pos + num_neg)
			perm_labels = labels[ridxs]
			perm_disc_inputs = disc_inputs[ridxs]

			images = disc_inputs * mixup_coef + perm_disc_inputs * (1 - mixup_coef)
			labels = labels * mixup_coef_lab + perm_labels * (1 - mixup_coef_lab)
		else:
			images = torch.cat((pos_input, neg_input), 0)
			labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)

		loss = torch.nn.BCELoss()
		m = nn.Sigmoid()

		if self.from_vision:
			# TODO: action should still obey the bounds after adding gaussian noise
			images = images + self.gaussian_noise_coef * torch.randn_like(images)
			output = m(self.discriminator.final_out(images))
			discrim_loss = loss(output, labels)
		else:
			output = m(self.discriminator(images))
			discrim_loss = loss(output, labels)

		if self.from_vision and self.share_encoder == 2:
			self.encoder_opt.zero_grad(set_to_none=True)

		self.discrim_opt.zero_grad(set_to_none=True)
		discrim_loss.backward()
		self.discrim_opt.step()

		if self.from_vision and self.share_encoder == 2:
			self.encoder_opt.step()

		if self.use_tb:
			metrics['discriminator_loss'] = discrim_loss.item()
			if not self.mixup:
				metrics['discriminator_acc'] = ((output > 0.5) == labels).type(torch.float).mean().item()
			if self.discrim_val_data is not None:
				with torch.no_grad():
					_, output = self.compute_reward(self.discrim_val_data['observations'], self.discrim_val_data['actions'], return_sig=True, evald=True)
					val_labels = torch.ones(self.discrim_val_data['observations'].shape[0], 1).to(self.device)
					metrics['val_loss'] = loss(output, val_labels).item()
					metrics['val_acc'] = ((output > 0.5) == val_labels).type(torch.float).mean().item()

		return metrics

	def compute_reward(self, obs, action, return_sig=False, evald=False):
		if self.from_vision:
			if evald and type(obs) is np.ndarray:
					obs = torch.from_numpy(obs).to(self.device)
					action = torch.from_numpy(action).to(self.device)
			if self.share_encoder:
				obs = self.encoder(obs)
			sig_term = torch.sigmoid(self.discriminator(obs, action))
		else:
			sig_term = torch.sigmoid(self.discriminator(torch.cat([obs, action], axis=1)))

		if self.reward_type == 'logd':
			actual_reward = torch.log(torch.minimum(sig_term + self.eps, torch.tensor(1.)))
		else:
			actual_reward = -torch.log(1 - sig_term + self.eps)
		
		if not return_sig:
			return actual_reward
		else:
			return actual_reward, sig_term

	def transition_tuple(self, replay_iter, demo_iter=None, oversample_count=None, gt_reward=False, online_buf_len=None, offline_buf_len=None):
		obs, action, reward, discount, next_obs, step_type, next_step_type = super().transition_tuple(replay_iter,
																									  demo_iter=demo_iter,
																									  oversample_count=oversample_count,
																									  online_buf_len=online_buf_len,
																									  offline_buf_len=offline_buf_len)
		if gt_reward:
			return (obs, action, reward, discount, next_obs, step_type, next_step_type)
		else:
			with torch.no_grad():
				GAIL_reward = self.compute_reward(obs, action)
			return (obs, action, GAIL_reward, discount, next_obs, step_type, next_step_type)


class VICEAgent(DACAgent):
	def __init__(self, *args,
				 discrim_hidden_size=128,
				 discrim_lr=3e-4,
				 mixup=True,
				 reward_type='logd',
				 share_encoder=False,
				 gaussian_noise_coef=0.,
				 use_trunk=True,
				 spectral_norm=True,
				 pos_dataset=None,
				 **kwargs):

		super(VICEAgent, self).__init__(*args, **kwargs)
		self.discrim_hidden_size = discrim_hidden_size
		self.discrim_lr = discrim_lr
		self.discrim_hidden_size = discrim_hidden_size
		self.mixup = mixup
		self.reward_type = reward_type
		self.share_encoder = share_encoder
		self.gaussian_noise_coef = gaussian_noise_coef
		self.use_trunk = use_trunk
		self.spectral_norm = spectral_norm
		self.pos_dataset = pos_dataset
		self.num_goals = self.pos_dataset.shape[0]
		self.eps = 1e-10

		if self.from_vision:
			self.discriminator = DiscrimVision(obs_shape=self.obs_shape,
											   feature_dim=self.feature_dim,
											   hidden_dim=self.discrim_hidden_size,
											   repr_dim=self.model_repr_dim,
											   create_inp_encoder=not bool(self.share_encoder),
											   use_spectral_norm=self.spectral_norm,
											   use_trunk=self.use_trunk,).to(self.device)
		else:
			self.discriminator = nn.Sequential(nn.Linear(self.obs_shape[0], discrim_hidden_size),
											   nn.ReLU(inplace=True),
											   nn.Linear(discrim_hidden_size, discrim_hidden_size),
											   nn.ReLU(inplace=True),
											   nn.Linear(discrim_hidden_size, 1)).to(self.device)

		self.discrim_opt = torch.optim.Adam(self.discriminator.parameters(), lr=discrim_lr)
		self.zero_alpha = torch.tensor(0.).to(self.device)

	def update_discriminator(self, neg_replay_iter):
		metrics = dict()

		batch_neg = next(neg_replay_iter)
		obs_neg = utils.to_torch(batch_neg, self.device)[0]
		num_pos = num_neg = obs_neg.shape[0]

		# shuffle the goal states then sample a minibatch
		if self.num_goals < num_neg:
			ridxs = torch.cat([torch.randperm(self.num_goals) for _ in range(math.ceil(num_neg / self.num_goals))])[:num_neg]
		else:
			ridxs = torch.randperm(self.num_goals)[:num_neg]
		obs_pos = self.pos_dataset[ridxs].to(self.device).type(torch.cuda.FloatTensor)

		'''augment the images and encode + trunk images before mixup,
		the shared encoder has not seen mixed up images.'''
		if self.from_vision:
			obs_pos = self.aug(obs_pos.float())
			obs_neg = self.aug(obs_neg.float())

			# frozen shared encoder
			if self.share_encoder == 1:
				with torch.no_grad():
					obs_pos = self.encoder(obs_pos)
					obs_neg = self.encoder(obs_neg)
			# update shared encoder
			elif self.share_encoder == 2:
				obs_pos = self.encoder(obs_pos)
				obs_neg = self.encoder(obs_neg)
			# use and train discriminator's own encoder
			elif self.share_encoder == 0:
				obs_pos = self.discriminator.encode(obs_pos)
				obs_neg = self.discriminator.encode(obs_neg)

			if self.use_trunk:
				obs_pos = self.discriminator.trunk_pass(obs_pos)
				obs_neg = self.discriminator.trunk_pass(obs_neg)

		pos_input = obs_pos
		neg_input = obs_neg

		# exit()
		if self.mixup:
			disc_inputs = torch.cat((pos_input, neg_input), 0)
			labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)

			# Beta(1,1) is Uniform[0,1]
			beta_dist = torch.distributions.beta.Beta(torch.tensor([1.0]), torch.tensor([1.0]))
			l = beta_dist.sample([num_pos + num_neg])
			mixup_coef = torch.reshape(l, (num_pos + num_neg, 1)).to(self.device)

			# permute the images and labels for mixing up
			ridxs = torch.randperm(num_pos + num_neg)
			perm_labels = labels[ridxs]
			perm_disc_inputs = disc_inputs[ridxs]

			# create mixed up inputs
			images = disc_inputs * mixup_coef + perm_disc_inputs * (1 - mixup_coef)
			labels = labels * mixup_coef + perm_labels * (1 - mixup_coef)
		else:
			images = torch.cat((pos_input, neg_input), 0)
			labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)

		loss = torch.nn.BCELoss()
		m = nn.Sigmoid()

		if self.from_vision:
			images = images + self.gaussian_noise_coef * torch.randn_like(images)
			output = m(self.discriminator.final_out(images))
			discrim_loss = loss(output, labels)
		else:
			output = m(self.discriminator(images))
			discrim_loss = loss(output, labels)

		if self.from_vision and self.share_encoder == 2:
			self.encoder_opt.zero_grad(set_to_none=True)

		self.discrim_opt.zero_grad(set_to_none=True)
		discrim_loss.backward()
		self.discrim_opt.step()

		if self.from_vision and self.share_encoder == 2:
			self.encoder_opt.step()

		if self.use_tb:
			metrics['discriminator_loss'] = discrim_loss.item()
			if not self.mixup:
				metrics['discriminator_acc'] = ((output > 0.5) == labels).type(torch.float).mean().item()
			if self.discrim_val_data is not None:
				with torch.no_grad():
					_, output = self.compute_reward(self.discrim_val_data['observations'], return_sig=True, evald=True)
					val_labels = torch.ones(self.discrim_val_data['observations'].shape[0], 1).to(self.device)
					metrics['val_loss'] = loss(output, val_labels).item()
					metrics['val_acc'] = ((output > 0.5) == val_labels).type(torch.float).mean().item()

		return metrics

	def compute_reward(self, obs, action=None, return_sig=False, evald=False):
		del action
		if self.from_vision:
			if evald and type(obs) is np.ndarray:
				obs = torch.from_numpy(obs).to(self.device)

			if self.share_encoder:
				obs = self.encoder(obs)

		sig_term = torch.sigmoid(self.discriminator(obs))
		if self.reward_type == 'logd':
			actual_reward = torch.log(torch.minimum(sig_term + self.eps, torch.tensor(1.)))
		else:
			actual_reward = -torch.log(1 - sig_term + self.eps)

		if not return_sig:
			return actual_reward
		else:
			return actual_reward, sig_term

	def transition_tuple(self, replay_iter, demo_iter=None, oversample_count=None, gt_reward=False, online_buf_len=None, offline_buf_len=None):
		obs, action, reward, discount, next_obs, step_type, next_step_type = super().transition_tuple(replay_iter,
																									  demo_iter=demo_iter,
																									  oversample_count=oversample_count,
																									  gt_reward=gt_reward,
																									  online_buf_len=online_buf_len,
																									  offline_buf_len=offline_buf_len)
		if gt_reward:
			return (obs, action, reward, discount, next_obs, step_type, next_step_type)
		else:
			with torch.no_grad():
				VICE_reward = self.compute_reward(next_obs, None)
			return (obs, action, VICE_reward, discount, next_obs, step_type, next_step_type)


class DACSACAgent(SACAgent):
	def __init__(self,
				 *agent_args,
				 discrim_hidden_size=128,
				 discrim_lr=3e-4,
				 mixup=True,
				 reward_type='logd',
				 share_encoder=False,
				 gaussian_noise_coef=0.,
				 use_trunk=True,
				 spectral_norm=True,
				 discrim_val_data=None,
				 **agent_kwargs):

		super(DACSACAgent, self).__init__(**agent_kwargs)
		self.discrim_hidden_size = discrim_hidden_size
		self.discrim_lr = discrim_lr
		self.mixup = mixup
		self.eps = 1e-10
		self.reward_type = reward_type
		self.share_encoder = share_encoder
		self.gaussian_noise_coef = gaussian_noise_coef
		self.use_trunk = use_trunk
		self.spectral_norm = spectral_norm
		self.discrim_val_data = discrim_val_data

		if self.from_vision:
			self.discriminator = DiscrimVisionAction(obs_shape=self.obs_shape,
													 action_dim=self.action_shape[0],
													 feature_dim=self.feature_dim,
													 hidden_dim=discrim_hidden_size,
													 create_inp_encoder=not bool(self.share_encoder),
													 use_spectral_norm=self.spectral_norm,
													 use_trunk=self.use_trunk,).to(self.device)
		else:
			self.discriminator = nn.Sequential(nn.Linear(self.obs_shape[0] + self.action_shape[0],
														discrim_hidden_size),
											   nn.ReLU(inplace=True),
											   nn.Linear(discrim_hidden_size, discrim_hidden_size),
											   nn.ReLU(inplace=True),
											   nn.Linear(discrim_hidden_size, 1)).to(self.device)

		self.discrim_opt = torch.optim.Adam(self.discriminator.parameters(), lr=discrim_lr)

	def update_discriminator(self, pos_replay_iter, neg_replay_iter):
		metrics = dict()

		batch_pos = next(pos_replay_iter)
		obs_pos, action_pos, _, _, _, _, _ = utils.to_torch(batch_pos, self.device)
		num_pos = obs_pos.shape[0]

		batch_neg = next(neg_replay_iter)
		obs_neg, action_neg, _, _, _, _, _ = utils.to_torch(batch_neg, self.device)
		num_neg = obs_neg.shape[0]

		'''augment the images and encode + trunk images before mixup,
		the shared encoder has not seen mixed up images.'''
		if self.from_vision:
			# TODO: maybe remove augmentation from the goal image?
			obs_pos = self.aug(obs_pos.float())
			obs_neg = self.aug(obs_neg.float())

			# frozen shared encoder
			if self.share_encoder == 1:
				with torch.no_grad():
					obs_pos = self.encoder(obs_pos)
					obs_neg = self.encoder(obs_neg)
			# update shared encoder
			elif self.share_encoder == 2:
				obs_pos = self.encoder(obs_pos)
				obs_neg = self.encoder(obs_neg)
			# use and train discriminator's own encoder
			elif self.share_encoder == 0:
				obs_pos = self.discriminator.encode(obs_pos)
				obs_neg = self.discriminator.encode(obs_neg)

			# run mixup on low-dim / trunk representation
			# TODO: maybe do mixup before the trunk layer?
			if self.use_trunk:
				obs_pos = self.discriminator.trunk_pass(obs_pos)
				obs_neg = self.discriminator.trunk_pass(obs_neg)

		# TODO: add gaussian noise to just the actions independently?
		pos_input = torch.cat([obs_pos, action_pos], 1)
		neg_input = torch.cat([obs_neg, action_neg], 1)

		if self.mixup:
			alpha = 1.0
			beta_dist = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

			l = beta_dist.sample([num_pos + num_neg])
			mixup_coef = torch.reshape(l, (num_pos + num_neg, 1)).to(self.device)

			mixup_coef_lab = torch.reshape(l, (num_pos + num_neg, 1)).to(self.device)
			labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)

			disc_inputs = torch.cat((pos_input, neg_input), 0)
			ridxs = torch.randperm(num_pos + num_neg)
			perm_labels = labels[ridxs]
			perm_disc_inputs = disc_inputs[ridxs]

			images = disc_inputs * mixup_coef + perm_disc_inputs * (1 - mixup_coef)
			labels = labels * mixup_coef_lab + perm_labels * (1 - mixup_coef_lab)
		else:
			images = torch.cat((pos_input, neg_input), 0)
			labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)

		loss = torch.nn.BCELoss()
		m = nn.Sigmoid()

		if self.from_vision:
			# TODO: action should still obey the bounds after adding gaussian noise
			images = images + self.gaussian_noise_coef * torch.randn_like(images)
			output = m(self.discriminator.final_out(images))
			discrim_loss = loss(output, labels)
		else:
			output = m(self.discriminator(images))
			discrim_loss = loss(output, labels)

		if self.from_vision and self.share_encoder == 2:
			self.encoder_opt.zero_grad(set_to_none=True)

		self.discrim_opt.zero_grad(set_to_none=True)
		discrim_loss.backward()
		self.discrim_opt.step()

		if self.from_vision and self.share_encoder == 2:
			self.encoder_opt.step()

		if self.use_tb:
			metrics['discriminator_loss'] = discrim_loss.item()
			if not self.mixup:
				metrics['discriminator_acc'] = ((output > 0.5) == labels).type(torch.float).mean().item()
			if self.discrim_val_data is not None:
				with torch.no_grad():
					_, output = self.compute_reward(self.discrim_val_data['observations'], self.discrim_val_data['actions'], return_sig=True, evald=True)
					val_labels = torch.ones(self.discrim_val_data['observations'].shape[0], 1).to(self.device)
					metrics['val_loss'] = loss(output, val_labels).item()
					metrics['val_acc'] = ((output > 0.5) == val_labels).type(torch.float).mean().item()

		return metrics

	def compute_reward(self, obs, action, return_sig=False, evald=False):
		if self.from_vision:
			if evald and type(obs) is np.ndarray:
					obs = torch.from_numpy(obs).to(self.device)
					action = torch.from_numpy(action).to(self.device)
			if self.share_encoder:
				obs = self.encoder(obs)
			sig_term = torch.sigmoid(self.discriminator(obs, action))
		else:
			sig_term = torch.sigmoid(self.discriminator(torch.cat([obs, action], axis=1)))

		if self.reward_type == 'logd':
			actual_reward = torch.log(torch.minimum(sig_term + self.eps, torch.tensor(1.)))
		else:
			actual_reward = -torch.log(1 - sig_term + self.eps)
		
		if not return_sig:
			return actual_reward
		else:
			return actual_reward, sig_term

	def transition_tuple(self, replay_iter, demo_iter=None, oversample_count=None, gt_reward=False, online_buf_len=None, offline_buf_len=None):
		obs, action, reward, discount, next_obs, step_type, next_step_type = super().transition_tuple(replay_iter,
																									  demo_iter=demo_iter,
																									  oversample_count=oversample_count,
																									  online_buf_len=online_buf_len,
																									  offline_buf_len=offline_buf_len)
		if gt_reward:
			return (obs, action, reward, discount, next_obs, step_type, next_step_type)
		else:
			with torch.no_grad():
				GAIL_reward = self.compute_reward(obs, action)
			return (obs, action, GAIL_reward, discount, next_obs, step_type, next_step_type)


class VICESACAgent(DACSACAgent):
	def __init__(self, *args,
				 discrim_hidden_size=128,
				 discrim_lr=3e-4,
				 mixup=True,
				 reward_type='logd',
				 share_encoder=False,
				 gaussian_noise_coef=0.,
				 use_trunk=True,
				 spectral_norm=True,
				 pos_dataset=None,
				 **kwargs):

		super(VICESACAgent, self).__init__(*args, **kwargs)
		self.discrim_hidden_size = discrim_hidden_size
		self.discrim_lr = discrim_lr
		self.discrim_hidden_size = discrim_hidden_size
		self.mixup = mixup
		self.reward_type = reward_type
		self.share_encoder = share_encoder
		self.gaussian_noise_coef = gaussian_noise_coef
		self.use_trunk = use_trunk
		self.spectral_norm = spectral_norm
		self.pos_dataset = pos_dataset
		self.num_goals = self.pos_dataset.shape[0]
		self.eps = 1e-10

		if self.from_vision:
			self.discriminator = DiscrimVision(obs_shape=self.obs_shape,
											   feature_dim=self.feature_dim,
											   hidden_dim=self.discrim_hidden_size,
											   repr_dim=self.model_repr_dim,
											   create_inp_encoder=not bool(self.share_encoder),
											   use_spectral_norm=self.spectral_norm,
											   use_trunk=self.use_trunk,).to(self.device)
		else:
			self.discriminator = nn.Sequential(nn.Linear(self.obs_shape[0], discrim_hidden_size),
											   nn.ReLU(inplace=True),
											   nn.Linear(discrim_hidden_size, discrim_hidden_size),
											   nn.ReLU(inplace=True),
											   nn.Linear(discrim_hidden_size, 1)).to(self.device)

		self.discrim_opt = torch.optim.Adam(self.discriminator.parameters(), lr=discrim_lr)
		self.zero_alpha = torch.tensor(0.).to(self.device)

	def update_discriminator(self, neg_replay_iter):
		metrics = dict()

		batch_neg = next(neg_replay_iter)
		obs_neg = utils.to_torch(batch_neg, self.device)[0]
		num_pos = num_neg = obs_neg.shape[0]

		# shuffle the goal states then sample a minibatch
		if self.num_goals < num_neg:
			ridxs = torch.cat([torch.randperm(self.num_goals) for _ in range(math.ceil(num_neg / self.num_goals))])[:num_neg]
		else:
			ridxs = torch.randperm(self.num_goals)[:num_neg]
		obs_pos = self.pos_dataset[ridxs].to(self.device).type(torch.cuda.FloatTensor)

		'''augment the images and encode + trunk images before mixup,
		the shared encoder has not seen mixed up images.'''
		if self.from_vision:
			obs_pos = self.aug(obs_pos.float())
			obs_neg = self.aug(obs_neg.float())

			# frozen shared encoder
			if self.share_encoder == 1:
				with torch.no_grad():
					obs_pos = self.encoder(obs_pos)
					obs_neg = self.encoder(obs_neg)
			# update shared encoder
			elif self.share_encoder == 2:
				obs_pos = self.encoder(obs_pos)
				obs_neg = self.encoder(obs_neg)
			# use and train discriminator's own encoder
			elif self.share_encoder == 0:
				obs_pos = self.discriminator.encode(obs_pos)
				obs_neg = self.discriminator.encode(obs_neg)

			if self.use_trunk:
				obs_pos = self.discriminator.trunk_pass(obs_pos)
				obs_neg = self.discriminator.trunk_pass(obs_neg)

		pos_input = obs_pos
		neg_input = obs_neg

		# exit()
		if self.mixup:
			disc_inputs = torch.cat((pos_input, neg_input), 0)
			labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)

			# Beta(1,1) is Uniform[0,1]
			beta_dist = torch.distributions.beta.Beta(torch.tensor([1.0]), torch.tensor([1.0]))
			l = beta_dist.sample([num_pos + num_neg])
			mixup_coef = torch.reshape(l, (num_pos + num_neg, 1)).to(self.device)

			# permute the images and labels for mixing up
			ridxs = torch.randperm(num_pos + num_neg)
			perm_labels = labels[ridxs]
			perm_disc_inputs = disc_inputs[ridxs]

			# create mixed up inputs
			images = disc_inputs * mixup_coef + perm_disc_inputs * (1 - mixup_coef)
			labels = labels * mixup_coef + perm_labels * (1 - mixup_coef)
		else:
			images = torch.cat((pos_input, neg_input), 0)
			labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)

		loss = torch.nn.BCELoss()
		m = nn.Sigmoid()

		if self.from_vision:
			images = images + self.gaussian_noise_coef * torch.randn_like(images)
			output = m(self.discriminator.final_out(images))
			discrim_loss = loss(output, labels)
		else:
			output = m(self.discriminator(images))
			discrim_loss = loss(output, labels)

		if self.from_vision and self.share_encoder == 2:
			self.encoder_opt.zero_grad(set_to_none=True)

		self.discrim_opt.zero_grad(set_to_none=True)
		discrim_loss.backward()
		self.discrim_opt.step()

		if self.from_vision and self.share_encoder == 2:
			self.encoder_opt.step()

		if self.use_tb:
			metrics['discriminator_loss'] = discrim_loss.item()
			if not self.mixup:
				metrics['discriminator_acc'] = ((output > 0.5) == labels).type(torch.float).mean().item()
			if self.discrim_val_data is not None:
				with torch.no_grad():
					_, output = self.compute_reward(self.discrim_val_data['observations'], return_sig=True, evald=True)
					val_labels = torch.ones(self.discrim_val_data['observations'].shape[0], 1).to(self.device)
					metrics['val_loss'] = loss(output, val_labels).item()
					metrics['val_acc'] = ((output > 0.5) == val_labels).type(torch.float).mean().item()

		return metrics

	def compute_reward(self, obs, action=None, return_sig=False, evald=False):
		del action
		if self.from_vision:
			if evald and type(obs) is np.ndarray:
				obs = torch.from_numpy(obs).to(self.device)

			if self.share_encoder:
				obs = self.encoder(obs)

		sig_term = torch.sigmoid(self.discriminator(obs))
		if self.reward_type == 'logd':
			actual_reward = torch.log(torch.minimum(sig_term + self.eps, torch.tensor(1.)))
		else:
			actual_reward = -torch.log(1 - sig_term + self.eps)

		if not return_sig:
			return actual_reward
		else:
			return actual_reward, sig_term

	def transition_tuple(self, replay_iter, demo_iter=None, oversample_count=None, gt_reward=False, online_buf_len=None, offline_buf_len=None):
		obs, action, reward, discount, next_obs, step_type, next_step_type = super().transition_tuple(replay_iter,
																									  demo_iter=demo_iter,
																									  oversample_count=oversample_count,
																									  gt_reward=gt_reward,
																									  online_buf_len=online_buf_len,
																									  offline_buf_len=offline_buf_len)
		if gt_reward:
			return (obs, action, reward, discount, next_obs, step_type, next_step_type)
		else:
			with torch.no_grad():
				VICE_reward = self.compute_reward(next_obs, None)
			return (obs, action, VICE_reward, discount, next_obs, step_type, next_step_type)


class VICEMixtureAgent(VICEAgent):
	'''
	An agent that uses a mixture of policies to match a state distribution.
	NOTE: does not implement BC regularization.
	'''
	def __init__(self, *args,
				 num_policies=2,
				 diversity_weight=0.0,
				 **kwargs):

		# the mixture latent will be concatenated after the embedding
		if 'franka' in kwargs:
			kwargs['franka'] += num_policies
		else:
			kwargs['franka'] = num_policies
		super(VICEMixtureAgent, self).__init__(*args, **kwargs)
		self._num_policies = num_policies
		self._diversity_weight = diversity_weight

		if self.from_vision:
			self.latent_discriminator = LatentDiscriminator(num_outs=self._num_policies,
															obs_shape=self.obs_shape,
													  		feature_dim=self.feature_dim,
													  		hidden_dim=self.discrim_hidden_size,
													  		repr_dim=self.model_repr_dim,
													  		create_inp_encoder=not bool(self.share_encoder),
													  		use_spectral_norm=self.spectral_norm,
													  		use_trunk=self.use_trunk,).to(self.device)
		else:
			self.latent_discriminator = nn.Sequential(nn.Linear(self.obs_shape[0], self.discrim_hidden_size),
											   		  nn.ReLU(inplace=True),
											   		  nn.Linear(self.discrim_hidden_size, self.discrim_hidden_size),
											   		  nn.ReLU(inplace=True),
											   		  nn.Linear(self.discrim_hidden_size, self._num_policies)).to(self.device)

		self.latent_disc_opt = torch.optim.Adam(self.latent_discriminator.parameters(), lr=3e-4)
		self.switch_policy() # initialize a latent

	@property
	def latent(self):
		return self.cur_latent

	@property
	def num_policies(self):
		return self._num_policies

	def _get_new_latent(self):
		new_latent = np.zeros((self.num_policies,)).astype(np.float32)
		new_latent[np.random.randint(self.num_policies)] = 1.
		return new_latent

	def switch_policy(self):
		self.set_latent(self._get_new_latent())

	def set_latent(self, new_latent):
		self._cur_policy = np.argmax(new_latent)
		self.cur_latent = new_latent
		self.torch_latent = torch.from_numpy(self.cur_latent).to(self.device)

	def act(self, obs, uniform_action=False, eval_mode=False):
		obs = torch.as_tensor(obs, device=self.device)
		if self.from_vision:
			obs = self.encoder(obs.unsqueeze(0))[0]

		dist = self.actor(torch.concat([obs, self.torch_latent], dim=-1).unsqueeze(0))
		if eval_mode:
			print(f'evaluating policy {self._cur_policy}')
			action = dist.mean[0]
		else:
			action = dist.sample()[0]

		if uniform_action:
			action.uniform_(-1.0, 1.0)

		return action.cpu().numpy()

	def update(self, trans_tuple_fn, step, trans_tuple_demo=None):
		metrics = dict()

		for _ in range(self.utd_ratio):
			obs, latent, action, reward, discount, next_obs, next_latent, step_type, next_step_type = trans_tuple_fn()

			not_done = next_step_type.clone()
			not_done[not_done < 2] = 1
			not_done[not_done == 2] = 0

			# augment
			if self.from_vision:
				obs = self.aug(obs.float())
				next_obs = self.aug(next_obs.float())
				# encode
				obs = self.encoder(obs)
				with torch.no_grad():
					next_obs = self.encoder(next_obs)

			# concatenate with latent
			obs = torch.cat([obs, latent], dim=-1)
			next_obs = torch.cat([next_obs, next_latent], dim=-1)

			# update critic
			metrics.update(
				self.update_critic(obs, action, reward, discount, next_obs, step, not_done))

			# soft update target parameters
			for param, target_param in zip(self.Qens_params, self.Qtens_params):
				target_param.data.copy_(self.critic_target_tau * param.data + 
									   (1. - self.critic_target_tau) * target_param.data)

		# NOTE: BC regularization isn't implemented for mixture agents, should be 0.0
		if trans_tuple_demo is not None:
			obs_d, action_d, _, _, _, _, _ = trans_tuple_demo
			if self.from_vision:
				obs_d = self.aug(obs_d.float())
				# TODO: pass the gradients from BC loss to the encoder
				with torch.no_grad():
					obs_d = self.encoder(obs_d)

			if self.bc_reg_lambda == 'soft_qfilter':
				with torch.no_grad():
					dist_d = self.actor(obs_d)
					Q_p = torch.mean(self.ensemble_forward_pass(obs_d, dist_d.rsample()).transpose(0, 1), dim=1)
					Q_bc = torch.mean(self.ensemble_forward_pass(obs_d, action_d).transpose(0, 1), dim=1)
					mask = Q_bc > Q_p
					bc_reg = mask.float().mean()

			elif self.bc_reg_lambda == 'hard_qfilter':
				with torch.no_grad():
					dist_d = self.actor(obs_d)
					Q_p = torch.mean(self.ensemble_forward_pass(obs_d, dist_d.rsample()).transpose(0, 1), dim=1)
					Q_bc = torch.mean(self.ensemble_forward_pass(obs_d, action_d).transpose(0, 1), dim=1)
					mask = Q_bc > Q_p
					# hard_qfilter: remove the state-action pairs that have lower Q-vals
					obs_d, action_d = obs_d[mask], action_d[mask]
					bc_reg = mask.float().mean()
			else:
				# either a fixed float or linear decay
				bc_reg = utils.schedule(self.bc_reg_lambda, step)

			metrics.update(self.update_actor(obs.detach(),
											 step,
											 bc_reg=bc_reg,
											 obs_d=obs_d,
											 action_d=action_d))
		else:
			metrics.update(self.update_actor(obs.detach(), step))

		# update alpha
		metrics.update(self.update_alpha(obs.detach(), step))

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()
			if self.bc_reg_lambda != 0.0:
				metrics['bc_reg_coef'] = bc_reg

		return metrics

	def update_latent_discriminator(self, replay_iter):
		metrics = dict()

		obs, latent = utils.to_torch(next(replay_iter), self.device)[:2]
		if self.from_vision:
			obs = self.aug(obs.float())

			# encode the observations if sharing the encoder
			# encoder is not updated by latent disc's gradients
			if self.share_encoder == 1:
				with torch.no_grad():
					obs = self.encoder(obs)
			# encoder is _updated_ by latent disc's gradients
			elif self.share_encoder == 2:
				obs = self.encoder(obs)

		output = torch.softmax(self.latent_discriminator(obs), dim=1)
		discrim_loss = torch.nn.CrossEntropyLoss()(output, latent)

		if self.from_vision and self.share_encoder == 2:
			self.encoder_opt.zero_grad(set_to_none=True)

		self.latent_disc_opt.zero_grad(set_to_none=True)
		discrim_loss.backward()
		self.latent_disc_opt.step()

		if self.from_vision and self.share_encoder == 2:
			self.encoder_opt.step()

		if self.use_tb:
			metrics['latent_disc_loss'] = discrim_loss.item()
			metrics['latent_disc_acc'] = (torch.argmax(output, axis=1) == torch.argmax(latent, axis=1)).float().mean().item()
			metrics['latent_disc_prob'] =  output[torch.arange(output.shape[0]), torch.argmax(latent, axis=1)].mean().item()

		return metrics

	def compute_reward(self, obs, latent, action=None, return_sig=False):
		del action
		if self.from_vision:
			if self.share_encoder:
				obs = self.encoder(obs)

		# NOTE: the MEDAL/VICE discriminator does not see the latent
		sig_term = torch.sigmoid(self.discriminator(obs))
		if self.reward_type == 'logd':
			dist_match_reward = torch.log(torch.minimum(sig_term + self.eps, torch.tensor(1.)))
		else:
			dist_match_reward = -torch.log(1 - sig_term + self.eps)

		latent_probs = torch.softmax(self.latent_discriminator(obs), dim=1)
		class_probs = latent_probs[torch.arange(latent.shape[0]), torch.argmax(latent, axis=1)]
		diversity_reward = torch.log(class_probs) + torch.log(torch.tensor(self._num_policies))

		# final reward is a convex combination of diversity and distribution matching rewards
		actual_reward = (1. - self._diversity_weight) * dist_match_reward + self._diversity_weight * diversity_reward.unsqueeze(1)
		if not return_sig:
			return actual_reward
		else:
			return actual_reward, {'sigmoid': sig_term,
								   'class_probs': class_probs,
								   'dist_match': dist_match_reward,
								   'diversity': diversity_reward}

	def transition_tuple(self, replay_iter, demo_iter=None, oversample_count=None):
		'''batch: (obs, latent, action, reward, discount, next_obs, next_latent, step_type, next_step_type)'''
		batch = utils.to_torch(next(replay_iter), self.device)

		if demo_iter is not None:
			num_offline = oversample_count
			num_online = batch[0].shape[0] - num_offline

			# assign latents randomly to demo transitions
			batch_d = utils.to_torch(next(demo_iter), self.device)
			demo_latent = torch.from_numpy(np.stack([self._get_new_latent() for _ in range(num_offline)])).to(self.device)
			# batch_d: (obs, action, reward, discount, next_obs, step_type, next_step_type)
			batch_d = batch_d[:1] + (demo_latent,) + batch_d[1:5] + (demo_latent,) + batch_d[5:]

			new_batch = ()
			for x, y in zip(batch, batch_d):
				new_batch += (torch.cat((x[:num_online], y[:num_offline]), axis=0),)
			batch = new_batch

		with torch.no_grad():
			# batch[5]: next_obs, batch[1]: latent
			new_reward = self.compute_reward(batch[5], batch[1])
			batch = batch[:3] + (new_reward,) + batch[4:]

		return batch

class SQILAgent(REDQAgent):
	def __init__(self, *agent_args, **kwargs):
		super(SQILAgent, self).__init__(*agent_args, **kwargs)
		self.zero_alpha = torch.tensor(0.).to(self.device)

	@property
	def alpha(self):
		return self.zero_alpha

	def update_alpha(self, *args):
		metrics = dict()
		return metrics

	def transition_tuple(self, replay_iter, demo_iter=None, oversample_count=None, gt_reward=False):
		obs, action, reward, discount, next_obs, step_type, next_step_type = super().transition_tuple(replay_iter,
																									  demo_iter=demo_iter,
																									  oversample_count=oversample_count)
		if gt_reward:
			return (obs, action, reward, discount, next_obs, step_type, next_step_type)
		else:
			# TODO: regularize the Q-value function to effectively learn policies (mixup perhaps?)
			SQIL_reward = torch.cat((torch.zeros(obs.shape[0] - oversample_count, 1).to(self.device),
									 torch.ones(oversample_count, 1).to(self.device)), 0)
			return (obs, action, SQIL_reward, discount, next_obs, step_type, next_step_type)

class REDQFrankaAgent(REDQAgent):
	def __init__(self, *agent_args, **agent_kwargs):
		# infer the specs
		obs_spec = agent_kwargs.pop('obs_spec')
		agent_kwargs['obs_shape'] = list(obs_spec['images'].shape)
		self.franka_dim = obs_spec['state'].shape[0]
		super().__init__(*agent_args, **agent_kwargs)

		# recreate actor with franka dimensions
		self.actor = SACActor(self.model_repr_dim, self.action_shape, self.feature_dim,
							  self.hidden_dim, self.log_std_bounds, franka=self.franka_dim).to(self.device)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

		# recreate critics with franka dimensions
		self.critic_list, self.critic_target_list = [], []
		for _ in range(self.num_Q):
			new_critic = Qnet(self.model_repr_dim, self.action_shape, self.feature_dim,
							  self.hidden_dim, franka=self.franka_dim).to(self.device)
			self.critic_list.append(new_critic)
			new_critic_target = Qnet(self.model_repr_dim, self.action_shape, self.feature_dim,
									 self.hidden_dim, franka=self.franka_dim).to(self.device)
			# initialize target critic network
			new_critic_target.load_state_dict(new_critic.state_dict())
			self.critic_target_list.append(new_critic_target)

		# setting up feedforward pass via vmap
		self.Qens, self.Qens_params, self.Qens_buffers = combine_state_for_ensemble(self.critic_list)
		[p.requires_grad_() for p in self.Qens_params];
		self.Qtens, self.Qtens_params, self.Qtens_buffers = combine_state_for_ensemble(self.critic_target_list)
		self.critic_opt = torch.optim.Adam(self.Qens_params, lr=self.lr)
		self.train()

	def act(self, obs, uniform_action=False, eval_mode=False):
		obs_img = torch.as_tensor(obs['images'], device=self.device).float()
		obs_state = torch.as_tensor(obs['state'], device=self.device).float()

		h = self.encoder(obs=obs_img)
		dist = self.actor(torch.cat([h, obs_state], axis=-1).unsqueeze(0))

		if eval_mode:
			action = dist.mean[0]
		else:
			action = dist.sample()[0]

		if uniform_action:
			action.uniform_(-1.0, 1.0)

		return action.cpu().numpy()

	def update(self, trans_tuple_fn, step, trans_tuple_demo=None):
		metrics = dict()
		for _ in range(self.utd_ratio):
			obs_img, obs_state, action, reward, discount, next_obs_img, next_obs_state, step_type, next_step_type = trans_tuple_fn()
			obs_state = obs_state.float()
			next_obs_state = next_obs_state.float()

			not_done = next_step_type.clone()
			not_done[not_done < 2] = 1
			not_done[not_done == 2] = 0

			# augment
			obs_img = self.aug(obs_img.float())
			next_obs_img = self.aug(next_obs_img.float())

			h_obs = self.encoder(obs=obs_img)
			obs = torch.cat([h_obs, obs_state], axis=1)

			with torch.no_grad():
				h_next_obs = self.encoder(obs=next_obs_img)
				next_obs = torch.cat([h_next_obs, next_obs_state], axis=1)

			# update critic
			metrics.update(
				self.update_critic(obs, action, reward, discount, next_obs, step, not_done))

			# soft update target parameters
			for param, target_param in zip(self.Qens_params, self.Qtens_params):
				target_param.data.copy_(self.critic_target_tau * param.data + 
									   (1. - self.critic_target_tau) * target_param.data)

		# update actor
		if trans_tuple_demo is not None:
			obs_img_d, obs_state_d, action_d, _, _, _, _, _, _ = trans_tuple_demo
			obs_img_d = self.aug(obs_img_d.float())
			obs_state_d = obs_state_d.float()
			# encode
			with torch.no_grad():
				h_d = self.encoder(obs=obs_img_d)
			obs_d = torch.cat([h_d, obs_state_d], axis=1)

			# can be a linear decay or fixed value
			bc_reg = utils.schedule(self.bc_reg_lambda, step)
			metrics.update(self.update_actor(obs.detach(),
											 step,
											 bc_reg=bc_reg,
											 obs_d=obs_d,
											 action_d=action_d))
		else:
			metrics.update(self.update_actor(obs.detach(), step))

		# update alpha
		metrics.update(self.update_alpha(obs.detach(), step))

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()
			if self.bc_reg_lambda != 0.0:
				metrics['bc_reg_coef'] = bc_reg

		return metrics

	def transition_tuple(self, replay_iter, demo_iter=None, oversample_count=None):

		def _split_obs_state(batch):
			obs, action, reward, discount, next_obs, step_type, next_step_type = batch
			obs_img, obs_state = obs['images'], obs['state']
			next_obs_img, next_obs_state = next_obs['images'], next_obs['state']
			return utils.to_torch((obs_img, obs_state,
								   action,
								   reward,
								   discount,
								   next_obs_img, next_obs_state,
								   step_type,
								   next_step_type),
								   self.device)

		batch = _split_obs_state(next(replay_iter))

		if demo_iter is not None:
			num_offline = oversample_count
			num_online = batch[0].shape[0] - num_offline
			batch_d = _split_obs_state(next(demo_iter))

			new_batch = ()
			for x, y in zip(batch, batch_d):
				new_batch += (torch.cat((x[:num_online], y[:num_offline]), axis=0),)

			return new_batch
		else:
			return batch

class DACFrankaAgent(REDQFrankaAgent):
	def __init__(self,
				 *agent_args,
				 discrim_hidden_size=128,
				 discrim_lr=3e-4,
				 mixup=True,
				 reward_type='logd',
				 share_encoder=False,
				 gaussian_noise_coef=0.,
				 use_trunk=True,
				 spectral_norm=True,
				 discrim_val_data=None,
				 **agent_kwargs):

		super(DACFrankaAgent, self).__init__(**agent_kwargs)
		self.discrim_hidden_size = discrim_hidden_size
		self.discrim_lr = discrim_lr
		self.mixup = mixup
		self.eps = 1e-10
		self.reward_type = reward_type
		self.share_encoder = share_encoder
		self.gaussian_noise_coef = gaussian_noise_coef
		self.use_trunk = use_trunk
		self.spectral_norm = spectral_norm
		self.discrim_val_data = discrim_val_data

		self.discriminator = DiscrimVisionActionFranka(obs_shape=self.obs_shape,
													   action_dim=self.action_shape[0],
													   feature_dim=self.feature_dim,
													   hidden_dim=self.discrim_hidden_size,
													   create_inp_encoder=not bool(self.share_encoder)).to(self.device)

		self.discrim_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.discrim_lr)

	def update_discriminator(self, pos_replay_iter, neg_replay_iter):
		metrics = dict()

		batch_pos = next(pos_replay_iter)
		obs_pos, action_pos, _, _, _, _, _ = batch_pos
		obs_img_pos, obs_state_pos = obs_pos['images'], obs_pos['state']
		obs_img_pos, obs_state_pos, action_pos = utils.to_torch((obs_img_pos, obs_state_pos, action_pos), self.device)
		num_pos = obs_state_pos.shape[0]

		batch_neg = next(neg_replay_iter)
		obs_neg, action_neg, _, _, _, _, _ = batch_neg
		obs_img_neg, obs_state_neg = obs_neg['images'], obs_neg['state']
		obs_img_neg, obs_state_neg, action_neg = utils.to_torch((obs_img_neg, obs_state_neg, action_neg), self.device)
		num_neg = obs_state_neg.shape[0]

		# augment the images and encode + trunk images before mixup
		# the shared encoder has not seen mixed up images!!
		obs_img_pos = self.aug(obs_img_pos.float())
		obs_img_neg = self.aug(obs_img_neg.float())

		# frozen shared encoder
		if self.share_encoder == 1:
			with torch.no_grad():
				obs_pos_h = self.encoder(obs_img_pos)
				obs_neg_h = self.encoder(obs_img_neg)
		# update shared encoder
		elif self.share_encoder == 2:
			obs_pos_h = self.encoder(obs_img_pos)
			obs_neg_h = self.encoder(obs_img_neg)
		elif self.share_encoder == 0:
			raise ValueError('Discriminator must share encoder currently')

		# run mixup on low-dim / trunk representation
		if self.use_trunk:
			obs_pos_trunked_h = self.discriminator.trunk_pass(obs_pos_h)
			obs_neg_trunked_h = self.discriminator.trunk_pass(obs_neg_h)
			obs_pos = torch.cat([obs_pos_trunked_h, obs_state_pos], axis=1)
			obs_neg = torch.cat([obs_neg_trunked_h, obs_state_neg], axis=1)
		else:
			raise ValueError('Must trunk images for Franka')

		pos_input = torch.cat([obs_pos, action_pos], axis=1)
		neg_input = torch.cat([obs_neg, action_neg], axis=1)
		if self.mixup:
			alpha = 1.0
			beta_dist = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

			l = beta_dist.sample([num_pos + num_neg])
			mixup_coef = torch.reshape(l, (num_pos + num_neg, 1)).to(self.device)
	
			mixup_coef_lab = torch.reshape(l, (num_pos + num_neg, 1)).to(self.device)
			labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)

			disc_inputs = torch.cat((pos_input, neg_input), 0)
			ridxs = torch.randperm(num_pos + num_neg)
			perm_labels = labels[ridxs]
			perm_disc_inputs = disc_inputs[ridxs]

			images = disc_inputs * mixup_coef + perm_disc_inputs * (1 - mixup_coef)
			labels = labels * mixup_coef_lab + perm_labels * (1 - mixup_coef_lab)
		else:
			images = torch.cat((pos_input, neg_input), 0)
			labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)

		loss = torch.nn.BCELoss()
		m = nn.Sigmoid()

		images = images + self.gaussian_noise_coef * torch.randn_like(images)
		output = m(self.discriminator.final_out(images))
		discrim_loss = loss(output, labels)

		if self.share_encoder == 2:
			self.encoder_opt.zero_grad(set_to_none=True)

		self.discrim_opt.zero_grad(set_to_none=True)
		discrim_loss.backward()
		self.discrim_opt.step()

		if self.share_encoder == 2:
			self.encoder_opt.step()

		if self.use_tb:
			metrics['discriminator_loss'] = discrim_loss.item()
		if not self.mixup:
				metrics['discriminator_acc'] = ((output > 0.5) == labels).type(torch.float).mean().item()
		if self.discrim_val_data is not None:
			with torch.no_grad():
				_, output = self.compute_reward(self.discrim_val_data['observations'],
												self.discrim_val_data['actions'],
												return_sig=True, evald=True)
				val_labels = torch.ones(self.discrim_val_data['observations']['state'].shape[0], 1).to(self.device)
				metrics['val_loss'] = loss(output, val_labels).item()
				metrics['val_acc'] = ((output > 0.5) == val_labels).type(torch.float).mean().item()

		return metrics

	def compute_reward(self, obs, action, return_sig=False, evald=False):
		# only share encoder
		if self.share_encoder:
			if evald:
				ob_img = torch.from_numpy(obs['images']).to(self.device)
				ob_state = torch.from_numpy(obs['state']).to(self.device)
				action = torch.from_numpy(action).to(self.device)
			else:
				ob_img = obs['images']
				ob_state = obs['state']

			ob_h = self.encoder(ob_img)
			trunked_h = self.discriminator.trunk_pass(ob_h)
			# concat the trunked image embedding and state for discriminator
			obs = torch.cat([trunked_h, ob_state], axis=1)
			sig_term = torch.sigmoid(self.discriminator(obs, action))
		else:
			raise ValueError('Discriminator must share encoder currently')

		if self.reward_type == 'logd':
			actual_reward = torch.log(sig_term + self.eps)
		else:
			actual_reward = -torch.log(1 - sig_term + self.eps)

		if not return_sig:
			return actual_reward
		else:
			return actual_reward, sig_term

	def transition_tuple(self, replay_iter, demo_iter=None, oversample_count=None):
		obs_img, obs_state, action, _, discount, next_obs_img, next_obs_state, step_type, next_step_type = super().transition_tuple(replay_iter,
																													    			demo_iter=demo_iter,
																																	oversample_count=oversample_count)
		obs_dict = {'images': obs_img, 'state': obs_state}
		with torch.no_grad():
			GAIL_reward = self.compute_reward(obs_dict, action)
		return (obs_img, obs_state, action, GAIL_reward, discount, next_obs_img, next_obs_state, step_type, next_step_type)

class VICEFrankaAgent(REDQFrankaAgent):
	def __init__(self,
				 *args,
				 discrim_hidden_size=256,
				 discrim_lr=3e-4,
				 mixup=True,
				 reward_type='logd',
				 share_encoder=False,
				 gaussian_noise_coef=0.,
				 use_trunk=True,
				 discrim_val_data=None,
				 pos_dataset=None,
				 state_dim=8,
				 ignore_view=None,
				 skip_reward_computation=False,
				 **kwargs):

		super(VICEFrankaAgent, self).__init__(*args, **kwargs)
		self.mixup = mixup
		self.eps = 1e-10
		self.reward_type = reward_type
		self.share_encoder = share_encoder
		self.gaussian_noise_coef = gaussian_noise_coef
		self.use_trunk = use_trunk
		self.discrim_val_data = discrim_val_data
		self.pos_dataset = pos_dataset
		self._skip_reward_computation = skip_reward_computation

		self.discriminator = DiscrimVisionFranka(obs_shape=self.obs_shape,
												 feature_dim=self.feature_dim,
												 hidden_dim=discrim_hidden_size,
												 repr_dim=self.model_repr_dim,
												 create_inp_encoder=not bool(self.share_encoder),
												 # specify state dim to be used by VICE discriminator
												 state_dim=state_dim,
												 # can ignore third / first person views from the camera  
												 ignore_view=ignore_view,).to(self.device)

		# preload goal states as torch tensors
		obs_img_pos, obs_state_pos = self.pos_dataset['images'], self.pos_dataset['states']
		self.obs_img_pos, self.obs_state_pos = utils.to_torch((obs_img_pos, obs_state_pos), self.device)
		self.num_goals = self.obs_img_pos.shape[0]

		self.discrim_opt = torch.optim.Adam(self.discriminator.parameters(), lr=discrim_lr)

	def update_discriminator(self, neg_replay_iter):
		metrics = dict()

		batch_neg = next(neg_replay_iter)
		obs_neg, _, _, _, _, _, _ = batch_neg
		# convert state / image data from dict to torch tensor
		obs_img_neg, obs_state_neg = obs_neg['images'], obs_neg['state']
		obs_img_neg, obs_state_neg = utils.to_torch((obs_img_neg, obs_state_neg), self.device)
		obs_state_neg = obs_state_neg.float()
		num_pos = num_neg = obs_img_neg.shape[0]

		# shuffle the goal states then sample a minibatch
		if self.num_goals < num_neg:
			ridxs = torch.cat([torch.randperm(self.num_goals) for _ in range(math.ceil(num_neg / self.num_goals))])[:num_neg]
		else:
			ridxs = torch.randperm(self.num_goals)
		obs_img_pos, obs_state_pos = self.obs_img_pos[ridxs][:num_neg], self.obs_state_pos[ridxs][:num_neg]

		obs_state_pos = obs_state_pos.float()
		obs_img_pos = self.aug(obs_img_pos.float())
		obs_img_neg = self.aug(obs_img_neg.float())
		
		# frozen shared encoder
		if self.share_encoder == 1:
			with torch.no_grad():
				obs_pos_h = self.encoder(obs_img_pos)
				obs_neg_h = self.encoder(obs_img_neg)
		# update shared encoder
		elif self.share_encoder == 2:
			obs_pos_h = self.encoder(obs_img_pos)
			obs_neg_h = self.encoder(obs_img_neg)
		# use and train discriminator's own encoder
		elif self.share_encoder == 0:
			obs_pos_h = self.discriminator.encode(obs_img_pos)
			obs_neg_h = self.discriminator.encode(obs_img_neg)

		if self.use_trunk:
			obs_pos_trunked_h = self.discriminator.trunk_pass(obs_pos_h)
			obs_neg_trunked_h = self.discriminator.trunk_pass(obs_neg_h)

			obs_pos = torch.cat([obs_pos_trunked_h, obs_state_pos], axis=1)
			obs_neg = torch.cat([obs_neg_trunked_h, obs_state_neg], axis=1)
		else:
			raise ValueError('Must trunk images for Franka')

		pos_input = obs_pos
		neg_input = obs_neg

		if self.mixup:
			disc_inputs = torch.cat((pos_input, neg_input), 0)
			labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)
			# Beta(1,1) is Uniform[0,1]
			beta_dist = torch.distributions.beta.Beta(torch.tensor([1.0]), torch.tensor([1.0]))

			l = beta_dist.sample([num_pos + num_neg])
			mixup_coef = torch.reshape(l, (num_pos + num_neg, 1)).to(self.device)

			# permute images and labels for mixup
			ridxs = torch.randperm(num_pos + num_neg)
			perm_labels = labels[ridxs]
			perm_disc_inputs = disc_inputs[ridxs]

			# create mixed up inputs
			images = disc_inputs * mixup_coef + perm_disc_inputs * (1 - mixup_coef)
			labels = labels * mixup_coef + perm_labels * (1 - mixup_coef)
		else:
			images = torch.cat((pos_input, neg_input), 0)
			labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(self.device)

		loss = torch.nn.BCELoss()
		m = nn.Sigmoid()

		if self.from_vision:
			images = images + self.gaussian_noise_coef * torch.randn_like(images)
			output = m(self.discriminator.final_out(images))
			discrim_loss = loss(output, labels)
		else:
			output = m(self.discriminator(images))
			discrim_loss = loss(output, labels)

		if self.share_encoder == 2:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.discrim_opt.zero_grad(set_to_none=True)
		discrim_loss.backward()
		self.discrim_opt.step()
		if self.share_encoder == 2:
			self.encoder_opt.step()

		if self.use_tb:
			metrics['discriminator_loss'] = discrim_loss.item()
			if not self.mixup:
				metrics['discriminator_acc'] = ((output > 0.5) == labels).type(torch.float).mean().item()
			if self.discrim_val_data is not None:
				with torch.no_grad():
					_, output = self.compute_reward(self.discrim_val_data['observations'], return_sig=True, evald=True)
					val_labels = torch.ones(self.discrim_val_data['observations'].shape[0], 1).to(self.device)
					metrics['val_loss'] = loss(output, val_labels).item()
					metrics['val_acc'] = ((output > 0.5) == val_labels).type(torch.float).mean().item()

		return metrics

	def compute_reward(self, obs, action=None, return_sig=False, evald=False):
		del action
		if evald:
			ob_img = torch.from_numpy(obs['images']).to(self.device)
			ob_state = torch.from_numpy(obs['state']).to(self.device)
		else:
			ob_img = obs['images']
			ob_state = obs['state']

		ob_img = ob_img.float()
		ob_state = ob_state.float()

		if self.share_encoder:
			ob_h = self.encoder(ob_img)
			trunked_h = self.discriminator.trunk_pass(ob_h)
			# concat the trunked image embedding and state for discriminator
			obs = torch.cat([trunked_h, ob_state], axis=1)
		else:
			obs = {'images': ob_img, 'state': ob_state}

		sig_term = torch.sigmoid(self.discriminator(obs))
		if self.reward_type == 'logd':
			actual_reward = torch.log(torch.minimum(sig_term + self.eps, torch.tensor(1.)))
		else:
			actual_reward = -torch.log(1 - sig_term + self.eps)

		if not return_sig:
			return actual_reward
		else:
			return actual_reward, sig_term

	def transition_tuple(self, replay_iter, demo_iter=None, oversample_count=None, gt_reward=False):
		if self._skip_reward_computation:
			return super().transition_tuple(replay_iter, demo_iter, oversample_count)

		obs_img, obs_state, action, env_reward, discount, next_obs_img, next_obs_state, step_type, next_step_type = super().transition_tuple(replay_iter,
																																    demo_iter=demo_iter,
																																		    oversample_count=oversample_count)
		
		if gt_reward:
			return (obs_img, obs_state, action, env_reward, discount, next_obs_img, next_obs_state, step_type, next_step_type)

		# VICE reward should be based on the next observation
		next_obs_dict = {'images': next_obs_img, 'state': next_obs_state}
		with torch.no_grad():
			VICE_reward = self.compute_reward(next_obs_dict)
		return (obs_img, obs_state, action, VICE_reward, discount, next_obs_img, next_obs_state, step_type, next_step_type)

class BCAgent:
	def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
				 hidden_dim, use_tb, from_vision, aug_pad=4, repr_dim=None):

		self.obs_shape = obs_shape
		self.action_shape = action_shape
		self.lr = lr
		self.feature_dim = feature_dim
		self.hidden_dim = hidden_dim
		self.device = device
		self.use_tb = use_tb
		self.from_vision = from_vision
		self.aug_pad = aug_pad
		self.aug = None

		# models
		if self.from_vision:
			self.encoder = Encoder(obs_shape).to(device)
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
			if self.aug_pad > 0:
				self.aug = RandomShiftsAug(pad=self.aug_pad)
			self.model_repr_dim = self.encoder.repr_dim if repr_dim is None else repr_dim
		else:
			self.model_repr_dim = self.obs_shape[0]

		self.actor = DDPGActor(self.model_repr_dim, self.action_shape, self.feature_dim,
							   self.hidden_dim).to(device)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

		self.train()
		self.training = True

	def train(self, training=True):
		self.actor.train(training)
		if self.from_vision:
			self.encoder.train(training)

	def act(self, obs, uniform_action=False, eval_mode=False):
		obs = torch.as_tensor(obs, device=self.device)
		if self.from_vision:
			obs = self.encoder(obs.unsqueeze(0))[0]

		dist = self.actor(obs, std=0.1)
		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample()

		if uniform_action:
			action.uniform_(-1.0, 1.0)

		return action.cpu().numpy()

	def update(self, demo_iter, step):
		metrics = dict()
		obs, action, _, _, _, _, _ = utils.to_torch(next(demo_iter), self.device)

		# augment and encode
		if self.from_vision:
			if self.aug is not None:
				obs = self.aug(obs.float())
			obs = self.encoder(obs)
			self.encoder_opt.zero_grad(set_to_none=True)

		dist = self.actor(obs, std=0.1)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		actor_loss = (-1 * log_prob).mean()

		# compute gradients
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()

		# update networks
		self.actor_opt.step()
		if self.from_vision:
			self.encoder_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()

		return metrics
	
	def compute_log_prob(self, obs, action):
		''' useful when only the loss is needed, for example validation.'''
		if self.from_vision:
			obs = self.encoder(obs)
		dist = self.actor(obs, std=0.1)
		return dist.log_prob(action).sum(-1, keepdim=True).mean()

	def save_snapshot(self):
		keys_to_save = ['actor']
		if self.from_vision:
			keys_to_save += ['encoder']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

class BCFrankaAgent(BCAgent):
	def __init__(self,
				 *agent_args,
				 **agent_kwargs):

		# infer the specs
		obs_spec = agent_kwargs.pop('obs_spec')
		agent_kwargs['obs_shape'] = obs_spec['images'].shape
		franka_dim = obs_spec['state'].shape[0]

		super(BCFrankaAgent, self).__init__(*agent_args, **agent_kwargs)
		self.actor = DDPGActor(self.model_repr_dim, self.action_shape, self.feature_dim,
							   self.hidden_dim, franka=franka_dim).to(self.device)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

		self.train()
		self.training = True

	def act(self, obs, uniform_action=False, eval_mode=False):
		obs_img = torch.as_tensor(obs['images'], device=self.device)
		obs_state = torch.as_tensor(obs['state'], device=self.device).float()

		h  = self.encoder(obs=obs_img)
		obs_trunk = self.actor.trunk_pass(h)
		obs = torch.cat([obs_trunk, obs_state], axis=-1)
		dist = self.actor.final_out(obs, std=0.1)

		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample()

		if uniform_action:
			action.uniform_(-1.0, 1.0)

		return action.cpu().numpy()

	def _split_obs_state(self, batch):
		obs, action, reward, discount, next_obs, step_type, next_step_type = batch
		obs_img, obs_state = obs['images'], obs['state']
		next_obs_img, next_obs_state = next_obs['images'], next_obs['state']
		return utils.to_torch((obs_img, obs_state,
							   action,
							   reward,
							   discount,
							   next_obs_img, next_obs_state,
							   step_type,
							   next_step_type),
							   self.device)

	def update(self, demo_iter, step):
		del step
		metrics = dict()
		obs_img, obs_state, action, _, _, _, _, _, _ = self._split_obs_state(next(demo_iter))

		# augment and encode
		if self.aug is not None:
			obs_img = self.aug(obs_img.float())
		obs_h = self.encoder(obs=obs_img)
		obs_trunk = self.actor.trunk_pass(obs_h)
		lowdim_obs = torch.cat([obs_trunk, obs_state], axis=-1)

		dist = self.actor.final_out(lowdim_obs, std=0.1)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		actor_loss = (-1 * log_prob).mean()

		# compute gradients
		self.encoder_opt.zero_grad(set_to_none=True)
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()

		# update networks
		self.actor_opt.step()
		self.encoder_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['actor_log_prob'] = log_prob.mean().item()

		return metrics

	def compute_log_prob(self, obs_img, obs_state, actions):
		''' useful when only the loss is needed, for example validation.'''
		metrics = dict()

		obs_img = torch.from_numpy(obs_img).to(self.device)
		obs_state = torch.from_numpy(obs_state).to(self.device)
		action = torch.from_numpy(actions).to(self.device)

		obs_h = self.encoder(obs=obs_img)
		# do not trunk state, only the image
		obs_trunk = self.actor.trunk_pass(obs_h)
		lowdim_obs = torch.cat([obs_trunk, obs_state], axis=-1)
		dist = self.actor.final_out(lowdim_obs, std=0.1)
		mean_log_prob = dist.log_prob(action).sum(-1, keepdim=True).mean()

		if self.use_tb:
			metrics['val_log_prob'] = mean_log_prob.item()

		return metrics

class POTILAgent:
	def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
				 hidden_dim, critic_target_tau, num_expl_steps,
				 update_every_steps, stddev_schedule, stddev_clip, use_tb, augment,
				 rewards, sinkhorn_rew_scale, update_target_every,
				 auto_rew_scale, auto_rew_scale_factor, obs_type, bc_weight_type, bc_weight_schedule,
				 repr_dim=None):
		self.device = device
		self.lr = lr
		self.critic_target_tau = critic_target_tau
		self.update_every_steps = update_every_steps
		self.use_tb = use_tb
		self.num_expl_steps = num_expl_steps
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.augment = augment
		self.rewards = rewards
		self.sinkhorn_rew_scale = sinkhorn_rew_scale
		self.update_target_every = update_target_every
		self.auto_rew_scale = auto_rew_scale
		self.auto_rew_scale_factor = auto_rew_scale_factor
		self.use_encoder = True if obs_type == 'pixels' else False
		self.bc_weight_type = bc_weight_type
		self.bc_weight_schedule = bc_weight_schedule
		self.obs_shape = obs_shape
		self.feature_dim = feature_dim
		self.hidden_dim = hidden_dim
		self.action_shape = action_shape

		# models
		if self.use_encoder:
			self.encoder = Encoder(obs_shape).to(device)
			self.encoder_target = Encoder(obs_shape).to(device)
			repr_dim = self.encoder.repr_dim if repr_dim is None else repr_dim
		else:
			repr_dim = obs_shape[0]

		self.trunk_target = nn.Sequential(
			nn.Linear(repr_dim, feature_dim),
			nn.LayerNorm(feature_dim), nn.Tanh()).to(device)

		self.actor = DDPGActor(repr_dim, action_shape, feature_dim,
						   hidden_dim).to(device)

		self.critic = Critic(repr_dim, action_shape, feature_dim,
							 hidden_dim).to(device)
		self.critic_target = Critic(repr_dim, action_shape,
									feature_dim, hidden_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		# optimizers
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

		# data augmentation
		self.aug = RandomShiftsAug(pad=4)

		self.train()
		self.critic_target.train()

	def __repr__(self):
		return "potil"

	def train(self, training=True):
		self.training = training
		if self.use_encoder:
			self.encoder.train(training)
		self.actor.train(training)
		self.critic.train(training)

	def act(self, obs, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device)

		obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder else obs.unsqueeze(0)

		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor(obs, stddev)

		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample(clip=None)
			if step < self.num_expl_steps:
				action.uniform_(-1.0, 1.0)
		return action.cpu().numpy()[0]

	def update_critic(self, obs, action, reward, discount, next_obs, step):
		metrics = dict()

		with torch.no_grad():
			stddev = utils.schedule(self.stddev_schedule, step)
			dist = self.actor(next_obs, stddev)
			next_action = dist.sample(clip=self.stddev_clip)
			target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
			target_V = torch.min(target_Q1, target_Q2)
			target_Q = reward + (discount * target_V)

		Q1, Q2 = self.critic(obs, action)

		critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

		# optimize encoder and critic
		if self.use_encoder:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_opt.step()
		if self.use_encoder:
			self.encoder_opt.step()

		if self.use_tb:
			metrics['critic_target_q'] = target_Q.mean().item()
			metrics['critic_q1'] = Q1.mean().item()
			metrics['critic_q2'] = Q2.mean().item()
			metrics['critic_loss'] = critic_loss.item()
			
		return metrics

	def update_actor(self, obs, obs_bc, obs_qfilter, action_bc, bc_regularize, step):
		metrics = dict()

		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor(obs, stddev)
		action = dist.sample(clip=self.stddev_clip)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		Q1, Q2 = self.critic(obs, action)
		Q = torch.min(Q1, Q2)

		# Compute bc weight
		if not bc_regularize:
			bc_weight = 0.0
		elif self.bc_weight_type == "linear":
			bc_weight = utils.schedule(self.bc_weight_schedule, step)
		elif self.bc_weight_type == "qfilter":
			"""
			Soft Q-filtering inspired from 			
			Nair, Ashvin, et al. "Overcoming exploration in reinforcement 
			learning with demonstrations." 2018 IEEE international 
			conference on robotics and automation (ICRA). IEEE, 2018.
			"""
			with torch.no_grad():
				stddev = 0.1
				dist_qf = self.actor_bc(obs_qfilter, stddev)
				action_qf = dist_qf.mean
				Q1_qf, Q2_qf = self.critic(obs_qfilter.clone(), action_qf)
				Q_qf = torch.min(Q1_qf, Q2_qf)
				bc_weight = (Q_qf>Q).float().mean().detach()

		actor_loss = - Q.mean() * (1-bc_weight)

		if bc_regularize:
			stddev = 0.1
			dist_bc = self.actor(obs_bc, stddev)
			log_prob_bc = dist_bc.log_prob(action_bc).sum(-1, keepdim=True)
			actor_loss += - log_prob_bc.mean()*bc_weight*0.03

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()
		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['actor_logprob'] = log_prob.mean().item()
			metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
			metrics['actor_q'] = Q.mean().item()
			if bc_regularize and self.bc_weight_type == "qfilter":
				metrics['actor_qf'] = Q_qf.mean().item()
			metrics['bc_weight'] = bc_weight
			metrics['regularized_rl_loss'] = -Q.mean().item()* (1-bc_weight)
			metrics['rl_loss'] = -Q.mean().item()
			if bc_regularize:
				metrics['regularized_bc_loss'] = - log_prob_bc.mean().item()*bc_weight*0.03
				metrics['bc_loss'] = - log_prob_bc.mean().item()*0.03
		
		return metrics

	def update(self, replay_iter, expert_replay_iter, step, bc_regularize=False):
		metrics = dict()

		if step % self.update_every_steps != 0:
			return metrics

		batch = next(replay_iter)
		obs, action, reward, discount, next_obs, _, _ = utils.to_torch(
			batch, self.device)

		# augment
		if self.use_encoder and self.augment:
			obs_qfilter = self.aug(obs.clone().float())
			obs = self.aug(obs.float())
			next_obs = self.aug(next_obs.float())
		else:
			obs_qfilter = obs.clone().float()
			obs = obs.float()
			next_obs = next_obs.float()

		if self.use_encoder:
			# encode
			obs = self.encoder(obs)
			with torch.no_grad():
				next_obs = self.encoder(next_obs)

		if bc_regularize:
			batch = next(expert_replay_iter)
			obs_bc, action_bc = utils.to_torch(batch, self.device)
			# augment
			if self.use_encoder and self.augment:
				obs_bc = self.aug(obs_bc.float())
			else:
				obs_bc = obs_bc.float()
			# encode
			if bc_regularize and self.bc_weight_type == "qfilter":
				obs_qfilter = self.encoder_bc(obs_qfilter) if self.use_encoder else obs_qfilter
				obs_qfilter = obs_qfilter.detach()
			else:
				obs_qfilter = None
			obs_bc = self.encoder(obs_bc) if self.use_encoder else obs_bc 
			# Detach grads
			obs_bc = obs_bc.detach()
		else:
			obs_qfilter = None
			obs_bc = None 
			action_bc = None

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()

		# update critic
		metrics.update(
			self.update_critic(obs, action, reward, discount, next_obs, step))

		# update actor
		metrics.update(self.update_actor(obs.detach(), obs_bc, obs_qfilter, action_bc, bc_regularize, step))

		# update critic target
		utils.soft_update_params(self.critic, self.critic_target,
								 self.critic_target_tau)

		return metrics

	def ot_rewarder(self, observations, demos, step):

		if step % self.update_target_every == 0:
			if self.use_encoder:
				self.encoder_target.load_state_dict(self.encoder.state_dict())
			self.trunk_target.load_state_dict(self.actor.trunk.state_dict())
			self.target_updated = True

		scores_list = list()
		ot_rewards_list = list()
		for demo in demos:
			obs = torch.tensor(observations).to(self.device).float()
			obs = self.trunk_target(self.encoder_target(obs)) if self.use_encoder else self.trunk_target(obs)
			exp = torch.tensor(demo).to(self.device).float()
			exp = self.trunk_target(self.encoder_target(exp)) if self.use_encoder else self.trunk_target(exp)
			obs = obs.detach()
			exp = exp.detach()
			
			if self.rewards == 'sinkhorn_cosine':
				cost_matrix = cosine_distance(
					obs, exp)  # Get cost matrix for samples using critic network.
				transport_plan = optimal_transport_plan(
					obs, exp, cost_matrix, method='sinkhorn',
					niter=100).float()  # Getting optimal coupling
				ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
					torch.mm(transport_plan,
							 cost_matrix.T)).detach().cpu().numpy()
				
			elif self.rewards == 'sinkhorn_euclidean':
				cost_matrix = euclidean_distance(
					obs, exp)  # Get cost matrix for samples using critic network.
				transport_plan = optimal_transport_plan(
					obs, exp, cost_matrix, method='sinkhorn',
					niter=100).float()  # Getting optimal coupling
				ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
					torch.mm(transport_plan,
							 cost_matrix.T)).detach().cpu().numpy()
				
			elif self.rewards == 'cosine':
				exp = torch.cat((exp, exp[-1].unsqueeze(0)))
				ot_rewards = -(1. - F.cosine_similarity(obs, exp))
				ot_rewards *= self.sinkhorn_rew_scale
				ot_rewards = ot_rewards.detach().cpu().numpy()
				
			elif self.rewards == 'euclidean':
				exp = torch.cat((exp, exp[-1].unsqueeze(0)))
				ot_rewards = -(obs - exp).norm(dim=1)
				ot_rewards *= self.sinkhorn_rew_scale
				ot_rewards = ot_rewards.detach().cpu().numpy()
				
			else:
				raise NotImplementedError()

			scores_list.append(np.sum(ot_rewards))
			ot_rewards_list.append(ot_rewards)

		# find demo that lead to largest reward, then return the
		# rewards asscoiated with that demo
		closest_demo_index = np.argmax(scores_list)
		return ot_rewards_list[closest_demo_index]

	def load_bc_snapshot(self, payload):
		for k, v in payload.items():
			if k in self.__dict__:
				self.__dict__[k] = v

		self.critic_target.load_state_dict(self.critic.state_dict())
		if self.use_encoder:
			self.encoder_target.load_state_dict(self.encoder.state_dict())
		self.trunk_target.load_state_dict(self.actor.trunk.state_dict())

		if self.bc_weight_type == "qfilter":
			# Store a copy of the BC policy with frozen weights
			if self.use_encoder:
				self.encoder_bc = copy.deepcopy(self.encoder)
				for param in self.encoder_bc.parameters():
					param.requires_grad = False
			self.actor_bc = copy.deepcopy(self.actor)
			for param in self.actor_bc.parameters():
				param.required_grad = False

		# Update optimizers
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

class POTILFrankaAgent(POTILAgent):
	def __init__(self, repr_dim,
					*agent_args,
					**agent_kwargs):
		super(POTILFrankaAgent, self).__init__(**agent_kwargs)
		self.model_repr_dim = repr_dim
		self.trunk_target = nn.Sequential(
			nn.Linear(self.model_repr_dim, self.feature_dim),
			nn.LayerNorm(self.feature_dim), nn.Tanh()).to(self.device)

		self.actor = DDPGActor(self.model_repr_dim, self.action_shape, self.feature_dim,
							   self.hidden_dim, franka=True).to(self.device)
		self.critic = Critic(self.model_repr_dim, self.action_shape, self.feature_dim,
							 self.hidden_dim, franka=True).to(self.device)
		self.critic_target = Critic(self.model_repr_dim, self.action_shape,
									self.feature_dim, self.hidden_dim, franka=True).to(self.device)

		self.critic_target.load_state_dict(self.critic.state_dict())
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

		self.train()
		self.critic_target.train()
	
	def act(self, obs, step, eval_mode):
		obs_img = obs['imgs']
		obs_state = obs['state']
		obs_img = torch.as_tensor(obs_img, device=self.device)
		obs_state = torch.as_tensor(obs_state, device=self.device)

		h  = self.encoder(obs=obs_img)
		obs_trunk = self.actor.trunk_pass(h)
		obs = torch.cat([obs_trunk, obs_state], axis=-1)

		stddev = utils.schedule(self.stddev_schedule, step)
		dist = self.actor.final_out(obs, std=stddev)

		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample(clip=None)
			if step < self.num_expl_steps:
				action.uniform_(-1.0, 1.0)

		return action.cpu().numpy()

	def update_critic(self, obs, action, reward, discount, next_obs, step):
		metrics = dict()

		# action needs to be re-computed for the next state
		# passing through action requires splitting obs into [trunk(img), state]
		next_obs_h, next_obs_state = next_obs[:, :-4], next_obs[:, -4:]

		with torch.no_grad():
			stddev = utils.schedule(self.stddev_schedule, step)
			next_obs_trunk = self.actor.trunk_pass(next_obs_h)
			lowdim_next_obs = torch.cat([next_obs_trunk, next_obs_state], axis=-1)
			dist = self.actor.final_out(lowdim_next_obs, stddev)
			next_action = dist.sample(clip=self.stddev_clip)
			# to avoid truncation of the robot state, concatenate with actions
			next_action_cat = torch.cat([next_obs_state, next_action], axis=-1)
			# pass in image embeddings to critic where it will get trunked
			target_Q1, target_Q2 = self.critic_target(next_obs_h, next_action_cat)
			target_V = torch.min(target_Q1, target_Q2)
			target_Q = reward + (discount * target_V)

		# no need to trunk images, handled within critic
		obs_h, obs_state = obs[:, :-4], obs[:, -4:]
		action_cat = torch.cat([obs_state, action], axis=-1)
		Q1, Q2 = self.critic(obs_h, action_cat)

		critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

		# optimize encoder and critic
		self.encoder_opt.zero_grad(set_to_none=True)
		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_opt.step()
		self.encoder_opt.step()

		if self.use_tb:
			metrics['critic_target_q'] = target_Q.mean().item()
			metrics['critic_q1'] = Q1.mean().item()
			metrics['critic_q2'] = Q2.mean().item()
			metrics['critic_loss'] = critic_loss.item()
			
		return metrics

	def _transition_tuple(self, replay_iter, demo_iter=None, oversample_count=None):

		def _split_obs_state(batch):
			obs, action, reward, discount, next_obs, step_type, next_step_type = batch
			obs_img, obs_state = obs['imgs'], obs['state']
			next_obs_img, next_obs_state = next_obs['imgs'], next_obs['state']
			return utils.to_torch((obs_img, obs_state,
								   action,
								   reward,
								   discount,
								   next_obs_img, next_obs_state,
								   step_type,
								   next_step_type),
								   self.device)

		batch = _split_obs_state(next(replay_iter))

		if demo_iter is not None:
			num_offline = oversample_count
			num_online = batch[0].shape[0] - num_offline
			batch_d = _split_obs_state(next(demo_iter))

			new_batch = ()
			for x, y in zip(batch, batch_d):
				new_batch += (torch.cat((x[:num_online], y[:num_offline]), axis=0),)

			return new_batch
		else:
			return batch

	def update_actor(self, obs, obs_bc, obs_qfilter, action_bc, bc_regularize, step):
		metrics = dict()
		# do not trunk state, only the image
		obs_h, obs_state = obs[:, :-4], obs[:, -4:]
		obs_trunk = self.actor.trunk_pass(obs_h)
		lowdim_obs = torch.cat([obs_trunk, obs_state], axis=-1)
		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor.final_out(lowdim_obs, std=stddev)
		action = dist.sample(clip=self.stddev_clip)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		# to avoid truncation of the robot state, concatenate with actions
		action_cat = torch.cat([obs_state, action], axis=-1)
		Q1, Q2 = self.critic(obs_h, action_cat)
		Q = torch.min(Q1, Q2)

		# Compute bc weight
		if not bc_regularize:
			bc_weight = 0.0
		elif self.bc_weight_type == "linear":
			bc_weight = utils.schedule(self.bc_weight_schedule, step)
		elif self.bc_weight_type == "qfilter":
			"""
			Soft Q-filtering inspired from 			
			Nair, Ashvin, et al. "Overcoming exploration in reinforcement 
			learning with demonstrations." 2018 IEEE international 
			conference on robotics and automation (ICRA). IEEE, 2018.
			"""
			with torch.no_grad():
				stddev_qfil = 0.1
				obs_qfilter_h, obs_qfil_state = obs_qfilter[:, :-4], obs_qfilter[:, -4:]
				obs_qfil_trunk = self.actor_bc.trunk_pass(obs_qfilter_h)
				# trunk only images again for filter obs
				lowdim_qfil = torch.cat([obs_qfil_trunk, obs_qfil_state], axis=-1)
				dist_qf = self.actor_bc.final_out(lowdim_qfil, stddev_qfil)
				action_qf = dist_qf.mean
				# to avoid truncation of the robot state, concatenate with actions
				action_qf_cat = torch.cat([obs_qfil_state, action_qf], axis=-1)
				# Critic will trunk image embeddings
				Q1_qf, Q2_qf = self.critic(obs_qfilter_h.clone(), action_qf_cat)
				Q_qf = torch.min(Q1_qf, Q2_qf)
				bc_weight = (Q_qf>Q).float().mean().detach()

		actor_loss = - Q.mean() * (1-bc_weight)

		if bc_regularize:
			stddev = 0.1
			obs_bc_h, obs_bc_state = obs_bc[:, :-4], obs_bc[:, -4:]
			obs_bc_trunk = self.actor.trunk_pass(obs_bc_h)
			lowdim_bc_obs = torch.cat([obs_bc_trunk, obs_bc_state], axis=-1)
			dist_bc = self.actor.final_out(lowdim_bc_obs, stddev)
			log_prob_bc = dist_bc.log_prob(action_bc).sum(-1, keepdim=True)
			actor_loss += - log_prob_bc.mean()*bc_weight*0.03

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()
		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['actor_logprob'] = log_prob.mean().item()
			metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
			metrics['actor_q'] = Q.mean().item()
			if bc_regularize and self.bc_weight_type == "qfilter":
				metrics['actor_qf'] = Q_qf.mean().item()
			metrics['bc_weight'] = bc_weight
			metrics['regularized_rl_loss'] = -Q.mean().item()* (1-bc_weight)
			metrics['rl_loss'] = -Q.mean().item()
			if bc_regularize:
				metrics['regularized_bc_loss'] = - log_prob_bc.mean().item()*bc_weight*0.03
				metrics['bc_loss'] = - log_prob_bc.mean().item()*0.03
		
		return metrics

	def update(self, replay_iter, expert_replay_iter, step, bc_regularize=False):
		metrics = dict()

		if step % self.update_every_steps != 0:
			return metrics
		batch = self._transition_tuple(replay_iter)
		obs_img, obs_state, action, reward, discount, next_obs_img, next_obs_state, step_type, next_step_type = batch

		# augment
		obs_qfilter_img = self.aug(obs_img.clone().float())
		obs_img = self.aug(obs_img.float())
		next_obs_img = self.aug(next_obs_img.float())

		# encode
		h_obs = self.encoder(obs=obs_img)
		obs = torch.cat([h_obs, obs_state], axis=1)

		# concate embedded images and state for each set of obs
		with torch.no_grad():
			h_next_obs = self.encoder(obs=next_obs_img)
			next_obs = torch.cat([h_next_obs, next_obs_state], axis=1)

		if bc_regularize:
			batch_d = self._transition_tuple(expert_replay_iter)
			obs_img_bc, obs_state_bc, action_bc, _, _, _, _, _, _ = batch_d
			obs_img_bc = self.aug(obs_img_bc.float())
			obs_bc_h = self.encoder(obs=obs_img_bc)
			# Detach grads
			obs_bc_h = obs_bc_h.detach()
			obs_bc = torch.cat([obs_bc_h, obs_state_bc], axis=1)

			# encode
			if bc_regularize and self.bc_weight_type == "qfilter":
				obs_qfilter_h = self.encoder_bc(obs=obs_qfilter_img)
				obs_qfilter = torch.cat([obs_qfilter_h, obs_state], axis=1)
			else:
				obs_qfilter = None
		else:
			obs_qfilter = None
			obs_bc = None 
			action_bc = None

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()

		# update critic
		metrics.update(
			self.update_critic(obs, action, reward, discount, next_obs, step))

		# update actor
		metrics.update(self.update_actor(obs.detach(), obs_bc, obs_qfilter, action_bc, bc_regularize, step))

		# update critic target
		utils.soft_update_params(self.critic, self.critic_target,
								 self.critic_target_tau)

		return metrics

	def process_demos(self, demos):
		'''
		Helper function to preprocess flattened
		demos for reward calculation.
		NOTE: this assumes all demos are of length 100
		'''
		self.all_demo_imgs = []
		self.all_demo_states = []
		demo_imgs = demos['imgs']
		demo_states = demos['state']
		self.all_demo_imgs = [demo_imgs[i:i + 100] for i in range(0, len(demo_imgs), 100)]
		self.all_demo_states = [demo_states[i:i + 100] for i in range(0, len(demo_states), 100)]
		# self.all_demo_imgs = [self.all_demo_imgs[1]]
		# self.all_demo_states = [self.all_demo_states[1]]

	def ot_rewarder(self, observations, demos, step):
		if step % self.update_target_every == 0:
			if self.use_encoder:
				self.encoder_target.load_state_dict(self.encoder.state_dict())
			self.trunk_target.load_state_dict(self.actor.trunk.state_dict())
			self.target_updated = True

		obs_img = torch.tensor(np.stack([obs['imgs'] for obs in observations])).to(self.device).float()
		obs_state = torch.tensor(np.stack([obs['state'] for obs in observations])).to(self.device)

		obs_trunk = self.trunk_target(self.encoder_target(obs_img))
		obs_trunk = obs_trunk.detach()
		obs = torch.cat([obs_trunk, obs_state], axis=-1)
		scores_list = list()
		ot_rewards_list = list()
		for idx in range(len(self.all_demo_imgs)):
			exp_img = torch.tensor(self.all_demo_imgs[idx]).to(self.device).float()
			exp_trunk = self.trunk_target(self.encoder_target(exp_img))
			exp_trunk = exp_trunk.detach()
			exp_state = torch.tensor(self.all_demo_states[idx]).to(self.device)
			exp = torch.cat([exp_trunk, exp_state], axis=-1)

			if self.rewards == 'sinkhorn_cosine':
				cost_matrix = cosine_distance(
					obs, exp)  # Get cost matrix for samples using critic network.
				transport_plan = optimal_transport_plan(
					obs, exp, cost_matrix, method='sinkhorn',
					niter=100).float()  # Getting optimal coupling
				ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
					torch.mm(transport_plan,
							 cost_matrix.T)).detach().cpu().numpy()

			elif self.rewards == 'sinkhorn_euclidean':
				cost_matrix = euclidean_distance(
					obs, exp)  # Get cost matrix for samples using critic network.
				transport_plan = optimal_transport_plan(
					obs, exp, cost_matrix, method='sinkhorn',
					niter=100).float()  # Getting optimal coupling
				ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
					torch.mm(transport_plan,
							 cost_matrix.T)).detach().cpu().numpy()

			elif self.rewards == 'cosine':
				exp = torch.cat((exp, exp[-1].unsqueeze(0)))
				ot_rewards = -(1. - F.cosine_similarity(obs, exp))
				ot_rewards *= self.sinkhorn_rew_scale
				ot_rewards = ot_rewards.detach().cpu().numpy()

			elif self.rewards == 'euclidean':
				exp = torch.cat((exp, exp[-1].unsqueeze(0)))
				ot_rewards = -(obs - exp).norm(dim=1)
				ot_rewards *= self.sinkhorn_rew_scale
				ot_rewards = ot_rewards.detach().cpu().numpy()

			else:
				raise NotImplementedError()

			scores_list.append(np.sum(ot_rewards))
			ot_rewards_list.append(ot_rewards)

		closest_demo_index = np.argmax(scores_list)
		return ot_rewards_list[closest_demo_index]
