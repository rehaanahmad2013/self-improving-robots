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
				 use_tb, bc_reg_lambda=0.0, repr_dim=None):

		self.obs_shape = obs_shape
		self.action_shape = action_shape
		self.lr = lr
		self.feature_dim = feature_dim
		self.hidden_dim = hidden_dim
		self.device = device
		self.critic_target_tau = critic_target_tau
		self.reward_scale_factor = reward_scale_factor
		self.use_tb = use_tb

		# Changed log_std_bounds from [-10, 2] -> [-20, 2]
		self.log_std_bounds = [-20, 2]
		# Changed self.init_temperature to 1.0
		self.init_temperature = 1.0
		self.bc_reg_lambda = bc_reg_lambda
		self.repr_dim = repr_dim

		# models
		self.encoder = Encoder(obs_shape).to(device)
		# overwrite hard-coded representation dim for convnet
		self.encoder.repr_dim = repr_dim if repr_dim else self.encoder.repr_dim
		self.model_repr_dim = self.encoder.repr_dim

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
		self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
		# data augmentation
		self.aug = RandomShiftsAug(pad=4)

		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.training = True
		self.encoder.train()
		self.actor.train()
		self.critic.train()
		self.critic_target.train()

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def train(self, training=True):
		self.training = training
		self.encoder.train(training)
		self.actor.train(training)
		self.critic.train(training)

	def act(self, obs, uniform_action=False, eval_mode=False):
		obs = torch.as_tensor(obs, device=self.device)
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
		self.encoder_opt.zero_grad(set_to_none=True)
		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_opt.step()
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
		self.encoder_opt.zero_grad(set_to_none=True)

		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss_total.backward()
		self.critic_opt.step()

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

		self.discriminator = DiscrimVisionAction(obs_shape=self.obs_shape,
													action_dim=self.action_shape[0],
													feature_dim=self.feature_dim,
													hidden_dim=discrim_hidden_size,
													create_inp_encoder=not bool(self.share_encoder),
													use_spectral_norm=self.spectral_norm,
													use_trunk=self.use_trunk,).to(self.device)
	
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

		# TODO: action should still obey the bounds after adding gaussian noise
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
					_, output = self.compute_reward(self.discrim_val_data['observations'], self.discrim_val_data['actions'], return_sig=True, evald=True)
					val_labels = torch.ones(self.discrim_val_data['observations'].shape[0], 1).to(self.device)
					metrics['val_loss'] = loss(output, val_labels).item()
					metrics['val_acc'] = ((output > 0.5) == val_labels).type(torch.float).mean().item()

		return metrics

	def compute_reward(self, obs, action, return_sig=False, evald=False):
		if evald and type(obs) is np.ndarray:
				obs = torch.from_numpy(obs).to(self.device)
				action = torch.from_numpy(action).to(self.device)
		if self.share_encoder:
			obs = self.encoder(obs)
		sig_term = torch.sigmoid(self.discriminator(obs, action))
	
		if self.reward_type == 'logd':
			actual_reward = torch.log(torch.minimum(sig_term + self.eps, torch.tensor(1.)))
		else:
			actual_reward = -torch.log(1 - sig_term + self.eps)
		
		if not return_sig:
			return actual_reward
		else:
			return actual_reward, sig_term

	def transition_tuple(self, replay_iter, demo_iter=None, oversample_count=None, online_buf_len=None, offline_buf_len=None):
		obs, action, reward, discount, next_obs, step_type, next_step_type = super().transition_tuple(replay_iter,
																									  demo_iter=demo_iter,
																									  oversample_count=oversample_count,
																									  online_buf_len=online_buf_len,
																									  offline_buf_len=offline_buf_len)
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

		self.discriminator = DiscrimVision(obs_shape=self.obs_shape,
											feature_dim=self.feature_dim,
											hidden_dim=self.discrim_hidden_size,
											repr_dim=self.model_repr_dim,
											create_inp_encoder=not bool(self.share_encoder),
											use_spectral_norm=self.spectral_norm,
											use_trunk=self.use_trunk,).to(self.device)
		
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
					_, output = self.compute_reward(self.discrim_val_data['observations'], return_sig=True, evald=True)
					val_labels = torch.ones(self.discrim_val_data['observations'].shape[0], 1).to(self.device)
					metrics['val_loss'] = loss(output, val_labels).item()
					metrics['val_acc'] = ((output > 0.5) == val_labels).type(torch.float).mean().item()

		return metrics

	def compute_reward(self, obs, action=None, return_sig=False, evald=False):
		del action
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

	def transition_tuple(self, replay_iter, demo_iter=None, oversample_count=None, online_buf_len=None, offline_buf_len=None):
		obs, action, reward, discount, next_obs, step_type, next_step_type = super().transition_tuple(replay_iter,
																									  demo_iter=demo_iter,
																									  oversample_count=oversample_count,
																									  online_buf_len=online_buf_len,
																									  offline_buf_len=offline_buf_len)
		with torch.no_grad():
			VICE_reward = self.compute_reward(next_obs, None)
		return (obs, action, VICE_reward, discount, next_obs, step_type, next_step_type)

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

	def transition_tuple(self, replay_iter, demo_iter=None, oversample_count=None):
		if self._skip_reward_computation:
			return super().transition_tuple(replay_iter, demo_iter, oversample_count)

		obs_img, obs_state, action, env_reward, discount, next_obs_img, next_obs_state, step_type, next_step_type = super().transition_tuple(replay_iter,
																																    demo_iter=demo_iter,
																																		    oversample_count=oversample_count)
		
		return (obs_img, obs_state, action, env_reward, discount, next_obs_img, next_obs_state, step_type, next_step_type)

		# VICE reward should be based on the next observation
		next_obs_dict = {'images': next_obs_img, 'state': next_obs_state}
		with torch.no_grad():
			VICE_reward = self.compute_reward(next_obs_dict)
		return (obs_img, obs_state, action, VICE_reward, discount, next_obs_img, next_obs_state, step_type, next_step_type)
