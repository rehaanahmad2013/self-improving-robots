from random import seed
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
import time, sys, select, termios
import copy
import env_loader
import hydra
import numpy as np
import torch
import utils
import functools
from dm_env import specs
from logger import Logger
from buffers.replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from agents import VICEFrankaAgent
from backend.timestep import StepType, ExtendedTimeStep
from networks import DiscrimVisionFranka

torch.backends.cudnn.benchmark = True

'''
Both forward and backward agents are going to be setup as VICEFrankaAgent.
The positive set for the forward agent are just the goal states, whereas for
the backward agent, the positive set is a broader distribution.
TODO: share the encoder between the forward and backward agent.
'''
def make_agent(obs_spec, action_spec, cfg, pos_dataset, skip_reward_computation=False):
    return VICEFrankaAgent(obs_spec=obs_spec,
                           action_shape=action_spec.shape,
                           device=cfg.device,
                           lr=cfg.lr,
                           feature_dim=cfg.feature_dim,
                           hidden_dim=cfg.hidden_dim,
                           critic_target_tau=cfg.critic_target_tau, 
                           reward_scale_factor=cfg.reward_scale_factor,
                           bc_reg_lambda=cfg.bc_reg_lambda,
                           use_tb=cfg.use_tb,
                           repr_dim=32*43*43, # output size of the convnet for 100x100 images
                           # REDQ settings
                           num_Q=cfg.num_Q,
                           utd_ratio=cfg.utd_ratio,
                           train_alpha=cfg.train_alpha,
                           target_entropy=cfg.target_entropy,
                           # VICE config
                           skip_reward_computation=skip_reward_computation,
                           share_encoder=cfg.share_encoder,
                           use_trunk=cfg.trunk,
                           mixup=cfg.mixup,
                           reward_type=cfg.reward_type,
                           gaussian_noise_coef=cfg.gaussian_noise_coef,
                           pos_dataset=pos_dataset,
                           # specify separately for the discriminator (can ignore state)
                           state_dim=cfg.disc_state_dim,
                           ignore_view=cfg.ignore_view,)

class Workspace:
    def __init__(self, cfg, work_dir=None):
        if work_dir is None:
            self.work_dir = Path.cwd()
            print(f'New workspace: {self.work_dir}')
        else:
            self.work_dir = work_dir

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        print('setup done!')

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0 # how many episodes have been run

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.train_env , self.eval_env, _, _, self.forward_demos, self.backward_demos = env_loader.make(self.cfg.env_name,
                                                                                                        action_repeat=self.cfg.action_repeat,
                                                                                                        height=self.cfg.height,
                                                                                                        width=self.cfg.width,
                                                                                                        num_frames=1)

        # ----------------- REFACTOR DEMONSTRATIONS + CREATE GOAL STATES -----------------
        print('initialized environments!')
        self.forward_demos = utils.refactor_demos_franka(self.forward_demos,
                                                         self.train_env._franka_transform_observation)
        self.backward_demos = utils.refactor_demos_franka(self.backward_demos,
                                                          self.train_env._franka_transform_observation)
        def _get_subtrajectories_as_goals(demos, segment_length, last=True):
            '''
            Get subtrajectories from a set of demos that can be used as goal sets for VICE.
            segment_length(int)
            last(bool): whether to use the last segment_length of the trajectory
            '''
            term_idxs = np.where(demos['dones'])[0]
            start_end_pairs = [(0, term_idxs[0])] + [(term_idxs[ix-1]+1, term_idxs[ix]) for ix in range(1, len(term_idxs))]
            if last:
                goal_idxs = np.concatenate([np.arange(end - segment_length, end + 1) for _, end in start_end_pairs])
            else:
                goal_idxs = np.concatenate([np.arange(start, start + segment_length + 1) for start, end in start_end_pairs])
            return {
                'images': np.array([demos['next_observations'][idx]['images'] for idx in goal_idxs]),
                'states': np.array([demos['next_observations'][idx]['state'] for idx in goal_idxs]),
            }

        # forward goals: last X states of forward demos
        # backward goals: all states / last X states of forward demos + last X states of backward demos
        self.forward_success_states = _get_subtrajectories_as_goals(self.forward_demos,
                                                                    segment_length=20,)
        self.backward_success_states = _get_subtrajectories_as_goals(self.forward_demos,
                                                                     segment_length=50,
                                                                     last=False,)
        self.addn_backward_success_states = _get_subtrajectories_as_goals(self.backward_demos,
                                                                          segment_length=20,)
        # combine to create the final set of backward goals
        self.backward_success_states['images'] = np.concatenate([self.backward_success_states['images'],
                                                                 self.addn_backward_success_states['images']])
        self.backward_success_states['states'] = np.concatenate([self.backward_success_states['states'],
                                                                 self.addn_backward_success_states['states']])
        print('initialized goal states!')

        # ----------------- CREATE REPLAY STORAGES -----------------
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))
        self.replay_storage_f = ReplayBufferStorage(data_specs,
                                                    self.work_dir / 'forward_buffer')

        self.forward_demo_buffer = ReplayBufferStorage(data_specs,
                                                       self.work_dir / 'forward_demo_buffer')

        self.replay_storage_b = ReplayBufferStorage(data_specs,
                                                    self.work_dir / 'backward_buffer')

        self.backward_demo_buffer = ReplayBufferStorage(data_specs,
                                                        self.work_dir / 'backward_demo_buffer')
        print('created replay storages!')

        # ----------------- CREATE AGENTS -----------------
        self.forward_agent = make_agent(self.train_env.observation_spec(),
                                        self.train_env.action_spec(),
                                        self.cfg.forward_agent,
                                        pos_dataset=self.forward_success_states,
                                        skip_reward_computation=self.cfg.nstep > 1,)

        self.backward_agent = make_agent(self.train_env.observation_spec(),
                                         self.train_env.action_spec(),
                                         self.cfg.backward_agent,
                                         pos_dataset=self.backward_success_states,
                                         skip_reward_computation=self.cfg.nstep > 1,)
        print('initialized agents!')

        # ----------------- CREATE BUFFER SAMPLERS -----------------
        forward_reward_callable = None
        backward_reward_callable = None
        # SLOWS DOWN TRAINING A LOT
        if self.cfg.nstep > 1:
            forward_reward_callable = lambda x: self.forward_agent.compute_reward({k: np.expand_dims(v, 0) for k,v in x.items()}, evald=True).cpu().numpy()[0]
            backward_reward_callable = lambda x: self.backward_agent.compute_reward({k: np.expand_dims(v, 0) for k,v in x.items()}, evald=True).cpu().numpy()[0]

        self.forward_loader = make_replay_loader(
            self.work_dir / 'forward_buffer', int(1e7),
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount,
            custom_reward_callable=forward_reward_callable,)

        # for oversampling the demos in the forward buffer
        self.forward_demo_loader = make_replay_loader(
            self.work_dir / 'forward_demo_buffer', int(1e6),
            self.cfg.vice_discriminator.batch_size // 2, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount,
            custom_reward_callable=forward_reward_callable,)

        self.backward_loader = make_replay_loader(
            self.work_dir / 'backward_buffer', int(1e7),
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount,
            custom_reward_callable=backward_reward_callable,)

        # for oversampling the demos in the backward buffer
        self.backward_demo_loader = make_replay_loader(
            self.work_dir / 'backward_demo_buffer', int(1e6),
            self.cfg.medal_discriminator.batch_size // 2, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount,
            custom_reward_callable=backward_reward_callable,)

        self._forward_iter, self._backward_iter, self._forward_demo_iter, self._backward_demo_iter = None, None, None, None
        print('created replay loaders!')

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None, franka=True)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None, franka=True)

    @property
    def backward_demo_iter(self):
        if self._backward_demo_iter is None:
            self._backward_demo_iter = iter(self.backward_demo_loader)
        return self._backward_demo_iter

    @property
    def forward_demo_iter(self):
        if self._forward_demo_iter is None:
            self._forward_demo_iter = iter(self.forward_demo_loader)
        return self._forward_demo_iter

    @property
    def forward_iter(self):
        if self._forward_iter is None:
            self._forward_iter = iter(self.forward_loader)
        return self._forward_iter

    @property
    def backward_iter(self):
        if self._backward_iter is None:
            self._backward_iter = iter(self.backward_loader)
        return self._backward_iter

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def eval(self):
        steps, episode, total_reward, episode_success = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            print(f'eval episode! {episode}')
            time_step = self.eval_env.reset()
            if self.cfg.save_video:
                self.video_recorder.init(self.eval_env)
            episode_step, completed_successfully = 0, 0
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.forward_agent):
                    action = self.forward_agent.act(time_step.observation,
                                                    uniform_action=False,
                                                    eval_mode=True)
                time_step = self.eval_env.step(action)
                if self.cfg.save_video:
                    self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                episode_step += 1
                steps += 1

            episode += 1
            if self.cfg.save_video:
                self.video_recorder.save(f'{self.global_frame}_ep_{episode}_.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('success_avg', episode_success / episode)
            log('episode_length', steps * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def switch_agent(self):
        '''If the agent is in the forward mode, switch to the backward mode.
        Determined by the self.cur_id.'''
        # agent in backward mode, switch to the forward agent
        if self.cur_id == 'backward':
            print('--- switching to forward agent ---')
            self.cur_id = 'forward'

        # agent in forward mode, switch to the backward agent
        elif self.cur_id == 'forward':
            print('--- switching to backward agent ---')
            self.cur_id = 'backward'

    def get_online_discriminator_metrics(self, ondata, disc_out_callable):
        metrics = dict()

        ondata['images'] = torch.from_numpy(np.stack(ondata['images'])).to(self.device).type(torch.cuda.FloatTensor)
        ondata['state'] = torch.from_numpy(np.stack(ondata['state'])).to(self.device).type(torch.cuda.FloatTensor)
        with torch.no_grad():
            reward, output = disc_out_callable(ondata, return_sig=True)
        on_labels = torch.zeros(ondata['images'].shape[0], 1).to(self.device)

        metrics['online_reward'] = reward.mean().item()
        metrics['online_disc_bce'] = torch.nn.BCELoss()(output, on_labels).item()
        metrics['online_disc_acc'] = ((output > 0.5) == on_labels).type(torch.float).mean().item()
        metrics['online_disc_prob'] = output.mean().item()

        return metrics

    def update_metric_keys(self, metrics, prefix):
        return {f'{prefix}_{k}': v for k, v in metrics.items()}

    def _on_new_time_step(self, time_step, seeding=False, end_episode=False):
        if not seeding:
            self.replay_buffer[self.cur_id].add(time_step, end_episode=end_episode)
            self.ondata[self.cur_id]['images'].append(time_step.observation['images'])
            self.ondata[self.cur_id]['state'].append(time_step.observation['state'])
        else:
            self.replay_buffer['forward'].add(time_step, end_episode=end_episode)
            self.replay_buffer['backward'].add(time_step, end_episode=end_episode)
            self.ondata['forward']['images'].append(time_step.observation['images'])
            self.ondata['forward']['state'].append(time_step.observation['state'])
            self.ondata['backward']['images'].append(time_step.observation['images'])
            self.ondata['backward']['state'].append(time_step.observation['state'])

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        forward_switch_length = utils.Every(self.cfg.forward_switch_length,
                                            self.cfg.action_repeat)
        backward_switch_length = utils.Every(self.cfg.backward_switch_length,
                                             self.cfg.action_repeat)
        joint_reset_every_step = utils.Every(self.cfg.joint_reset_frequency,
                                             self.cfg.action_repeat)
        '''MEDAL discriminator updates more frequently, to match different parts of the state distribution.
        VICE discriminator updates slowly, to regularize the discriminator.'''
        vice_disc_train_every_step = utils.Every(self.cfg.vice_discriminator.train_interval,
                                                 self.cfg.action_repeat)
        medal_disc_train_every_step = utils.Every(self.cfg.medal_discriminator.train_interval,
                                                  self.cfg.action_repeat)

        def _get_new_ondata_dict():
            return {'images':[], 'state':[]}

        # setup such that cur_id can determine what to use
        self.ondata = {'forward': _get_new_ondata_dict(), 'backward': _get_new_ondata_dict()}
        self.agent = {'forward': self.forward_agent, 'backward': self.backward_agent}
        self.replay_buffer = {'forward': self.replay_storage_f, 'backward': self.replay_storage_b}
        self.replay_iter = {'forward': self.forward_iter, 'backward': self.backward_iter}
        self.demo_iter = {'forward': self.forward_demo_iter, 'backward': self.backward_demo_iter}

        # add demos to the buffer only when starting the training
        if self.global_step == 0:
            print('--- adding demos to the buffer ---')
            self.forward_demo_buffer.add_offline_data_franka(self.forward_demos)
            # self.backward_demo_buffer.add_offline_data_franka(self.forward_demos)
            self.backward_demo_buffer.add_offline_data_franka(self.backward_demos)
            print('--- added demos to the buffers ---')

        self.cur_id = 'backward'
        self.switch_agent() # switch to forward agent

        print('--- starting training ---')
        time_step = self.train_env.reset()
        self._on_new_time_step(time_step, seeding=seed_until_step(self.global_step))

        if self.cfg.save_train_video:
            self.train_video_recorder.init(time_step.observation)

        episode_step, episode_reward = 0, 0
        # start dumping logs only when all the keys have been updated
        mlog_f, mlog_b, mlog_v, mlog_m = False, False, False, False
        while train_until_step(self.global_step):
            # update the agents
            if not seed_until_step(self.global_step):
                if self.cur_id == 'forward':
                    trans_tuple_demo = self.agent[self.cur_id].transition_tuple(self.demo_iter[self.cur_id]) if self.cfg.forward_agent.bc_reg_lambda != 0.0 else None
                if self.cur_id == 'backward':
                    trans_tuple_demo = self.agent[self.cur_id].transition_tuple(self.demo_iter[self.cur_id]) if self.cfg.backward_agent.bc_reg_lambda != 0.0 else None

                metrics = self.agent[self.cur_id].update(trans_tuple_fn=functools.partial(self.agent[self.cur_id].transition_tuple,
                                                                                          replay_iter=self.replay_iter[self.cur_id],
                                                                                          demo_iter=self.demo_iter[self.cur_id],
                                                                                          oversample_count=self.cfg.oversample_count),
                                                         step=self.global_step,
                                                         trans_tuple_demo=trans_tuple_demo,)

                self.logger.log_metrics(self.update_metric_keys(metrics, prefix=self.cur_id),
                                        self.global_frame, ty='train')
                if self.cur_id == 'forward':
                    mlog_f = True
                elif self.cur_id == 'backward':
                    mlog_b = True

                # update VICE discriminator
                if vice_disc_train_every_step(self.global_step):
                    print(f'{self.global_step}, training vice discriminator')
                    # evaluate discriminator predictions on the most recent online data where it has not been trained
                    if len(self.ondata['forward']['images']) > 1:
                        print(f"evaluating VICE discriminator on online data of size {len(self.ondata['forward']['images'])}")
                        metrics = self.get_online_discriminator_metrics(self.ondata['forward'], self.agent['forward'].compute_reward)
                        self.logger.log_metrics(self.update_metric_keys(metrics, prefix='vice'),
                                                self.global_frame, ty='train')
                        self.ondata['forward'] = _get_new_ondata_dict()
                        # update flag here to ensure all metrics have been seen once
                        mlog_v = True

                    for _ in range(self.cfg.vice_discriminator.train_steps_per_iteration):
                        metrics = self.forward_agent.update_discriminator(self.forward_iter)
                    self.logger.log_metrics(self.update_metric_keys(metrics, prefix='vice'),
                                            self.global_frame, ty='train')

                # update MEDAL discriminator
                if medal_disc_train_every_step(self.global_step):
                    print(f'{self.global_step}, training MEDAL discriminator')
                    # evaluate discriminator predictions on the most recent online data where it has not been trained
                    if len(self.ondata['backward']['images']) > 1:
                        print(f"evaluating MEDAL discriminator on online data of size {len(self.ondata['backward']['images'])}")
                        metrics = self.get_online_discriminator_metrics(self.ondata['backward'], self.agent['backward'].compute_reward)
                        self.logger.log_metrics(self.update_metric_keys(metrics, prefix='medal'),
                                                self.global_frame, ty='train')
                        self.ondata['backward'] = _get_new_ondata_dict()
                        mlog_m = True

                    for _ in range(self.cfg.medal_discriminator.train_steps_per_iteration):
                        metrics = self.backward_agent.update_discriminator(self.backward_iter)
                    self.logger.log_metrics(self.update_metric_keys(metrics, prefix='medal'),
                                            self.global_frame, ty='train')

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent[self.cur_id]):
                action = self.agent[self.cur_id].act(time_step.observation,
                                                     uniform_action=seed_until_step(self.global_step),
                                                     eval_mode=False)

            # take env step
            time_step = self.train_env.step(action)
            switch_agent_flag = (self.cur_id == 'forward' and forward_switch_length(episode_step + 1)) or \
                                (self.cur_id == 'backward' and backward_switch_length(episode_step + 1))
            self._on_new_time_step(time_step, 
                                   seeding=seed_until_step(self.global_step),
                                   end_episode=switch_agent_flag)

            # update online metrics
            episode_reward += time_step.reward
            if self.cfg.save_train_video:
                self.train_video_recorder.record(time_step.observation)
            self._global_step += 1                
            episode_step += 1

            if switch_agent_flag:
                print(f'step: {self.global_step}')
                # check to ensure that env is always returning a mid type state
                assert time_step.step_type == StepType.MID

                # pretend episode ends when the policy switches
                self._global_episode += 1
                if self.cfg.save_train_video:
                    self.train_video_recorder.save(f'{self.global_frame}.mp4')

                self.switch_agent()

                manual_reset_done = False
                if self.cur_id == 'forward':
                    termios.tcflush(sys.stdin, termios.TCIOFLUSH)
                    print(f'To manually reset, enter anything.')
                    i, _, _ = select.select( [sys.stdin], [], [], 3)
                    if (i):
                        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
                        print(f'Manual reset mode.')
                        print('Resetting the arm.')
                        self.train_env.reset()
                        manual_reset_done = True
                        input('Press anything to continue.')

                # joint reset/manual reset allowed only when switching agent
                if not manual_reset_done and (joint_reset_every_step(self.global_step) or seed_until_step(self.global_step)):
                    print('automatic resetting of the arm.')
                    self.train_env.reset()

                print('Continuing autonomous training.')
                # NOTE: a manual intervention might have happened, get observation again from the environment.
                curr_obs = self.train_env._franka_transform_observation(self.train_env.get_observation())
                time_step = ExtendedTimeStep(
                    observation=curr_obs,
                    step_type=StepType.FIRST,
                    action=np.zeros_like(time_step.action),
                    reward=0.0,
                    discount=1.0,
                )
                time_step = self.train_env._modify_obs_dtype(time_step)
                self._on_new_time_step(time_step, seeding=seed_until_step(self.global_step))

                # wait until all the metrics schema is populated
                if mlog_f and mlog_b and mlog_v and mlog_m:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('forward_buffer_size', len(self.replay_storage_f))
                        log('backward_buffer_size', len(self.replay_storage_b))
                        log('forward_demo_buffer', len(self.forward_demo_buffer))
                        log('backward_demo_buffer', len(self.backward_demo_buffer))
                        log('step', self.global_step)

                # save snapshot
                if self.cfg.save_snapshot and (self.global_episode % 10 == 0):
                    self.save_snapshot(step=self.global_step)

                episode_step, episode_reward = 0, 0

    def save_snapshot(self, step):
        # Add in info about the current timestep
        snapshot = self.work_dir / f'snapshot_{step}.pt'
        keys_to_save = ['forward_agent', 'backward_agent', 'timer', '_global_step', '_global_episode']
        payload = {k: copy.copy(self.__dict__[k]) for k in keys_to_save}

        # TODO: spectral norm throws an error when saving, check if restoration works fine for the discriminator
        discrim_dict = {'vice_discriminator': payload['forward_agent'].__dict__['discriminator'].state_dict(), 
                        'vice_discrim_opt': payload['forward_agent'].__dict__['discrim_opt'].state_dict(),
                        'medal_discriminator': payload['backward_agent'].__dict__['discriminator'].state_dict(),
                        'medal_discrim_opt': payload['backward_agent'].__dict__['discrim_opt'].state_dict(),}

        del payload['forward_agent'].__dict__['discriminator']
        del payload['forward_agent'].__dict__['discrim_opt']
        del payload['backward_agent'].__dict__['discriminator']
        del payload['backward_agent'].__dict__['discrim_opt']

        payload.update(discrim_dict)
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, step, cfg, work_dir=None):
        if work_dir is None:
            snapshot = self.work_dir / f'snapshot_{step}.pt'
        else:
            snapshot = work_dir / f'snapshot_{step}.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)

        # ensure agent is loaded before discriminator stuff
        for k, v in sorted(payload.items()):
            if 'discrim' in k:
                continue
            else:
                self.__dict__[k] = v

        print(f'restoring discriminator and optimizer from {snapshot}')
        self.forward_agent.discriminator = DiscrimVisionFranka(obs_shape=self.forward_agent.obs_shape,
                                                               feature_dim=self.forward_agent.feature_dim,
                                                               repr_dim=32*43*43,
                                                               create_inp_encoder=not bool(self.forward_agent.share_encoder),
                                                               ignore_view=cfg.forward_agent.ignore_view,
                                                               state_dim=cfg.forward_agent.disc_state_dim).to(self.forward_agent.device)
        self.forward_agent.discrim_opt = torch.optim.Adam(self.forward_agent.discriminator.parameters(), lr=3e-4)

        self.backward_agent.discriminator = DiscrimVisionFranka(obs_shape=self.backward_agent.obs_shape,
                                                                feature_dim=self.backward_agent.feature_dim,
                                                                repr_dim=32*43*43,
                                                                create_inp_encoder=not bool(self.backward_agent.share_encoder),
                                                                ignore_view=cfg.backward_agent.ignore_view,
                                                                state_dim=cfg.backward_agent.disc_state_dim).to(self.backward_agent.device)
        self.backward_agent.discrim_opt = torch.optim.Adam(self.backward_agent.discriminator.parameters(), lr=3e-4)

        self.forward_agent.discriminator.load_state_dict(payload['vice_discriminator'])
        self.forward_agent.discrim_opt.load_state_dict(payload['vice_discrim_opt'])
        self.backward_agent.discriminator.load_state_dict(payload['medal_discriminator'])
        self.backward_agent.discrim_opt.load_state_dict(payload['medal_discrim_opt'])

@hydra.main(config_path='../cfgs', config_name='medal_franka')
def main(cfg):
    if cfg.mode == 'train':
        work_dir_restore = Path(cfg.work_dir_restore) if cfg.work_dir_restore != 'None' else None
        workspace = Workspace(cfg, work_dir=work_dir_restore)
        if work_dir_restore is not None:
            workspace.load_snapshot(step=cfg.restore_snapshot_idx,
                                    cfg=cfg,
                                    work_dir=work_dir_restore)
        workspace.train()

    elif cfg.mode == 'eval':
        print(f'evaluating the agent')
        cfg.use_tb = False
        workspace = Workspace(cfg)
        workspace.load_snapshot(step=cfg.eval_checkpoint_idx,
                                cfg=cfg,
                                work_dir=Path(cfg.eval_dir))
        workspace.eval()

if __name__ == '__main__':
    main()
