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
from agents import VICEAgent
from backend.timestep import StepType, ExtendedTimeStep

torch.backends.cudnn.benchmark = True

'''
Both forward and backward agents are going to be setup as VICEFrankaAgent.
The positive set for the forward agent are just the goal states, whereas for
the backward agent, the positive set are all the demonstration states.
TODO: share the encoder between the forward and backward agent.
'''
def make_agent(obs_spec, action_spec, cfg, pos_dataset):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    
    return VICEAgent(obs_shape=cfg.obs_shape,
                     action_shape=cfg.action_shape,
                     device=cfg.device,
                     lr=cfg.lr,
                     feature_dim=cfg.feature_dim,
                     hidden_dim=cfg.hidden_dim,
                     critic_target_tau=cfg.critic_target_tau, 
                     reward_scale_factor=cfg.reward_scale_factor,
                     bc_reg_lambda=cfg.bc_reg_lambda,
                     use_tb=cfg.use_tb,
                     # REDQ settings  
                     num_Q=cfg.num_Q,
                     utd_ratio=cfg.utd_ratio,
                     train_alpha=cfg.train_alpha,
                     target_entropy=cfg.target_entropy,
                     # VICE config
                     share_encoder=cfg.share_encoder,
                     use_trunk=cfg.trunk,
                     mixup=cfg.mixup,
                     reward_type=cfg.reward_type,
                     spectral_norm=cfg.spectral_norm,
                     gaussian_noise_coef=cfg.gaussian_noise_coef,
                     pos_dataset=pos_dataset,)

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

        self.forward_agent = make_agent(self.train_env.observation_spec(),
                                        self.train_env.action_spec(),
                                        self.cfg.forward_agent,
                                        pos_dataset=self.forward_success_states)
        self.backward_agent = make_agent(self.train_env.observation_spec(),
                                         self.train_env.action_spec(),
                                         self.cfg.backward_agent,
                                         pos_dataset=self.backward_success_states)

        self.timer = utils.Timer()
        self._global_step = 0
        self.prev_step = 0
        self._global_episode = 0 # how many episodes have been run

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.train_env , self.eval_env, _, _, self.forward_demos, self.backward_demos = env_loader.make(self.cfg.env_name,
                                                                                                        action_repeat=self.cfg.action_repeat,
                                                                                                        reward_type='sparse',
                                                                                                        height=self.cfg.height,
                                                                                                        width=self.cfg.width,
                                                                                                        num_frames=1)

        if 'rewards' in self.forward_demos.keys():
            goal_keys = np.where(self.forward_demos['rewards'] == 1.)[0]
        else:
            goal_width = 10 # last X states of trajectory as goal states
            goal_keys_end = np.where(self.forward_demos['terminals'])[0]
            goal_keys_start = goal_keys_end - goal_width
            goal_keys = np.concatenate([np.arange(start, end) for start, end in zip(goal_keys_start, goal_keys_end)], axis=0)

        self.forward_success_states = self.forward_demos['observations'][goal_keys].copy()
        self.forward_success_states = self.forward_success_states.transpose(0, 3, 1, 2)
        self.forward_success_states = torch.from_numpy(self.forward_success_states)

        self.backward_success_states = self.forward_demos['observations'].copy()
        self.backward_success_states = self.backward_success_states.transpose(0, 3, 1, 2) 
        self.backward_success_states = torch.from_numpy(self.backward_success_states)

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage_f = ReplayBufferStorage(data_specs,
                                                    self.work_dir / 'forward_buffer',)

        self.forward_demo_buffer = ReplayBufferStorage(data_specs,
                                                       self.work_dir / 'forward_demo_buffer',)

        self.replay_storage_b = ReplayBufferStorage(data_specs,
                                                    self.work_dir / 'backward_buffer',)

        # demo data for the backward policy
        self.backward_demo_buffer = ReplayBufferStorage(data_specs,
                                                        self.work_dir / 'backward_demo_buffer',)

        self.forward_loader = make_replay_loader(
            self.work_dir / 'forward_buffer', int(1e7),
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount,)

        # for oversampling the demos in the forward buffer
        self.forward_demo_loader = make_replay_loader(
            self.work_dir / 'forward_demo_buffer', int(1e6),
            self.cfg.vice_discriminator.batch_size // 2, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount,)

        self.backward_loader = make_replay_loader(
            self.work_dir / 'backward_buffer', int(1e7),
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount,)

        # for oversampling the demos in the backward buffer
        self.backward_demo_loader = make_replay_loader(
            self.work_dir / 'backward_demo_buffer', int(1e6),
            self.cfg.medal_discriminator.batch_size // 2, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount,)

        self._forward_iter, self._backward_iter, self._forward_demo_iter, self._backward_demo_iter = None, None, None, None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None,)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None,)

        # recording metrics for EARL
        np.save(self.work_dir / 'eval_interval.npy', self.cfg.eval_every_frames)
        try:
            self.deployed_policy_eval = np.load(self.work_dir / 'deployed_eval.npy').tolist()
        except:
            self.deployed_policy_eval = []

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

    def eval(self, eval_agent):
        steps, episode, total_reward, episode_success = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            episode_step, completed_successfully = 0, 0
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(eval_agent):
                    action = eval_agent.act(time_step.observation.astype("float32"),
                                            uniform_action=False,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)

                if hasattr(self.eval_env, 'is_successful') and self.eval_env.is_successful():
                    completed_successfully = 1

                total_reward += time_step.reward
                episode_step += 1
                steps += 1

            episode += 1
            episode_success += completed_successfully
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('success_avg', episode_success / episode)
            log('episode_length', steps * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

        # EARL deployed policy evaluation
        self.deployed_policy_eval.append(episode_success / episode)
        np.save(self.work_dir / 'deployed_eval.npy', self.deployed_policy_eval)

    def switch_agent(self):
        '''
        If the agent is in the forward mode, switch to the backward mode.
        Determined by the self.cur_id.
        '''
        # agent in backward mode, switch to the forward agent
        if self.cur_id == 'backward':
            print('--- switching to forward agent ---')
            self.cur_id = 'forward'

        # agent in forward mode, switch to the backward agent
        elif self.cur_id == 'forward':
            print('--- switching to backward agent ---')
            self.cur_id = 'backward'

    def should_switch_step(self, glob_step, envreward):
        if (glob_step - self.prev_step) == self.cfg.policy_switch_frequency:
            self.prev_step = glob_step
            return True

        if (self.cur_id == 'forward') and (self.cfg.early_terminate) and (envreward == 1.0):
            self.prev_step = glob_step
            return True

        return False

    def get_online_discriminator_metrics(self, ondata, disc_out_callable):
        metrics = dict()

        ondata= torch.from_numpy(np.stack(ondata)).to(self.device).type(torch.cuda.FloatTensor)
        with torch.no_grad():
            _, output = disc_out_callable(ondata, return_sig=True)
        on_labels = torch.zeros(ondata.shape[0], 1).to(self.device)

        metrics['online_disc_bce'] = torch.nn.BCELoss()(output, on_labels).item()
        metrics['online_disc_acc'] = ((output > 0.5) == on_labels).type(torch.float).mean().item()
        metrics['online_disc_prob'] = output.mean().item()

        return metrics

    def update_metric_keys(self, metrics, prefix):
        return {f'{prefix}_{k}': v for k, v in metrics.items()}

    def train(self):
        # predicates
        # print("test")
        # self.forward_demo_buffer.convert_state_to_vision("/iris/u/rehaan/reset_free_rl/earl_benchmark/demonstrations/tabletopWideFwd/demo_data.pkl", self.eval_env)
        # exit()
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        switch_agent_every_step = utils.Every(self.cfg.policy_switch_frequency,
                                              self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        manual_reset_every_step = utils.Every(self.cfg.manual_reset_frequency,
                                              self.cfg.action_repeat)
        '''
        MEDAL discriminator updates more frequently, to match different parts of the state distribution.
        VICE discriminator updates slowly, to regularize the discriminator.
        '''
        vice_disc_train_every_step = utils.Every(self.cfg.vice_discriminator.train_interval,
                                                 self.cfg.action_repeat)
        medal_disc_train_every_step = utils.Every(self.cfg.medal_discriminator.train_interval,
                                                  self.cfg.action_repeat)
        # setup such that cur_id can determine what to use
        self.ondata = {'forward': [], 'backward': []}
        self.agent = {'forward': self.forward_agent, 'backward': self.backward_agent}
        self.replay_buffer = {'forward': self.replay_storage_f, 'backward': self.replay_storage_b}
        self.demo_replay_buffer = {'forward': self.forward_demo_buffer, 'backward': self.backward_demo_buffer}
        self.replay_iter = {'forward': self.forward_iter, 'backward': self.backward_iter}
        self.demo_iter = {'forward': self.forward_demo_iter, 'backward': self.backward_demo_iter}

        # add demos to the buffer
        dummy_action = np.zeros(self.train_env.action_spec().shape, dtype=np.float32)
        self.forward_demo_buffer.add_offline_vision_data(self.forward_demos,
                                                         dummy_action,
                                                         self.train_env,
                                                         self.cfg.frame_stack,
                                                         self.cfg.height, self.cfg.width,
                                                         inf_bootstrap=True,)

        # backward buffer gets both forward/backward demos
        self.backward_demo_buffer.add_offline_vision_data(self.forward_demos,
                                                          dummy_action,
                                                          self.train_env,
                                                          self.cfg.frame_stack,
                                                          self.cfg.height, self.cfg.width,
                                                          inf_bootstrap=True,)
        if self.backward_demos is not None:
            self.backward_demo_buffer.add_offline_vision_data(self.backward_demos,
                                                              dummy_action,
                                                              self.train_env,
                                                              self.cfg.frame_stack,
                                                              self.cfg.height, self.cfg.width,
                                                              inf_bootstrap=True,)

        self.cur_id = 'backward'
        self.switch_agent()

        time_step = self.train_env.reset()
        # add seed frames to both buffers
        # NOTE: the run might be restoring from an already trained checkpoint
        if not seed_until_step(self.global_step):
            self.replay_buffer[self.cur_id].add(time_step)
            self.ondata[self.cur_id].append(time_step.observation)
        else:
            self.replay_buffer['forward'].add(time_step)
            self.replay_buffer['backward'].add(time_step)
            self.ondata['forward'].append(time_step.observation)
            self.ondata['backward'].append(time_step.observation)

        if self.cfg.save_train_video:
            self.train_video_recorder.init(time_step.observation)

        if (self.cfg.early_terminate) and (self.cfg.gt_reward == False):
            print("Invalid Configuration: early terminate is True with gt_reward as False")

        metrics = None
        episode_step, episode_reward = 0, 0
        # track whether logs have been updated with respective keys
        mlog_f, mlog_b, mlog_v, mlog_m = False, False, False, False
        while train_until_step(self.global_step):
            # update the agents
            if not seed_until_step(self.global_step):
                if self.cur_id == 'forward':
                    trans_tuple_demo = self.agent[self.cur_id].transition_tuple(self.demo_iter[self.cur_id], gt_reward=self.cfg.gt_reward) if self.cfg.forward_agent.bc_reg_lambda != 0.0 else None
                if self.cur_id == 'backward':
                    trans_tuple_demo = self.agent[self.cur_id].transition_tuple(self.demo_iter[self.cur_id], gt_reward=False) if self.cfg.backward_agent.bc_reg_lambda != 0.0 else None

                metrics = self.agent[self.cur_id].update(trans_tuple_fn=functools.partial(self.agent[self.cur_id].transition_tuple,
                                                                                          replay_iter=self.replay_iter[self.cur_id],
                                                                                          demo_iter=self.demo_iter[self.cur_id],
                                                                                          oversample_count=self.cfg.oversample_count,
                                                                                          gt_reward=(self.cfg.gt_reward and self.cur_id=='forward'),
                                                                                          online_buf_len=len(self.replay_buffer[self.cur_id]),
                                                                                          offline_buf_len=len(self.demo_replay_buffer[self.cur_id])),
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
                    print(f'{self.global_step}, training VICE discriminator')
                    # evaluate discriminator predictions on the most recent online data where it has not been trained
                    if len(self.ondata['forward']) > 1:
                        print(f"evaluating VICE discriminator on online data of size {len(self.ondata['forward'])}")
                        metrics = self.get_online_discriminator_metrics(self.ondata['forward'], self.agent['forward'].compute_reward)
                        self.logger.log_metrics(self.update_metric_keys(metrics, prefix='vice'),
                                                self.global_frame, ty='train')
                        self.ondata['forward'] = []

                    for _ in range(self.cfg.vice_discriminator.train_steps_per_iteration):
                        metrics = self.agent['forward'].update_discriminator(self.replay_iter['forward'])
                    self.logger.log_metrics(self.update_metric_keys(metrics, prefix='vice'),
                                            self.global_frame, ty='train')
                    mlog_v = True

                # update MEDAL discriminator
                if medal_disc_train_every_step(self.global_step):
                    print(f'{self.global_step}, training MEDAL discriminator')
                    # evaluate discriminator predictions on the most recent online data where it has not been trained
                    if len(self.ondata['backward']) > 1:
                        print(f"evaluating MEDAL discriminator on online data of size {len(self.ondata['backward'])}")
                        metrics = self.get_online_discriminator_metrics(self.ondata['backward'], self.agent['backward'].compute_reward)
                        self.logger.log_metrics(self.update_metric_keys(metrics, prefix='medal'),
                                                self.global_frame, ty='train')
                        self.ondata['backward'] = []

                    for _ in range(self.cfg.medal_discriminator.train_steps_per_iteration):
                        metrics = self.agent['backward'].update_discriminator(self.replay_iter['backward'])
                    self.logger.log_metrics(self.update_metric_keys(metrics, prefix='medal'),
                                            self.global_frame, ty='train')
                    mlog_m = True

            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval(self.agent['forward'])

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent[self.cur_id]):
                action = self.agent[self.cur_id].act(time_step.observation,
                                                     uniform_action=seed_until_step(self.global_step),
                                                     eval_mode=False)

            # take env step
            time_step = self.train_env.step(action)

            # collect online data for replay buffer and online disc evaluation
            if not seed_until_step(self.global_step):
                self.replay_buffer[self.cur_id].add(time_step, end_episode=switch_agent_every_step(self.global_step + 1))    
                self.ondata[self.cur_id].append(time_step.observation)
            else:
                # add the seed frames to both the buffers, it is a uniform policy
                self.replay_buffer['forward'].add(time_step, end_episode=switch_agent_every_step(self.global_step + 1))
                self.replay_buffer['backward'].add(time_step, end_episode=switch_agent_every_step(self.global_step + 1))
                self.ondata['forward'].append(time_step.observation)
                self.ondata['backward'].append(time_step.observation)

            # update online metrics
            episode_reward += time_step.reward
            if self.cfg.save_train_video:
                self.train_video_recorder.record(time_step.observation)
            self._global_step += 1                
            episode_step += 1

            if self.should_switch_step(self.global_step, time_step.reward) or manual_reset_every_step(self.global_step):
                print('Soft Reset, step: ', self.global_step)
                print("TIME STEP REWARD: " + str(time_step.reward))
                # check to ensure that franka is always returning a mid type state
                assert time_step.step_type == StepType.MID

                # pretend episode ends when the policy switches
                self._global_episode += 1
                if self.cfg.save_train_video:
                    self.train_video_recorder.save(f'{self.global_frame}.mp4')

                self.switch_agent()

                # this ensures that observation is corrected for the next agent
                if seed_until_step(self.global_step) or manual_reset_every_step(self.global_step):
                    print('---Manual Reset---')
                    print('--- manually switching to forward agent ---')
                    self.cur_id = 'forward'
                    time_step = self.train_env.reset()
                else:
                    print('Continuing autonomous training.')
                    # mark the start of the new episode for the new agent
                    # FIXME: this should be in the wrapper somewhere
                    curr_obs = self.train_env._extract_pixels()
                    curr_obs = np.concatenate([curr_obs for _ in range(self.cfg.frame_stack)], axis=0)
                    time_step = ExtendedTimeStep(
                        observation=curr_obs,
                        step_type=StepType.FIRST,
                        action=np.zeros_like(time_step.action),
                        reward=0.0,
                        discount=1.0,
                    )

                if not seed_until_step(self.global_step):
                    self.replay_buffer[self.cur_id].add(time_step)
                    self.ondata[self.cur_id].append(time_step.observation)
                else:
                    self.replay_buffer['forward'].add(time_step)
                    self.replay_buffer['backward'].add(time_step)
                    self.ondata['forward'].append(time_step.observation)
                    self.ondata['backward'].append(time_step.observation)

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
                if self.cfg.save_snapshot and (self.global_episode % 5 == 0):
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

        # ensure agent is loaded before discrim
        for k, v in sorted(payload.items()):
            if 'discrim' in k:
                continue
            else:
                self.__dict__[k] = v

        print(f'restoring discriminator and optimizer from {snapshot}')
        # TODO: if this is enough or do we need to recreate the networks/optimizers
        self.forward_agent.discriminator.load_state_dict(payload['vice_discriminator'])
        self.forward_agent.discrim_opt.load_state_dict(payload['vice_discrim_opt'])
        self.backward_agent.discriminator.load_state_dict(payload['medal_discriminator'])
        self.backward_agent.discrim_opt.load_state_dict(payload['medal_discrim_opt'])

@hydra.main(config_path='cfgs', config_name='medalplusplus')
def main(cfg):
    work_dir_t = None
    workspace = Workspace(cfg, work_dir=work_dir_t)

    if not cfg.eval_mode:
        # TODO: load checkpoint here if resuming a training run
        # workspace.load_snapshot(step=5000, cfg=cfg, work_dir=work_dir_t)
        workspace.train()
    else:
        print(f'\n eval!')
        work_dir = Path(cfg.eval_dir)
        workspace.load_snapshot(step=cfg.eval_checkpoint_idx, cfg=cfg, work_dir=work_dir)
        workspace.eval()


if __name__ == '__main__':
    main()
