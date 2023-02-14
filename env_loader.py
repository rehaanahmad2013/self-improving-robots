import earl_benchmark
import numpy as np
import os
from backend.wrappers import (
	ActionRepeatWrapper,
	ObsActionDTypeWrapper,
	ExtendedTimeStepWrapper,
	ActionScaleWrapper,
	DMEnvFromGymWrapper,
	GoalVisionWrapper,
	FrankaDMEnvFromGymWrapper,
	FrankaObsActionDTypeWrapper,
)
import hydra

def make(name, action_repeat, reward_type='sparse', height=84, width=84, num_frames=1):
	# setup franka environment
	if 'franka' in name:
		from robot_env import RobotEnv

		# read in demos for cube
		forward_demos = backward_demos = None
		DoF = 3
		if 'cubepickup' in name:
			episodic_path_length = 200
			forward_path = f'/iris/u/architsh/code/misc_data/franka_demos/cube_pickup/forward/demos.npz'
			forward_demos = dict(np.load(forward_path))
			backward_path = f'/iris/u/architsh/code/misc_data/franka_demos/cube_pickup/backward/demos.npz'
			backward_demos = dict(np.load(backward_path))
		elif 'clothhook' in name:
			episodic_path_length = 200
			forward_path = f'/iris/u/architsh/code/misc_data/franka_demos/cloth_hook_v2/forward/demos.npz'
			forward_demos = dict(np.load(forward_path))
			backward_path = f'/iris/u/architsh/code/misc_data/franka_demos/cloth_hook_v2/backward/demos.npz'
			backward_demos = dict(np.load(backward_path))
		elif 'clothflatten' in name:
			episodic_path_length = 500
			forward_path = f'/iris/u/architsh/code/misc_data/franka_demos/cloth_flatten/forward/demos.npz'
			forward_demos = dict(np.load(forward_path))
			backward_path = f'/iris/u/architsh/code/misc_data/franka_demos/cloth_flatten/backward/demos.npz'
			backward_demos = dict(np.load(backward_path))
		elif 'bowlcover' in name:
			episodic_path_length = 300
			forward_path = f'/iris/u/architsh/code/misc_data/franka_demos/bowl_cover/forward/demos.npz'
			forward_demos = dict(np.load(forward_path))
			backward_path = f'/iris/u/architsh/code/misc_data/franka_demos/bowl_cover/backward/demos.npz'
			backward_demos = dict(np.load(backward_path))
		elif 'peginsertion' in name:
			episodic_path_length = 200
			DoF = 4
			forward_path = f'/iris/u/architsh/code/misc_data/franka_demos/peg_insertion_v2/forward/demos.npz'
			forward_demos = dict(np.load(forward_path))
			backward_path = f'/iris/u/architsh/code/misc_data/franka_demos/peg_insertion_v2/backward/demos.npz'
			backward_demos = dict(np.load(backward_path))

		state_type = 'ee' # can be passed as a parameter if needed
		path_length = None
		goal_state = None
		if 'goal' in name:
			# env name of the form: franka_goal_left, franka_goal_right
			goal_state = 'left_open' if 'left' in name else 'right_closed'
		if 'episodic' in name:
			# env name of the form: franka_episodic
			path_length = episodic_path_length

		nuc_ip = '172.16.0.10'
		env_kwargs = {
			'ip_address' : nuc_ip,
			'max_path_length': path_length,
			'goal_state': goal_state,
			'DoF': DoF,
		}
		franka_env = RobotEnv(**env_kwargs)
		train_env = FrankaDMEnvFromGymWrapper(franka_env, state=state_type, height=height, width=width)
		train_env = FrankaObsActionDTypeWrapper(train_env, np.float32, {'images': np.uint8,
																		'state': np.float32})
		train_env = ActionRepeatWrapper(train_env, action_repeat)
		train_env = ExtendedTimeStepWrapper(train_env)

		# set the eval env to a fixed length so episodes end
		env_kwargs['max_path_length'] = episodic_path_length
		env_kwargs['pause_after_reset'] = False if 'goal' in name else True
		franka_env_eval = RobotEnv(**env_kwargs)
		eval_env = FrankaDMEnvFromGymWrapper(franka_env_eval, state=state_type, height=height, width=width)
		eval_env = FrankaObsActionDTypeWrapper(eval_env, np.float32, {'images': np.uint8,
																	  'state': np.float32})
		eval_env = ActionRepeatWrapper(eval_env, action_repeat)
		eval_env = ExtendedTimeStepWrapper(eval_env)

		return train_env, eval_env, None, None, forward_demos, backward_demos
	else:
		env_loader = earl_benchmark.EARLEnvs(
			name,
			reward_type=reward_type,
			reset_train_env_at_goal=False,
			# train_horizon=int(25000), # TEMPORARILY OVERRIDING train parameters
			wide_init_distr=True,
		)
		train_env, eval_env = env_loader.get_envs()

		reset_states = env_loader.get_initial_states()
		goal_states = env_loader.get_goal_states()

		if env_loader.has_demos():
			forward_demos, backward_demos = env_loader.get_demonstrations()
		else:
			forward_demos, backward_demos = None, None

		vision_goal_states = None
		if name == 'sawyer_peg':
			forward_demos = dict(np.load(hydra.utils.get_original_cwd() + '/vision_demos/sawyer_peg/forward.npz'))
			goal_states = vision_goal_states = None 
			backward_demos = dict(np.load(hydra.utils.get_original_cwd() + '/vision_demos/sawyer_peg/backward.npz'))
			
		elif name == 'sawyer_door':
			forward_demos = dict(np.load('/iris/u/ahmedah/test_arl/ARLBaselines/fwd_door_demos/total_imgs.npz'))
			goal_states = vision_goal_states = None
			backward_demos = None

		elif name == 'tabletop_manipulation':
			forward_demos = np.load(hydra.utils.get_original_cwd() + '/vision_demos/tabletop/forward.pkl', allow_pickle=True)
			backward_demos = np.load(hydra.utils.get_original_cwd() + '/vision_demos/tabletop/backward.pkl', allow_pickle=True)

		# add wrappers
		train_env = DMEnvFromGymWrapper(train_env, height, width)
		train_env = ObsActionDTypeWrapper(train_env, np.float32, np.float32)
		train_env = ActionRepeatWrapper(train_env, action_repeat)
		train_env = ActionScaleWrapper(train_env, minimum=-1.0, maximum=+1.0)
		train_env = GoalVisionWrapper(train_env, num_frames=num_frames, goal_states=goal_states, height=height, width=width, vision_goal_states=vision_goal_states)
		train_env = ExtendedTimeStepWrapper(train_env)

		eval_env = DMEnvFromGymWrapper(eval_env, height, width)
		eval_env = ObsActionDTypeWrapper(eval_env, np.float32, np.float32)
		eval_env = ActionRepeatWrapper(eval_env, action_repeat)
		eval_env = ActionScaleWrapper(eval_env, minimum=-1.0, maximum=+1.0)
		eval_env = GoalVisionWrapper(eval_env, num_frames=num_frames, goal_states=goal_states, height=height, width=width, vision_goal_states=vision_goal_states)
		vision_goal_states = eval_env.get_goal_images()
		if name == 'tabletop_manipulation':
			goal_states = [goal_states, vision_goal_states]
		eval_env = ExtendedTimeStepWrapper(eval_env)

		return train_env, eval_env, reset_states, goal_states, forward_demos, backward_demos
