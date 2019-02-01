# Convert to python
During Chinese New Year: 
* create character xml for Mujoco
* define rewards 
* connect the Mujoco env with the PPO learning method

# Structure
``` bash
├── args
│   ├── kin_char_args.txt
│   ├── run_humanoid3d_backflip_args.txt
│   ├── run_humanoid3d_cartwheel_args.txt
│   ├── run_humanoid3d_crawl_args.txt
│   ├── run_humanoid3d_dance_a_args.txt
│   ├── run_humanoid3d_dance_b_args.txt
│   ├── run_humanoid3d_getup_facedown_args.txt
│   ├── run_humanoid3d_getup_faceup_args.txt
│   ├── run_humanoid3d_jump_args.txt
│   ├── run_humanoid3d_kick_args.txt
│   ├── run_humanoid3d_punch_args.txt
│   ├── run_humanoid3d_roll_args.txt
│   ├── run_humanoid3d_run_args.txt
│   ├── run_humanoid3d_spin_args.txt
│   ├── run_humanoid3d_spinkick_args.txt
│   ├── run_humanoid3d_walk_args.txt
│   ├── train_humanoid3d_backflip_args.txt
│   ├── train_humanoid3d_cartwheel_args.txt
│   ├── train_humanoid3d_crawl_args.txt
│   ├── train_humanoid3d_dance_a_args.txt
│   ├── train_humanoid3d_dance_b_args.txt
│   ├── train_humanoid3d_getup_facedown_args.txt
│   ├── train_humanoid3d_getup_faceup_args.txt
│   ├── train_humanoid3d_jump_args.txt
│   ├── train_humanoid3d_kick_args.txt
│   ├── train_humanoid3d_punch_args.txt
│   ├── train_humanoid3d_roll_args.txt
│   ├── train_humanoid3d_run_args.txt
│   ├── train_humanoid3d_spin_args.txt
│   ├── train_humanoid3d_spinkick_args.txt
│   └── train_humanoid3d_walk_args.txt
├── data
│   ├── agents
│   ├── characters
│   ├── controllers
│   ├── motions
│   ├── policies
│   ├── shaders
│   ├── terrain
│   └── textures
├── DeepMimicCore
├── DeepMimic_Optimizer.py
├── DeepMimic.py
├── DeepMimic.pyproj
├── DeepMimic.sln
├── env
│   ├── action_space.py
│   ├── deepmimic_env.py
│   ├── env.py
├── learning
│   ├── agent_builder.py
│   ├── exp_params.py
│   ├── nets
│   ├── normalizer.py
│   ├── path.py
│   ├── pg_agent.py
│   ├── ppo_agent.py
│   ├── replay_buffer.py
│   ├── rl_agent.py
│   ├── rl_util.py
│   ├── rl_world.py
│   ├── solvers
│   ├── tf_agent.py
│   ├── tf_normalizer.py
│   └── tf_util.py
├── libraries
│   ├── bullet3
│   └── eigen
├── mpi_run.py
└── util
    ├── arg_parser.py
    ├── logger.py
    ├── math_util.py
    ├── mpi_util.py
    └── util.py

```

# Code

## Training-related

DeepMimic.py: 
``` python
def build_world(args, enable_draw, playback_speed=1):
    arg_parser = build_arg_parser(args)
    env = DeepMimicEnv(args, enable_draw)
    world = RLWorld(env, arg_parser)
    world.env.set_playback_speed(playback_speed)
    return world
```
``` python
def update_world(world, time_elapsed):
    num_substeps = world.env.get_num_update_substeps()
    timestep = time_elapsed / num_substeps
    num_substeps = 1 if (time_elapsed == 0) else num_substeps

    for i in range(num_substeps):
        world.update(timestep)

        valid_episode = world.env.check_valid_episode()
        if valid_episode:
            end_episode = world.env.is_episode_end()
            if (end_episode):
                world.end_episode()
                world.reset()
                break
        else:
            world.reset()
            break
    return
 ```

rl_world.py
``` python
# world.update()
    def update(self, timestep):
        self._update_agents(timestep)
        self._update_env(timestep)
        return
```

rl_agent.py
``` python
    def update(self, timestep):
        if self.need_new_action():
            self._update_new_action()

        if (self._mode == self.Mode.TRAIN and self.enable_training):
            self._update_counter += timestep

            while self._update_counter >= self.update_period:
                self._train()
                self._update_exp_params()
                self.world.env.set_sample_count(self._total_sample_count)
                self._update_counter -= self.update_period

        return
```
``` python
    def _train(self):
        samples = self.replay_buffer.total_count
        self._total_sample_count = int(MPIUtil.reduce_sum(samples))
        end_training = False
        
        if (self.replay_buffer_initialized):  
            if (self._valid_train_step()):
                prev_iter = self.iter
                iters = self._get_iters_per_update()
                avg_train_return = MPIUtil.reduce_avg(self.train_return)
            
                for i in range(iters):
                    curr_iter = self.iter
                    wall_time = time.time() - self.start_time
                    wall_time /= 60 * 60 # store time in hours

                    has_goal = self.has_goal()
                    s_mean = np.mean(self.s_norm.mean)
                    s_std = np.mean(self.s_norm.std)
                    g_mean = np.mean(self.g_norm.mean) if has_goal else 0
                    g_std = np.mean(self.g_norm.std) if has_goal else 0

                    self.logger.log_tabular("Iteration", self.iter)
                    self.logger.log_tabular("Wall_Time", wall_time)
                    self.logger.log_tabular("Samples", self._total_sample_count)
                    self.logger.log_tabular("Train_Return", avg_train_return)
                    self.logger.log_tabular("Test_Return", self.avg_test_return)
                    self.logger.log_tabular("State_Mean", s_mean)
                    self.logger.log_tabular("State_Std", s_std)
                    self.logger.log_tabular("Goal_Mean", g_mean)
                    self.logger.log_tabular("Goal_Std", g_std)
                    self._log_exp_params()

                    self._update_iter(self.iter + 1)
                    self._train_step()

                    Logger.print("Agent " + str(self.id))
                    self.logger.print_tabular()
                    Logger.print("") 

                    if (self._enable_output() and curr_iter % self.int_output_iters == 0):
                        self.logger.dump_tabular()

                if (prev_iter // self.int_output_iters != self.iter // self.int_output_iters):
                    end_training = self.enable_testing()

        else:

            Logger.print("Agent " + str(self.id))
            Logger.print("Samples: " + str(self._total_sample_count))
            Logger.print("") 

            if (self._total_sample_count >= self.init_samples):
                self.replay_buffer_initialized = True
                end_training = self.enable_testing()
        
        if self._need_normalizer_update:
            self._update_normalizers()
            self._need_normalizer_update = self.normalizer_samples > self._total_sample_count

        if end_training:
            self._init_mode_train_end()
 
        return
```


ppo_agent.py
``` python
    def _train_step(self):
        adv_eps = 1e-5

        start_idx = self.replay_buffer.buffer_tail
        end_idx = self.replay_buffer.buffer_head
        assert(start_idx == 0)
        assert(self.replay_buffer.get_current_size() <= self.replay_buffer.buffer_size) # must avoid overflow
        assert(start_idx < end_idx)

        idx = np.array(list(range(start_idx, end_idx)))        
        end_mask = self.replay_buffer.is_path_end(idx)
        end_mask = np.logical_not(end_mask) 
        
        vals = self._compute_batch_vals(start_idx, end_idx)
        new_vals = self._compute_batch_new_vals(start_idx, end_idx, vals)

        valid_idx = idx[end_mask]
        exp_idx = self.replay_buffer.get_idx_filtered(self.EXP_ACTION_FLAG).copy()
        num_valid_idx = valid_idx.shape[0]
        num_exp_idx = exp_idx.shape[0]
        exp_idx = np.column_stack([exp_idx, np.array(list(range(0, num_exp_idx)), dtype=np.int32)])
        
        local_sample_count = valid_idx.size
        global_sample_count = int(MPIUtil.reduce_sum(local_sample_count))
        mini_batches = int(np.ceil(global_sample_count / self.mini_batch_size))
        
        adv = new_vals[exp_idx[:,0]] - vals[exp_idx[:,0]]
        new_vals = np.clip(new_vals, self.val_min, self.val_max)

        adv_mean = np.mean(adv)
        adv_std = np.std(adv)
        adv = (adv - adv_mean) / (adv_std + adv_eps)
        adv = np.clip(adv, -self.norm_adv_clip, self.norm_adv_clip)

        critic_loss = 0
        actor_loss = 0
        actor_clip_frac = 0

        for e in range(self.epochs):
            np.random.shuffle(valid_idx)
            np.random.shuffle(exp_idx)

            for b in range(mini_batches):
                batch_idx_beg = b * self._local_mini_batch_size
                batch_idx_end = batch_idx_beg + self._local_mini_batch_size

                critic_batch = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
                actor_batch = critic_batch.copy()
                critic_batch = np.mod(critic_batch, num_valid_idx)
                actor_batch = np.mod(actor_batch, num_exp_idx)
                shuffle_actor = (actor_batch[-1] < actor_batch[0]) or (actor_batch[-1] == num_exp_idx - 1)

                critic_batch = valid_idx[critic_batch]
                actor_batch = exp_idx[actor_batch]
                critic_batch_vals = new_vals[critic_batch]
                actor_batch_adv = adv[actor_batch[:,1]]

                critic_s = self.replay_buffer.get('states', critic_batch)
                critic_g = self.replay_buffer.get('goals', critic_batch) if self.has_goal() else None
                curr_critic_loss = self._update_critic(critic_s, critic_g, critic_batch_vals)

                actor_s = self.replay_buffer.get("states", actor_batch[:,0])
                actor_g = self.replay_buffer.get("goals", actor_batch[:,0]) if self.has_goal() else None
                actor_a = self.replay_buffer.get("actions", actor_batch[:,0])
                actor_logp = self.replay_buffer.get("logps", actor_batch[:,0])
                curr_actor_loss, curr_actor_clip_frac = self._update_actor(actor_s, actor_g, actor_a, actor_logp, actor_batch_adv)
                
                critic_loss += curr_critic_loss
                actor_loss += np.abs(curr_actor_loss)
                actor_clip_frac += curr_actor_clip_frac

                if (shuffle_actor):
                    np.random.shuffle(exp_idx)

        total_batches = mini_batches * self.epochs
        critic_loss /= total_batches
        actor_loss /= total_batches
        actor_clip_frac /= total_batches

        critic_loss = MPIUtil.reduce_avg(critic_loss)
        actor_loss = MPIUtil.reduce_avg(actor_loss)
        actor_clip_frac = MPIUtil.reduce_avg(actor_clip_frac)

        critic_stepsize = self.critic_solver.get_stepsize()
        actor_stepsize = self.update_actor_stepsize(actor_clip_frac)

        self.logger.log_tabular('Critic_Loss', critic_loss)
        self.logger.log_tabular('Critic_Stepsize', critic_stepsize)
        self.logger.log_tabular('Actor_Loss', actor_loss) 
        self.logger.log_tabular('Actor_Stepsize', actor_stepsize)
        self.logger.log_tabular('Clip_Frac', actor_clip_frac)
        self.logger.log_tabular('Adv_Mean', adv_mean)
        self.logger.log_tabular('Adv_Std', adv_std)

        self.replay_buffer.clear()

        return
```

## Reward-related
``` bash
├── anim
│   ├── Character.cpp
│   ├── Character.h
│   ├── KinCharacter.cpp
│   ├── KinCharacter.h
│   ├── KinTree.cpp
│   ├── KinTree.h
│   ├── Motion.cpp
│   ├── Motion.h
│   ├── Shape.cpp
│   └── Shape.h
├── DeepMimicCore.cpp
├── DeepMimicCore.h
├── DeepMimicCore.i
├── DeepMimicCore.py
├── Main.cpp
├── Makefile
├── objs
│   ├── anim
│   ├── DeepMimicCore.o
│   ├── Main.o
│   ├── render
│   ├── scenes
│   ├── sim
│   └── util
├── render
│   ├── Camera.cpp
│   ├── Camera.h
│   ├── DrawCharacter.cpp
│   ├── DrawCharacter.h
│   ├── DrawGround.cpp
│   ├── DrawGround.h
│   ├── DrawKinTree.cpp
│   ├── DrawKinTree.h
│   ├── DrawMesh.cpp
│   ├── DrawMesh.h
│   ├── DrawObj.cpp
│   ├── DrawObj.h
│   ├── DrawPerturb.cpp
│   ├── DrawPerturb.h
│   ├── DrawSimCharacter.cpp
│   ├── DrawSimCharacter.h
│   ├── DrawUtil.cpp
│   ├── DrawUtil.h
│   ├── DrawWorld.cpp
│   ├── DrawWorld.h
│   ├── GraphUtil.cpp
│   ├── GraphUtil.h
│   ├── IBuffer.cpp
│   ├── IBuffer.h
│   ├── lodepng
│   ├── MatrixStack.cpp
│   ├── MatrixStack.h
│   ├── MeshUtil.cpp
│   ├── MeshUtil.h
│   ├── OBJLoader.h
│   ├── RenderState.h
│   ├── Shader.cpp
│   ├── Shader.h
│   ├── ShadowMap.cpp
│   ├── ShadowMap.h
│   ├── TextureDesc.cpp
│   ├── TextureDesc.h
│   ├── TextureUtil.cpp
│   ├── TextureUtil.h
│   ├── VertexBuffer.cpp
│   └── VertexBuffer.h
├── scenes
│   ├── DrawRLScene.cpp
│   ├── DrawRLScene.h
│   ├── DrawScene.cpp
│   ├── DrawScene.h
│   ├── DrawSceneImitate.cpp
│   ├── DrawSceneImitate.h
│   ├── DrawSceneKinChar.cpp
│   ├── DrawSceneKinChar.h
│   ├── DrawSceneSimChar.cpp
│   ├── DrawSceneSimChar.h
│   ├── RLScene.cpp
│   ├── RLScene.h
│   ├── RLSceneSimChar.cpp
│   ├── RLSceneSimChar.h
│   ├── SceneBuilder.cpp
│   ├── SceneBuilder.h
│   ├── Scene.cpp
│   ├── Scene.h
│   ├── SceneImitate.cpp
│   ├── SceneImitate.h
│   ├── SceneKinChar.cpp
│   ├── SceneKinChar.h
│   ├── SceneSimChar.cpp
│   └── SceneSimChar.h
├── sim
│   ├── ActionSpace.h
│   ├── AgentRegistry.cpp
│   ├── AgentRegistry.h
│   ├── CharController.cpp
│   ├── CharController.h
│   ├── ContactManager.cpp
│   ├── ContactManager.h
│   ├── Controller.cpp
│   ├── Controller.h
│   ├── CtController.cpp
│   ├── CtController.h
│   ├── CtCtrlUtil.cpp
│   ├── CtCtrlUtil.h
│   ├── CtPDController.cpp
│   ├── CtPDController.h
│   ├── CtrlBuilder.cpp
│   ├── CtrlBuilder.h
│   ├── CtVelController.cpp
│   ├── CtVelController.h
│   ├── DeepMimicCharController.cpp
│   ├── DeepMimicCharController.h
│   ├── ExpPDController.cpp
│   ├── ExpPDController.h
│   ├── GroundBuilder.cpp
│   ├── GroundBuilder.h
│   ├── Ground.cpp
│   ├── Ground.h
│   ├── GroundPlane.cpp
│   ├── GroundPlane.h
│   ├── ImpPDController.cpp
│   ├── ImpPDController.h
│   ├── MultiBody.cpp
│   ├── MultiBody.h
│   ├── ObjTracer.cpp
│   ├── ObjTracer.h
│   ├── PDController.cpp
│   ├── PDController.h
│   ├── Perturb.cpp
│   ├── Perturb.h
│   ├── PerturbManager.cpp
│   ├── PerturbManager.h
│   ├── RBDModel.cpp
│   ├── RBDModel.h
│   ├── RBDUtil.cpp
│   ├── RBDUtil.h
│   ├── SimBodyJoint.cpp
│   ├── SimBodyJoint.h
│   ├── SimBodyLink.cpp
│   ├── SimBodyLink.h
│   ├── SimBox.cpp
│   ├── SimBox.h
│   ├── SimCapsule.cpp
│   ├── SimCapsule.h
│   ├── SimCharacter.cpp
│   ├── SimCharacter.h
│   ├── SimCharBuilder.cpp
│   ├── SimCharBuilder.h
│   ├── SimCharGeneral.cpp
│   ├── SimCharGeneral.h
│   ├── SimCylinder.cpp
│   ├── SimCylinder.h
│   ├── SimJoint.cpp
│   ├── SimJoint.h
│   ├── SimObj.cpp
│   ├── SimObj.h
│   ├── SimPlane.cpp
│   ├── SimPlane.h
│   ├── SimRigidBody.cpp
│   ├── SimRigidBody.h
│   ├── SimSphere.cpp
│   ├── SimSphere.h
│   ├── SpAlg.cpp
│   ├── SpAlg.h
│   ├── World.cpp
│   └── World.h
└── util
    ├── Annealer.cpp
    ├── Annealer.h
    ├── ArgParser.cpp
    ├── ArgParser.h
    ├── BVHReader.cpp
    ├── BVHReader.h
    ├── CircularBuffer.h
    ├── FileUtil.cpp
    ├── FileUtil.h
    ├── IndexBuffer.h
    ├── IndexManager.cpp
    ├── IndexManager.h
    ├── json
    ├── JsonUtil.cpp
    ├── JsonUtil.h
    ├── MathUtil.cpp
    ├── MathUtil.h
    ├── Rand.cpp
    ├── Rand.h
    ├── Timer.cpp
    ├── Timer.h
    ├── Trajectory.cpp
    └── Trajectory.h

```


SceneImitate.cpp
``` C++
double cSceneImitate::CalcRewardImitate(const cSimCharacter& sim_char, const cKinCharacter& kin_char) const
{
	double pose_w = 0.5;
	double vel_w = 0.05;
	double end_eff_w = 0.15;
	double root_w = 0.2;
	double com_w = 0.1;

	double total_w = pose_w + vel_w + end_eff_w + root_w + com_w;
	pose_w /= total_w;
	vel_w /= total_w;
	end_eff_w /= total_w;
	root_w /= total_w;
	com_w /= total_w;

	const double pose_scale = 2;
	const double vel_scale = 0.1;
	const double end_eff_scale = 40;
	const double root_scale = 5;
	const double com_scale = 10;
	const double err_scale = 1;

	const auto& joint_mat = sim_char.GetJointMat();
	const auto& body_defs = sim_char.GetBodyDefs();
	double reward = 0;

	const Eigen::VectorXd& pose0 = sim_char.GetPose();
	const Eigen::VectorXd& vel0 = sim_char.GetVel();
	const Eigen::VectorXd& pose1 = kin_char.GetPose();
	const Eigen::VectorXd& vel1 = kin_char.GetVel();
	tMatrix origin_trans = sim_char.BuildOriginTrans();
	tMatrix kin_origin_trans = kin_char.BuildOriginTrans();

	tVector com0_world = sim_char.CalcCOM();
	tVector com_vel0_world = sim_char.CalcCOMVel();
	tVector com1_world;
	tVector com_vel1_world;
	cRBDUtil::CalcCoM(joint_mat, body_defs, pose1, vel1, com1_world, com_vel1_world);

	int root_id = sim_char.GetRootID();
	tVector root_pos0 = cKinTree::GetRootPos(joint_mat, pose0);
	tVector root_pos1 = cKinTree::GetRootPos(joint_mat, pose1);
	tQuaternion root_rot0 = cKinTree::GetRootRot(joint_mat, pose0);
	tQuaternion root_rot1 = cKinTree::GetRootRot(joint_mat, pose1);
	tVector root_vel0 = cKinTree::GetRootVel(joint_mat, vel0);
	tVector root_vel1 = cKinTree::GetRootVel(joint_mat, vel1);
	tVector root_ang_vel0 = cKinTree::GetRootAngVel(joint_mat, vel0);
	tVector root_ang_vel1 = cKinTree::GetRootAngVel(joint_mat, vel1);

	double pose_err = 0;
	double vel_err = 0;
	double end_eff_err = 0;
	double root_err = 0;
	double com_err = 0;
	double heading_err = 0;

	int num_end_effs = 0;
	int num_joints = sim_char.GetNumJoints();
	assert(num_joints == mJointWeights.size());

	double root_rot_w = mJointWeights[root_id];
	pose_err += root_rot_w * cKinTree::CalcRootRotErr(joint_mat, pose0, pose1);
	vel_err += root_rot_w * cKinTree::CalcRootAngVelErr(joint_mat, vel0, vel1);

	for (int j = root_id + 1; j < num_joints; ++j)
	{
		double w = mJointWeights[j];
		double curr_pose_err = cKinTree::CalcPoseErr(joint_mat, j, pose0, pose1);
		double curr_vel_err = cKinTree::CalcVelErr(joint_mat, j, vel0, vel1);
		pose_err += w * curr_pose_err;
		vel_err += w * curr_vel_err;

		bool is_end_eff = sim_char.IsEndEffector(j);
		if (is_end_eff)
		{
			tVector pos0 = sim_char.CalcJointPos(j);
			tVector pos1 = cKinTree::CalcJointWorldPos(joint_mat, pose1, j);
			double ground_h0 = mGround->SampleHeight(pos0);
			double ground_h1 = kin_char.GetOriginPos()[1];

			tVector pos_rel0 = pos0 - root_pos0;
			tVector pos_rel1 = pos1 - root_pos1;
			pos_rel0[1] = pos0[1] - ground_h0;
			pos_rel1[1] = pos1[1] - ground_h1;

			pos_rel0 = origin_trans * pos_rel0;
			pos_rel1 = kin_origin_trans * pos_rel1;

			double curr_end_err = (pos_rel1 - pos_rel0).squaredNorm();
			end_eff_err += curr_end_err;
			++num_end_effs;
		}
	}

	if (num_end_effs > 0)
	{
		end_eff_err /= num_end_effs;
	}

	double root_ground_h0 = mGround->SampleHeight(sim_char.GetRootPos());
	double root_ground_h1 = kin_char.GetOriginPos()[1];
	root_pos0[1] -= root_ground_h0;
	root_pos1[1] -= root_ground_h1;
	double root_pos_err = (root_pos0 - root_pos1).squaredNorm();
	
	double root_rot_err = cMathUtil::QuatDiffTheta(root_rot0, root_rot1);
	root_rot_err *= root_rot_err;

	double root_vel_err = (root_vel1 - root_vel0).squaredNorm();
	double root_ang_vel_err = (root_ang_vel1 - root_ang_vel0).squaredNorm();

	root_err = root_pos_err
			+ 0.1 * root_rot_err
			+ 0.01 * root_vel_err
			+ 0.001 * root_ang_vel_err;
	com_err = 0.1 * (com_vel1_world - com_vel0_world).squaredNorm();

	double pose_reward = exp(-err_scale * pose_scale * pose_err);
	double vel_reward = exp(-err_scale * vel_scale * vel_err);
	double end_eff_reward = exp(-err_scale * end_eff_scale * end_eff_err);
	double root_reward = exp(-err_scale * root_scale * root_err);
	double com_reward = exp(-err_scale * com_scale * com_err);

	reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward
		+ root_w * root_reward + com_w * com_reward;

	return reward;
}
```

