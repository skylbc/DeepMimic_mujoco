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



