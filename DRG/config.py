CONFIG = {"actions_list": ["no_change", "speed_up", "slow_down"],
          "state_features": ["position", "velocity"],
          "reward": 0,
          "fixed_rewards_dict": {"reached_goal": 20,
                           "per_step_cost": -1,   
                           },
          "min_velocity": 0,
          "max_velocity": 4,
          "min_position": 0,
          "max_position": 20,
          "init_state_pos": 0,
          "goal_state_pos": 20,
          "init_state_vel": 2,
          "road_grade":[0,0,0,0,1,1,1,1,2,2,2,2,2,2,1,1,1,1,0,0,0], # len = max_pos - min_pos + 1
          }
