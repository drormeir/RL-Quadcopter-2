import math
import numpy as np
from physics_sim import PhysicsSim
from physics_sim_utils import *

class DroneState():
    def __init__(self,
                 pos=[0,0,0],\
                 angle=[0.,0.,0.],\
                 velocities=[0.,0.,0.],\
                 angle_velocities=[0.,0.,0.],\
                 time=0,\
                 sim=None):
        if sim is not None:
            self.set_from_sim(sim)
        else:
            self.pos       = np.copy(pos)
            self.angle     = np.copy(angle)
            self.vel       = np.copy(velocities)
            self.angle_vel = np.copy(angle_velocities)
            self.time      = time
            self.__post_set_values()
        
    def set_from_sim(self, sim):
        self.pos       = np.copy(sim.pose[:3])
        self.angle     = np.copy(sim.pose[3:])
        self.vel       = np.copy(sim.v)
        self.angle_vel = np.copy(sim.angular_v)
        self.time      = sim.time
        self.__post_set_values()

    def __post_set_values(self):
        self.angle_cos  = np.cos(self.angle)
        self.angle_sin  = np.sin(self.angle)

    def is_valid(self):
        return np.isfinite(self.pos).all() and np.isfinite(self.angle).all() and np.isfinite(self.vel).all() and np.isfinite(self.angle_vel).all() and np.isfinite(self.angle_cos).all() and np.isfinite(self.angle_sin).all() and np.isfinite(self.time)
        
    def create_sim(self):
        return PhysicsSim(init_pose            = np.hstack((self.pos,self.angle)),\
                          init_velocities       = np.copy(self.vel),\
                          init_angle_velocities = np.copy(self.angle_vel),\
                          runtime               = self.time) 
        
    def __str__(self):
        return "XYZ: (" + str(self.pos[0]) + "," + str(self.pos[1]) + "," + str(self.pos[2]) + ") phi: " + str(self.angle[0]) + " theta: " + str(self.angle[1]) + " psi: " + str(self.angle[2]) + " Vxyz: (" + str(self.vel[0]) + "," + str(self.vel[1]) + "," + str(self.vel[2]) + ") Vphi: " + str(self.angle_vel[0]) + " Vtheta: " + str(self.angle_vel[1]) + " Vpsi: " + str(self.angle_vel[2]) + " time: " + str(self.time)

    def tolist(self):
        return list(self.pos) + list(self.angle) + list(self.vel) + list(self.angle_vel)
    
    def horizontal_direction_to(self, other):
        return other.pos[0] - self.pos[0], other.pos[1] - self.pos[1]

    def vertical_direction_to(self, other):
        return other.pos[2] - self.pos[2]

    def direction_3d_to(self, other):
        return other.pos[0] - self.pos[0], other.pos[1] - self.pos[1], other.pos[2] - self.pos[2]
    
    def horizontal_distance_to(self, other):
        dx,dy = self.horizontal_direction_to(other)
        return math.sqrt(dx*dx+dy*dy)
    
    def vertical_distance_to(self, other):
        return abs(self.pos[2] - other.pos[2])

    def distance_up_down_to(self, other):
        d = other.pos[2] - self.pos[2]
        return [max(d,0.), min(-d,0.)]

    def distance_to(self, other):
        return math.sqrt(sum((self.pos - other.pos)**2))
    
    def total_velocity(self):
        return math.sqrt(sum(self.vel **2))
    
    def total_angular_velocity(self):
        return math.sqrt(sum(self.angle_vel **2))
    
    def total_angular_velocity_hertz(self):
        return self.total_angular_velocity()/(2.*math.pi)
    
    def angle_vel_hertz(self):
        return self.angle_vel / (2.*math.pi)
    
    def horizontal_velocity(self):
        return math.sqrt(sum(self.vel[:2] **2))
        
    def rotate_by_phi(self, x_val, y_val):
        return rotate_by_unit_vector(x_val, y_val, self.angle_cos[0], self.angle_sin[0])
    
    def rotated_direction_3d_to(self, target):
        dir_x, dir_y, dir_z = self.direction_3d_to(target)
        proj_dir_on_phi, proj_dir_on_side = self.rotate_by_phi(dir_x, dir_y)
        return [proj_dir_on_phi, proj_dir_on_side, dir_z]
    
    def rotated_velocities(self):
        proj_dvel_on_phi, proj_dvel_on_side = self.rotate_by_phi(self.vel[0], self.vel[1])
        return [proj_dvel_on_phi, proj_dvel_on_side, self.vel[2]]
    
    def abs_velocities_3d_to(self, target, dt):
        dist = self.horizontal_distance_to(target)
        dir_x, dir_y, dir_z = self.direction_3d_to(target)
        vel_h = self.horizontal_velocity()
        if dist < 1e-6 or vel_h*dt >= dist:
            # touching the target or fly-by too fast
            proj_dvel_on_distance = 0.
            proj_dvel_on_side     = vel_h
        else:
            # usual case: normalizing direction
            dir_x /= dist
            dir_y /= dist
            # velocity on direction to target can be negative which means that target distance become bigger with time
            proj_dvel_on_distance, proj_dvel_on_side = rotate_by_unit_vector(self.vel[0], self.vel[1], dir_x, dir_y)
            # sideways velocity is symmetrical
            proj_dvel_on_side = abs(proj_dvel_on_side)
        vz = self.vel[2]
        if vz * dir_z < 0:
            vz = -abs(vz) # drone flies in the wrong vertical direction
        else:
            vz = abs(vz) # drone approaches the target with respect to z-axis
        return [proj_dvel_on_distance, proj_dvel_on_side, vz]
        
    # returns the cosine of angle between velocity vector and the direction to target
    def calc_alignment_of_velocity_to_target(self, target):
        dist = self.distance_to(target)
        if dist < 0.01:
            return 1.
        total_vel = self.total_velocity()
        if total_vel < 0.01:
            return 0.
        dir_x, dir_y, dir_z = self.direction_3d_to(target)
        return (dir_x * self.vel[0] + dir_y * self.vel[1] + dir_z * self.vel[2])/(dist*total_vel)
        

class TaskSeekAndHover():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_state, target_state, action_repeat=1):
        """Initialize a Task object.
        Params
        ======
            init_state: initial state of the quadcopter as DroneState type
            target_state: target/goal state for the agent
            tolerance_state: the maximum distance from target that is considered OK 
        """
        # Simulation
        assert np.isfinite(init_state.time) and init_state.time > 0
        self.sim = init_state.create_sim()
        # Goal
        self.init_state         = init_state
        self.target_state       = target_state
        self.original_bounds    = np.hstack((self.sim.lower_bounds.reshape(3,1),self.sim.upper_bounds.reshape(3,1)))
        self.characteristic_vel = max(self.sim.upper_bounds - self.sim.lower_bounds) / self.sim.runtime
        self.action_low         = 1
        self.action_high        = 900
        self.action_size        = 4
        self.action_repeat      = action_repeat
        self.state_size         = len(self.__get_sim_state().values_4_actor) * self.action_repeat
        print("Initializing mission:")
        print("Starting position:",str(init_state))
        print("")
        print("Target position:",str(target_state))
        print("")
        print("State size:",self.state_size)
        
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        rotor_speeds    = np.clip(rotor_speeds, self.action_low, self.action_high).tolist()
        self.curr_score = 0.0
        values_4_actor  = []
        for _ in range(self.action_repeat):
            next_state = self.__next_sim_state(rotor_speeds)
            if next_state is not None:
                self.curr_state  = next_state
            self.curr_score += self.curr_state.value
            values_4_actor.append(self.curr_state.values_4_actor)
        values_4_actor = np.concatenate(values_4_actor)
        self.curr_score /= self.action_repeat
        return values_4_actor, self.curr_score, self.sim.done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.curr_state = self.__get_sim_state()
        self.curr_score = self.curr_state.value
        return np.concatenate([self.curr_state.values_4_actor] * self.action_repeat)

    
    def __next_sim_state(self, rotor_speeds):
        if self.sim.done:
            return None
        prev_speed = self.__get_sim_state().total_velocity()
        self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
        if not is_sim_finite(self.sim):
            # overcome bugs in the simulator
            print("Warning: Infinite numbers in simulator!")
            self.sim.done = True
            return None
        if not self.sim.done:
            if self.sim.runtime - self.sim.time < 0.5*self.sim.dt:
                # numerical fix for timeout
                self.sim.done = True
            # continue normal simulation
            return self.__get_sim_state()
        if not is_sim_crash(self.sim):
            # done but not crash --> timeout
            return self.__get_sim_state()
        if not is_sim_kissing_the_ground(self.sim, prev_speed):
            # hard crash
            return self.__get_sim_state()
        # soft landing, make landing a valid state
        save_time       = self.sim.time
        land_pose       = np.copy(self.sim.pose)
        # land_pose[0] x value unchanged
        # land_pose[1] y value unchanged
        land_pose[2]    = self.sim.lower_bounds[2] # z value --> ground level
        # land_pose[3] phi angle unchanged
        land_pose[4]    = 0 # theta value zero --> perfect horizontal
        land_pose[5]    = 0 # psi value zero -->  no tilt
        self.sim.reset() # now sim is undone
        self.sim.time            = save_time
        self.sim.pose            = land_pose
        self.sim.v               = np.array([0.0, 0.0, 0.0])
        self.sim.angular_v       = np.array([0.0, 0.0, 0.0])
        return self.__get_sim_state()
    
    def __get_sim_state(self):
        sim_state      = DroneState(sim=self.sim)
        self.__calc_state_value(sim_state)
        values_4_actor = self.__values_for_actor(sim_state)
        if values_4_actor is None:
            return None
        sim_state.values_4_actor = values_4_actor
        self.curr_state = sim_state
        return sim_state

    def __calc_state_value(self, state):
        state.value     = 0.0
        distance_score  = self.__calc_relative_location_between_target_and_bounds(state)
        # keep drone in horizontal pose:  1.0 is maximum value when theta == 0                    
        theta_score     = 0.5*(1.0 + state.angle_cos[1])
        # I do not care about the value of phi
        # TODO: psi is always zero so I do not count that angle either
        angle_vel       = state.total_angular_velocity_hertz()
        angle_vel_score = 1./(1.+angle_vel) # lower angle vel --> better score
        time_left       = self.__calc_time_2_collision(state)
        collision_score = 1. if time_left > self.sim.runtime - state.time else time_left/(1+time_left)
        vel_dir_cos     = state.calc_alignment_of_velocity_to_target(self.target_state)
        vel_dir_score   = 0.5*(1.0 + vel_dir_cos)
        #state.value     = distance_score * collision_score * (theta_score + angle_vel_score + vel_dir_score)/3.
        state.value     = distance_score * collision_score * theta_score * angle_vel_score * vel_dir_score
            
    # this monotonous continuous function returns "1.0" at the target and "0.0" on the edges of the simulator
    # this function does not have any local minima or local maxima besides the global "1.0" at the target
    def __calc_relative_location_between_target_and_bounds(self, state):
        state_pos       = state.pos
        if any(state_pos <= self.original_bounds[:,0]) or any(state_pos >= self.original_bounds[:,1]):
            # crash or touching borders --> even if target is on border the drone should not touch it
            return 0.
        target_pos      = self.target_state.pos
        # ray tracing from the target through current location till touching the border via strait line
        distance_score = 1.0
        for i_axis in range(3):
            dif_axis = state_pos[i_axis] - target_pos[i_axis]
            i_bound  = 0 if dif_axis < 0. else 1
            x_bound  = 1. - (dif_axis / (self.original_bounds[i_axis][i_bound] - target_pos[i_axis]))
            assert x_bound >= 0. and x_bound <= 1.
            if x_bound < distance_score:
                distance_score = x_bound
        return distance_score
    
    def __calc_time_2_collision(self,state):
        max_time          = self.sim.runtime
        bound_2_collision = self.__get_original_bounds(state.vel)
        dist_2_collision  = abs(bound_2_collision-state.pos)
        res               = math.inf
        for i_axis in range(3):
            v,d = abs(state.vel[i_axis]),dist_2_collision[i_axis]
            if v*max_time > d:
                res = min(res,d/v)
        return res
    
    # The phi value may act in heretic way when trying to hover at the target
    # Therefore, I ommit the phi value and I rotated all the values with respect to that angle to compensate for the missing value
    def __values_for_actor(self, state):
        directions = state.rotated_direction_3d_to(self.target_state)
        # not all the directions are alike
        # directions[0] represents the distance to the target in the direction of phi
        # If that value is negative it means that the drone is looking in the wrong direction, it has to turn around
        charactaristic_distance = [100. if directions[0] >=0 else 5., 10., 10.]
        directions = normalize_by_characteristics(directions,charactaristic_distance)
        velocities = state.rotated_velocities()
        velocities = normalize_by_characteristics(velocities,self.characteristic_vel)
        angles_vel = state.angle_vel_hertz()#normalize_by_characteristics(state.angle_vel,[1., 1., 1.])
        angles     = [state.angle_cos[1], state.angle_sin[1]]
        result = list(directions) + list(velocities) + list(angles) + list(angles_vel)
        return result
        
    def __get_original_bounds(self,predicate):
        return self.__get_bounds(self.original_bounds,predicate)
    def __get_ind_bounds(self,predicate):
        return np.asarray([0 if p <= 0. else 1 for p in predicate], dtype=np.int32)
    def __get_bounds(self,bounds,predicate):
        predicate = self.__get_ind_bounds(predicate)
        return np.asarray([bounds[0][predicate[0]], bounds[1][predicate[1]], bounds[2][predicate[2]]])
######################################################################################################
    
class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    