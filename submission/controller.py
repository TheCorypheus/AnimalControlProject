import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from flygym.examples.locomotion import PreprogrammedSteps
np.set_printoptions(threshold=np.inf)
class Controller(BaseController):
    def __init__(
        self,
        timestep=1e-4,
        seed=0,
    ):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        
        self.heading_vec = np.array([1, 0])
        self.heading_change_per_step = 0.00044

        self.est_pos = np.array([0, 0])
        self.pos_change_per_step = 0.01

        self.coms = np.empty((721, 2))
        #controls how quickly the fly turns
        turn_intensity = 2.5

        #CPG used to turn left
        self.turn_left_steps = PreprogrammedSteps()
        self.turn_left_steps.swing_period["RF"] *= turn_intensity
        self.turn_left_steps.swing_period["RM"] *= turn_intensity
        self.turn_left_steps.swing_period["RH"] *= turn_intensity
        self.turn_left_steps.swing_period["LF"] /= turn_intensity
        self.turn_left_steps.swing_period["LM"] /= turn_intensity
        self.turn_left_steps.swing_period["LH"] /= turn_intensity

        #CPG used to turn right    
        self.turn_right_steps = PreprogrammedSteps()
        self.turn_right_steps.swing_period["LF"] *= turn_intensity
        self.turn_right_steps.swing_period["LM"] *= turn_intensity
        self.turn_right_steps.swing_period["LH"] *= turn_intensity
        self.turn_right_steps.swing_period["RF"] /= turn_intensity
        self.turn_right_steps.swing_period["RM"] /= turn_intensity
        self.turn_right_steps.swing_period["RH"] /= turn_intensity

        #CPG for stopping
        self.stop_steps = PreprogrammedSteps()
        self.stop_steps.swing_period["LF"] = [0, 0]
        self.stop_steps.swing_period["LM"] = [0, 0]
        self.stop_steps.swing_period["LH"] = [0, 0]
        self.stop_steps.swing_period["RF"] = [0, 0]
        self.stop_steps.swing_period["RM"] = [0, 0]
        self.stop_steps.swing_period["RH"] = [0, 0]

    def get_actions(self, obs: Observation) -> Action:
        #saves position and size of potential objects for both eyes
        visual_information = self.process_visual_observation(obs, 0.06)
        #origin reached for level 4
        if(obs["reached_odour"] and np.linalg.norm(self.est_pos) < 0.05):
            joint_angles, adhesion = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.stop_steps,
                action=np.array([1.0, 1.0]),
            )
            self.quit = True
        #odor reached and turning towards origin for level 4    
        elif(obs["reached_odour"] and self.signed_angle(self.heading_vec, -self.est_pos) > 0):
            joint_angles, adhesion = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.turn_right_steps,
                action=np.array([1.0, 1.0]),
            )
            self.heading_vec = self.rotate_vector(-self.heading_change_per_step, self.heading_vec)
            self.est_pos = self.est_pos + self.heading_vec * self.pos_change_per_step
            #print("right")
        elif(obs["reached_odour"] and self.signed_angle(self.heading_vec, -self.est_pos) <= 0):
            joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.turn_left_steps,
            action=np.array([1.0, 1.0]),
            )   
            self.heading_vec = self.rotate_vector(self.heading_change_per_step, self.heading_vec)
            self.est_pos = self.est_pos + self.heading_vec * self.pos_change_per_step
        #object detected to the left    
        elif(visual_information[2] > 0.01):
            joint_angles, adhesion = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.turn_right_steps,
                action=np.array([1.0, 1.0]),
            )
            self.heading_vec = self.rotate_vector(-self.heading_change_per_step, self.heading_vec)
            self.est_pos = self.est_pos + self.heading_vec * self.pos_change_per_step
        #object detected to the right    
        elif(visual_information[5] > 0.01):
            joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.turn_left_steps,
            action=np.array([1.0, 1.0]),
            )   
            self.heading_vec = self.rotate_vector(self.heading_change_per_step, self.heading_vec)
            self.est_pos = self.est_pos + self.heading_vec * self.pos_change_per_step
        #odor intensity igher to the left
        elif(obs["odor_intensity"][0][0] > obs["odor_intensity"][0][1]):            
            joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.turn_left_steps,
            action=np.array([1.0, 1.0]),
            )
            self.heading_vec = self.rotate_vector(self.heading_change_per_step, self.heading_vec)
            self.est_pos = self.est_pos + self.heading_vec * self.pos_change_per_step
        else:    
            joint_angles, adhesion = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.turn_right_steps,
                action=np.array([1.0, 1.0]),
            )
            self.heading_vec = self.rotate_vector(-self.heading_change_per_step, self.heading_vec)
            self.est_pos = self.est_pos + self.heading_vec * self.pos_change_per_step

        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }

    def done_level(self, obs: Observation):
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
    
    #Function adapted from Neuromechfly tutorial. Returns array with [x, y, size, x, y, size]
    #where [0:3] corresponds to the left eye and the other to the right eye
    #x and y correspond to the position of an object and size to its size 
    def process_visual_observation(self, raw_obs, obj_threshold=0.15):
        features = np.zeros((2, 3))
        for i, ommatidia_readings in enumerate(raw_obs["vision"]):
            is_obj = ommatidia_readings.max(axis=1) < obj_threshold
            is_obj_coords = self.coms[is_obj]
            if is_obj_coords.shape[0] > 0:
                features[i, :2] = is_obj_coords.mean(axis=0)
            features[i, 2] = is_obj_coords.shape[0]    
        features[:, 0] /= 450 # normalize y_center
        features[:, 1] /= 512  # normalize x_center
        features[:, 2] /= 721  # normalize area
        return features.ravel()
    
    #rotates a vector in 2d by a give angle
    def rotate_vector(self, angle, vec):
        rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return np.dot(rot_mat, vec)
    #returns the signed angle between two vectors from first to second vector
    def signed_angle(self, vec2, vec1):
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        angle =  np.arctan2(vec1[0] * vec2[1] - vec1[1] * vec2[0], vec1[0] * vec2[0] + vec1[1] * vec2[1])
        #print(angle)
        return angle