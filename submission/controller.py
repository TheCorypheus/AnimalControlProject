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
        
        self.coms = np.empty((721, 2))
        
        turn_intensity = 2.5

        self.turn_left_steps = PreprogrammedSteps()
        self.turn_left_steps.swing_period["RF"] *= turn_intensity
        self.turn_left_steps.swing_period["RM"] *= turn_intensity
        self.turn_left_steps.swing_period["RH"] *= turn_intensity
        self.turn_left_steps.swing_period["LF"] /= turn_intensity
        self.turn_left_steps.swing_period["LM"] /= turn_intensity
        self.turn_left_steps.swing_period["LH"] /= turn_intensity
            
        self.turn_right_steps = PreprogrammedSteps()
        self.turn_right_steps.swing_period["LF"] *= turn_intensity
        self.turn_right_steps.swing_period["LM"] *= turn_intensity
        self.turn_right_steps.swing_period["LH"] *= turn_intensity
        self.turn_right_steps.swing_period["RF"] /= turn_intensity
        self.turn_right_steps.swing_period["RM"] /= turn_intensity
        self.turn_right_steps.swing_period["RH"] /= turn_intensity

    def get_actions(self, obs: Observation) -> Action:
        visual_information = self.process_visual_observation(obs, 0.06)
        if(visual_information[2] > 0.01):
            joint_angles, adhesion = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.turn_right_steps,
                action=np.array([1.0, 1.0]),
            )
        elif(visual_information[5] > 0.01):
            joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.turn_left_steps,
            action=np.array([1.0, 1.0]),
            )   
        elif(obs["odor_intensity"][0][0] > obs["odor_intensity"][0][1]):            
            joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.turn_left_steps,
            action=np.array([1.0, 1.0]),
            )
        else:    
            joint_angles, adhesion = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.turn_right_steps,
                action=np.array([1.0, 1.0]),
            )

        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }

    def done_level(self, obs: Observation):
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()

    def turn_left(turn_intensity: float, steps: PreprogrammedSteps) -> PreprogrammedSteps:
        steps.swing_period["RF"] *= turn_intensity
        steps.swing_period["RM"] *= turn_intensity
        steps.swing_period["RH"] *= turn_intensity
        return steps
    
    def turn_right(turn_intensity: float, steps: PreprogrammedSteps) -> PreprogrammedSteps:
        steps.swing_period["LF"] *= turn_intensity
        steps.swing_period["LM"] *= turn_intensity
        steps.swing_period["LH"] *= turn_intensity
        return steps
    
    #Function adapted from Neuromechfly tutorial
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