import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from flygym.examples.locomotion import PreprogrammedSteps

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

        turn_intensity = 1.5

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
        if(obs["odor_intensity"][0][0] > obs["odor_intensity"][0][1]):
            
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