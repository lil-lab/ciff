import os

import torch

from agents.agent_observed_state import AgentObservedState
from models.model.abstract_model import AbstractModel
from models.module.goal_position_module import GoalPositionModule
from models.module.symbolic_embddings import RadiusModule, AngleModule


class ModelGoalPosition(AbstractModel):
    def __init__(self, config, constants):
        AbstractModel.__init__(self, config, constants)
        self.radius_model = RadiusModule(15)
        self.angle_model = AngleModule(48)
        num_actions = config["num_actions"]
        self.goal_module = GoalPositionModule(
            radius_module=self.radius_model,
            angle_module=self.angle_model,
            num_actions=num_actions)

        if torch.cuda.is_available():
            self.goal_module.cuda()

    def get_probs_batch(self, agent_observed_state_list, mode=None):
        for aos in agent_observed_state_list:
            assert isinstance(aos, AgentObservedState)
        # print "batch size:", len(agent_observed_state_list)

        # sort list by instruction length
        agent_observed_state_list = sorted(
            agent_observed_state_list,
            key=lambda aos_: len(aos_.get_instruction()),
            reverse=True
        )

        agent_positions = [aos.get_position_orientation()
                           for aos in agent_observed_state_list]
        goal_positions = [aos.get_final_goal_position()
                          for aos in agent_observed_state_list]
        probs_batch = self.goal_module(agent_positions, goal_positions)
        return probs_batch

    def load_saved_model(self, load_dir):
        if torch.cuda.is_available():
            torch_load = torch.load
        else:
            torch_load = lambda f_: torch.load(f_, map_location=lambda s_, l_: s_)
        goal_module_path = os.path.join(load_dir, "goal_module_state.bin")
        self.goal_module.load_state_dict(torch_load(goal_module_path))

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save state file for goal nn
        goal_module_path = os.path.join(save_dir, "goal_module_state.bin")
        torch.save(self.goal_module.state_dict(), goal_module_path)

    def get_parameters(self):
        parameters = list(self.goal_module.parameters())
        return parameters

