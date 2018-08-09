from rlbot.agents.base_agent import BaseAgent
import os
import sys
path = os.path.dirname(os.path.abspath(__file__))


class TorchLearner(BaseAgent):
    def __init__(self, name, team, index):
        sys.path.append(path)
        from output_formatter import OutputFormatter
        from input_formatter import InputFormatter
        import torch

        BaseAgent.__init__(self, name, team, index)
        self.actor_model = None
        self.single_model = None
        self.torch = torch
        self.output_formatter = OutputFormatter()
        self.input_formatter = InputFormatter(self.index, self.index, self.team)

    def initialize_agent(self):
        from torch_model import SingleAction, SymmetricModel
        self.actor_model = SymmetricModel()
        self.actor_model.load_state_dict(self.torch.load('levi/cool_atba.actor'))
        self.single_model = SingleAction(self.actor_model)

    def get_output(self, game_tick_packet):
        spatial, car_stats = self.input_formatter.get_input(game_tick_packet)

        with self.torch.no_grad():
            action = self.single_model.get_action(spatial, car_stats)

        in_the_air = game_tick_packet.game_cars[self.index].jumped
        player_input = self.output_formatter.get_output(action, in_the_air)

        return player_input
