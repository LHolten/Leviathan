from rlbot.agents.base_agent import BaseAgent
import os
import sys
path = os.path.dirname(os.path.abspath(__file__))


class TorchLearner(BaseAgent):
    def __init__(self, name, team, index):
        sys.path.append(path)
        from output_formatter import OutputFormatter
        from input_formatter import InputFormatter
        from cool_atba import Atba
        import torch

        BaseAgent.__init__(self, name, team, index)
        self.atba = Atba()
        self.torch = torch
        self.output_formatter = OutputFormatter()
        self.input_formatter = InputFormatter(self.index, self.index, self.team)

    def get_output(self, game_tick_packet):
        spatial, car_stats = self.input_formatter.get_input(game_tick_packet)

        atba_action = self.atba.get_action(spatial, car_stats)

        in_the_air = game_tick_packet.game_cars[self.index].jumped
        player_input = self.output_formatter.get_output(atba_action, in_the_air)

        return player_input
