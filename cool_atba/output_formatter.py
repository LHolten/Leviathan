from rlbot.utils.structures.bot_input_struct import PlayerInput
from random import random


class OutputFormatter:
    def __init__(self):
        self.player_input = PlayerInput()

    def get_output(self, action, in_the_air):
        self.player_input.throttle = action[0]
        self.player_input.pitch = action[1]
        self.player_input.boost = action[2] > semi_random(3)
        self.player_input.handbrake = action[3] > semi_random(3)

        action_1 = action[4] > semi_random(5)
        action_2 = action[5] > semi_random(5)

        jump_on_ground = not self.player_input.jump and not in_the_air and action_1
        flip_in_air = not self.player_input.jump and action_2
        jump_in_air = in_the_air and (flip_in_air or not action_2) and (action_1 or action_2)

        self.player_input.jump = jump_on_ground or jump_in_air

        self.player_input.roll = action[6]
        self.player_input.steer = action[7]
        self.player_input.yaw = action[8]

        return self.player_input


def semi_random(power):
    return pow(random() - random(), power)
