import torch
import math


class Atba:
    def get_action(self, spatial, car_stats):
        action = torch.zeros(9)

        left_vector = spatial[:, 10]
        forward_vector = spatial[:, 9]
        up_vector = spatial[:, 11]
        car_location = spatial[:, 0]
        ball_location = spatial[:, 2]
        goal_location = torch.tensor([0, 5.12, 0.3], dtype=torch.float)
        own_goal_location = torch.tensor([0, -5.12, 0.3], dtype=torch.float)

        relative_ball = ball_location - car_location
        ball_distance = relative_ball.norm()
        relative_ball /= ball_distance
        relative_goal = goal_location - car_location
        relative_goal /= relative_goal.norm()
        relative_own_goal = own_goal_location - car_location
        relative_own_goal /= relative_own_goal.norm()

        # offence/ defence switching

        offence = (1 + ball_location[1] / 5.12) / 2
        defence = 1 - offence

        ball_direction = (1 + offence * relative_ball @ relative_goal - defence * relative_ball @ relative_own_goal)/2
        ball_direction = pow(ball_direction, 1.4)
        not_ball_direction = 1 - ball_direction

        car_offence = (1 + car_location[1] / 5.12) / 2
        car_defence = 1 - car_offence

        # controls

        left_ball = relative_ball @ left_vector
        left_opp_goal = relative_goal @ left_vector
        left_own_goal = relative_own_goal @ left_vector
        left_goal = car_defence * left_own_goal - car_offence * left_opp_goal
        steer = ball_direction * left_ball + not_ball_direction * left_goal
        roll = ball_direction * left_goal + not_ball_direction * -left_vector[2]

        forward_ball = relative_ball @ forward_vector
        forward_opp_goal = relative_goal @ forward_vector
        forward_own_goal = relative_own_goal @ forward_vector
        forward_goal = car_defence * forward_own_goal - car_offence * forward_opp_goal
        pitch = ball_direction * forward_goal + not_ball_direction * forward_vector[2]

        up_ball = relative_ball @ up_vector
        jump = ball_direction * up_ball + not_ball_direction * -1
        throttle = ball_direction * math.copysign(pow(1 - abs(up_ball), 6), forward_ball) + not_ball_direction * 1

        action[0] = throttle
        action[1] = math.copysign(pow(abs(pitch), 2), pitch)
        action[2] = 1 if throttle > 0.75 else -1
        action[3] = 1 if abs(steer) > 0.65 else -1
        action[4] = jump if ball_distance < 0.4 else -1
        action[5] = 1 if ball_distance < 0.3 else -1
        action[6] = math.copysign(pow(abs(roll), 2), -roll)
        action[7] = math.copysign(pow(abs(steer), 0.1), steer)
        action[8] = math.copysign(pow(abs(steer), 0.1), steer)

        return action
