import math
import torch


def make_tensor(l):
    return torch.tensor(l)


class InputFormatter:
    def __init__(self, index, opp_index, team):
        self.index = index
        self.opp_index = opp_index
        self.team = team
        self.switch_tensor = torch.tensor([1, 0])

    def get_input(self, packet):
        own_car = packet.game_cars[self.index]
        opp_car = packet.game_cars[self.opp_index]
        game_ball = packet.game_ball

        own_car_location = own_car.physics.location
        opp_car_location = opp_car.physics.location
        game_ball_location = game_ball.physics.location

        own_car_velocity = own_car.physics.velocity
        opp_car_velocity = opp_car.physics.velocity
        game_ball_velocity = game_ball.physics.velocity

        own_car_angular = own_car.physics.angular_velocity
        opp_car_angular = opp_car.physics.angular_velocity
        game_ball_angular = game_ball.physics.angular_velocity

        own_theta = get_all_vectors(own_car)
        opp_theta = get_all_vectors(opp_car)

        spatial_x = make_tensor([own_car_location.x, opp_car_location.x, game_ball_location.x,
                                 own_car_velocity.x, opp_car_velocity.x, game_ball_velocity.x,
                                 own_car_angular.x, opp_car_angular.x, game_ball_angular.x])
        spatial_x = torch.cat([spatial_x, own_theta[0], opp_theta[0]])

        spatial_y = make_tensor([own_car_location.y, opp_car_location.y, game_ball_location.y,
                                 own_car_velocity.y, opp_car_velocity.y, game_ball_velocity.y,
                                 own_car_angular.y, opp_car_angular.y, game_ball_angular.y])
        spatial_y = torch.cat([spatial_y, own_theta[1], opp_theta[1]])

        spatial_z = make_tensor([own_car_location.z, opp_car_location.z, game_ball_location.z,
                                 own_car_velocity.z, opp_car_velocity.z, game_ball_velocity.z,
                                 own_car_angular.z, opp_car_angular.z, game_ball_angular.z])
        spatial_z = torch.cat([spatial_z, own_theta[2], opp_theta[2]])

        spatial = torch.stack([spatial_x, spatial_y, spatial_z])

        spatial[:, 0:6] /= 1000

        own_car_stats = make_tensor([own_car.boost / 100,
                                     1 if own_car.jumped else 0,
                                     1 if own_car.double_jumped else 0,
                                     1 if own_car.is_demolished else 0,
                                     1 if own_car.has_wheel_contact else 0])
        opp_car_stats = make_tensor([opp_car.boost / 100,
                                     1 if opp_car.jumped else 0,
                                     1 if opp_car.double_jumped else 0,
                                     1 if opp_car.is_demolished else 0,
                                     1 if opp_car.has_wheel_contact else 0])

        car_stats = torch.stack([own_car_stats, opp_car_stats])

        if self.team == 1:
            spatial[0:2] *= -1
            car_stats = torch.index_select(car_stats, 0, self.switch_tensor)

        return spatial, car_stats


def get_all_vectors(car):
    pitch = float(car.physics.rotation.pitch)
    yaw = float(car.physics.rotation.yaw)
    roll = float(car.physics.rotation.roll)

    c_r = math.cos(roll)
    s_r = math.sin(roll)
    c_p = math.cos(pitch)
    s_p = math.sin(pitch)
    c_y = math.cos(yaw)
    s_y = math.sin(yaw)

    theta = torch.zeros(3, 3)
    #   front direction
    theta[0, 0] = c_p * c_y
    theta[1, 0] = c_p * s_y
    theta[2, 0] = s_p

    #   left direction
    theta[0, 1] = c_y * s_p * s_r - c_r * s_y
    theta[1, 1] = s_y * s_p * s_r + c_r * c_y
    theta[2, 1] = -c_p * s_r

    #   up direction
    theta[0, 2] = -c_r * c_y * s_p - s_r * s_y
    theta[1, 2] = -c_r * s_y * s_p + s_r * c_y
    theta[2, 2] = c_p * c_r

    return theta
