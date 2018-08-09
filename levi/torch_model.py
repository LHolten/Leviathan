import torch
import torch.nn as nn


class SpatialInput(nn.Module):
    def __init__(self, size):
        nn.Module.__init__(self)

        self.location = nn.Linear(3, size, bias=True)
        self.velocity = nn.Linear(3, size, bias=True)
        self.angular_velocity = nn.Linear(3, size, bias=True)
        self.normal = nn.Linear(6, size, bias=False)

    def forward(self, spatial):
        processed_location = self.location(spatial[:, 0:3])
        processed_velocity = self.velocity(spatial[:, 3:6])
        processed_angular_velocity = self.angular_velocity(spatial[:, 6:9])
        processed_normal = self.normal(spatial[:, 9:15])

        return processed_location * processed_velocity * processed_angular_velocity * processed_normal


class ActorModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.input_x = SpatialInput(10)
        self.input_y = SpatialInput(10)
        self.input_z = SpatialInput(10)

        self.linear = nn.Linear(30, 9)

    def forward(self, spatial, car_stats):
        processed_x = self.input_x(spatial[:, 0])
        processed_y = self.input_y(spatial[:, 1])
        processed_z = self.input_z(spatial[:, 2])

        processed_spatial = torch.cat([processed_x, processed_y, processed_z], dim=1)

        return self.linear(processed_spatial)


class SymmetricModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.actor = ActorModel()
        self.tanh = nn.Tanh()

    def forward(self, spatial, car_stats):
        spatial_inv = torch.tensor(spatial)
        spatial_inv[:, 0] *= -1  # invert x coordinates
        spatial_inv[:, :, 10] *= -1  # invert own car left normal
        spatial_inv[:, :, 13] *= -1  # invert opp car left normal
        spatial_inv[:, :, 6:9] *= -1  # invert angular velocity

        output = self.actor(spatial, car_stats)
        output_inv = self.actor(spatial_inv, car_stats)

        output[:, 0:6] += output_inv[:, 0:6]
        output[:, 6:9] += -1 * output_inv[:, 6:9]

        output = self.tanh(output)

        return output

    def forward_single(self, spatial, car_stats):
        return torch.squeeze(self.forward(torch.unsqueeze(spatial, 0), torch.unsqueeze(car_stats, 0)))


class SingleAction:
    def __init__(self, model=SymmetricModel()):
        self.model = model

    def get_action(self, spatial, car_stats):
        return torch.squeeze(self.model.forward(torch.unsqueeze(spatial, 0), torch.unsqueeze(car_stats, 0)))
