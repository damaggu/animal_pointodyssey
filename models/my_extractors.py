import torch
import torch.nn as nn
from torchvision import transforms

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict

from gymnasium.spaces.utils import flatten

from numpy.typing import NDArray
from typing import Any

import numpy as np
from collections import OrderedDict

class ResNetExtractor(nn.Module):
    def __init__(self, output_dim: int = 512):
        super(ResNetExtractor, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.out_layer = nn.Sequential(nn.Linear(512, output_dim), nn.LayerNorm(512), nn.Tanh())

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        input_batch = self.preprocess(img)

        x = self.resnet.conv1(input_batch)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return self.out_layer(x)
        # return x

class ResidualMLP(nn.Module):
    def __init__(self, hidden_dim: int, n_blocks: int):
        super(ResidualMLP, self).__init__()
        self.blocks = nn.ParameterList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                      nn.ReLU(),
                                                      nn.Linear(hidden_dim, hidden_dim),
                                                      nn.ReLU()) for _ in range(n_blocks)])
        self.layer_norm = nn.ParameterList([nn.LayerNorm(hidden_dim) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, layer_norm in zip(self.blocks, self.layer_norm):
            x = x + block(x)
            x = layer_norm(x)
        return x



class MyCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space,
                 image_key: str = 'walker/egocentric_camera',
                 freeze_resnet: bool = True,
                 video_enc_dim: int = 512,
                 proprio_encoder_dim: int = 512,
                 n_blocks: int = 4,
                 out_dim: int = 1024):
        # super(MyCombinedExtractor, self).__init__(observation_space, features_dim=798)
        # super(MyCombinedExtractor, self).__init__(observation_space, features_dim=670)
        super(MyCombinedExtractor, self).__init__(observation_space, features_dim=out_dim)

        self.video_encoder = ResNetExtractor(video_enc_dim)
        if freeze_resnet:
            for param in self.video_encoder.resnet.parameters():
                param.requires_grad = False

        self.proprio_encoder = nn.Sequential(
            nn.Linear(158, proprio_encoder_dim),
            nn.LayerNorm(proprio_encoder_dim),
            nn.ReLU(),
        )

        hidden_dim = video_enc_dim + proprio_encoder_dim
        self.combined_encoder = ResidualMLP(hidden_dim=hidden_dim, n_blocks=n_blocks)

        self.image_key = image_key
        self.observation_space = observation_space

    def forward(self, observations: TensorDict) -> torch.Tensor:
        # Extract image features
        image = observations[self.image_key]
        image_features = self.video_encoder(image)

        # Extract other features
        other_features_flat = self._flatten_other_features(observations)
        other_features = self.proprio_encoder(other_features_flat)

        # Concatenate them
        # combined = torch.cat([image_features, other_features, other_features_flat], dim=1)
        # combined = torch.cat([image_features, other_features_flat], dim=1)
        combined = torch.cat([image_features, other_features], dim=1)
        return combined

    def _flatten_other_features(self, observations: dict[str, Any]): # -> dict[str, Any] | NDArray[Any]:
        if self.observation_space.is_np_flattenable:
            return torch.cat(
                [torch.flatten(s, 1)
                 for key, s in observations.items()
                 if key != self.image_key],
            dim=1)
        # return OrderedDict((key, flatten(s, observations[key]))
        #                    for key, s in self.observation_space.spaces.items()
        #                    if key != self.image_key)