import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from nip import nip
from typing import Callable
from efficientnet_pytorch import EfficientNet


def _replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int = 16) -> nn.Module:
    """
    Replaces all BatchNorm layers with GroupNorm.
    Derived from:
    https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/nomad/nomad_vint.py#L135
    """
    _replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def _replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.
    Derived from:
    https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/nomad/nomad_vint.py#L151

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


class AbstractObservationEncoder(ABC, nn.Module):
    """Base class for creating observation encoders for ViNT.
    See example from the original ViNT:
    https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/vint.py#L40
    https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/vint.py#L99
    """

    def __init__(self, 
                encoding_size: int) -> None:
        super(AbstractObservationEncoder, self).__init__()
        self._encoding_size = encoding_size

    @abstractmethod
    def forward(self, context: torch.Tensor, observation: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def encoding_size(self) -> int:
        return self._encoding_size


class AbstractGoalEncoder(ABC, nn.Module):
    """Base class for creating goal encoders for ViNT.
    See example from the original ViNT:
    https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/vint.py#L42
    https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/vint.py#L82
    """

    def __init__(self, 
                encoding_size: int) -> None:
        super(AbstractGoalEncoder, self).__init__()
        self._encoding_size = encoding_size

    @abstractmethod
    def forward(self, context: torch.Tensor, observation: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    @property
    def encoding_size(self) -> int:
        return self._encoding_size


class AbstractTraversabilityEncoder(ABC, nn.Module):
    def __init__(self,
                 encoding_size: int) -> None:
        super(AbstractTraversabilityEncoder, self).__init__()
        self._encoding_size = encoding_size

    @abstractmethod
    def forward(self, context: torch.Tensor, observation: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    @property
    def encoding_size(self) -> int:
        return self._encoding_size
    
@nip
class EfficientNetObservationEncoder(AbstractObservationEncoder):
    """Observation Encoder for ViNT."""

    def __init__(self, 
                encoder_name: str = "efficientnet-b0",
                encoding_size: int = 512,
                context_size: int = 5,
                replace_batch_norm: bool = True) -> None:
        """Observation encoder based on EfficientNet model. Derived from the original ViNT/NoMaD implementation.
        Processes context elements and observation separately, and forms a sequence of (context_size + 1) embeddings. 

        Args:
            encoder_name (str, optional): EfficientNet version name. Defaults to "efficientnet-b0".
            encoding_size (int, optional): Size of the encoded vector. Defaults to 512.
            context_size (int, optional): Length of the context. Defaults to 5.
            replace_batch_norm (bool, optional): Replaces BatchNorm with GroupNorm (proven to work better). Defaults to True.
        """
        super(EfficientNetObservationEncoder, self).__init__(encoding_size)
        self._encoder = EfficientNet.from_name(encoder_name, in_channels=3)
        if replace_batch_norm:
            self._encoder = _replace_bn_with_gn(self._encoder)
        self._num_features = self._encoder._fc.in_features
        self._context_size = context_size
        if self._num_features != self._encoding_size:
            self._compress_enc = nn.Linear(self._num_features, self.encoding_size)
        else:
            self._compress_enc = nn.Identity()

    def forward(self, 
                context: torch.Tensor, 
                observation: torch.Tensor, 
                goal: torch.Tensor) -> torch.Tensor:
        BS, L, C, H, W = context.shape
        full_obs = torch.cat((context, observation.unsqueeze(1)), dim=1)
        full_obs = full_obs.reshape((BS * (L+1), C, H, W))

        obs_encoding = self._encoder.extract_features(full_obs)
        obs_encoding = self._encoder._avg_pooling(obs_encoding)

        if self._encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self._encoder._dropout(obs_encoding)

        obs_encoding = self._compress_enc(obs_encoding)
        obs_encoding = obs_encoding.reshape((BS, (L+1), -1))

        return obs_encoding


@nip
class EfficientNetGoalEncoder(AbstractGoalEncoder):
    """Goal Encoder for ViNT."""

    def __init__(self, 
                 encoder_name: str = "efficientnet-b0",
                 encoding_size: int = 512,
                 replace_batch_norm: bool = True) -> None:
        """Goal encoder based on EfficientNet model. Derived from the original ViNT/NoMaD implementation.
        Processes observation+goal as an "image" and forms single embedding. 

        Args:
            obs_encoder (str, optional): EfficientNet version name. Defaults to "efficientnet-b0".
            encoding_size (int, optional): Size of the encoded vector. Defaults to 512.
            context_size (int, optional): Length of the context. Defaults to 5.
            replace_batch_norm (bool, optional): Replaces BatchNorm with GroupNorm (proven to work better). Defaults to True.
        """
        super(EfficientNetGoalEncoder, self).__init__(encoding_size)
        self._encoder = EfficientNet.from_name(encoder_name, in_channels=6)
        if replace_batch_norm:
            self._encoder = _replace_bn_with_gn(self._encoder)
        self._num_features = self._encoder._fc.in_features
        if self._num_features != self._encoding_size:
            self._compress_enc = nn.Linear(self._num_features, self._encoding_size)
        else:
            self._compress_enc = nn.Identity()

    def forward(self, 
                context: torch.Tensor, 
                observation: torch.Tensor, 
                goal: torch.Tensor) -> torch.Tensor:
        obsgoal_img = torch.cat([observation, goal], dim=1)
        goal_encoding = self._encoder.extract_features(obsgoal_img)
        goal_encoding = self._encoder._avg_pooling(goal_encoding)
        if self._encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = self._encoder._dropout(goal_encoding)
        goal_encoding = self._compress_enc(goal_encoding)
        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)
        return goal_encoding
    
@nip
class EfficientNetTraversabilityEncoder(AbstractTraversabilityEncoder):
    """traversability Encoder for ViNT."""
    def __init__(self, 
                 encoder_name: str = "efficientnet-b0",
                 encoding_size: int = 512,
                 replace_batch_norm: bool = True) -> None:
        """traversability encoder based on EfficientNet model. Derived from the original ViNT/NoMaD implementation.
        Processes observation+traversability as an "image" and forms single embedding. 

        Args:
            obs_encoder (str, optional): EfficientNet version name. Defaults to "efficientnet-b0".
            encoding_size (int, optional): Size of the encoded vector. Defaults to 512.
            context_size (int, optional): Length of the context. Defaults to 5.
            replace_batch_norm (bool, optional): Replaces BatchNorm with GroupNorm (proven to work better). Defaults to True.
        """
        super(EfficientNetTraversabilityEncoder, self).__init__(encoding_size)
        self._encoder = EfficientNet.from_name(encoder_name, in_channels=4)
        if replace_batch_norm:
            self._encoder = _replace_bn_with_gn(self._encoder)

        self._num_features = self._encoder._fc.in_features
        if self._num_features!= self._encoding_size:
            self._compress_enc = nn.Linear(self._num_features, self._encoding_size)
        else:
            self._compress_enc = nn.Identity()
    
    def forward(self, 
                observation: torch.Tensor, 
                traversability: torch.Tensor) -> torch.Tensor:
        travobs_img = torch.cat([observation, traversability], dim=1)
        traversability_encoding = self._encoder.extract_features(travobs_img)
        traversability_encoding = self._encoder._avg_pooling(traversability_encoding)
        if self._encoder._global_params.include_top:
            traversability_encoding = traversability_encoding.flatten(start_dim=1)
            traversability_encoding = self._encoder._dropout(traversability_encoding)
        traversability_encoding = self._compress_enc(traversability_encoding)
        if len(traversability_encoding.shape) == 2:
            traversability_encoding = traversability_encoding.unsqueeze(1)
        return traversability_encoding
 
    
