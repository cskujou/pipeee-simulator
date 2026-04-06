import torch

from pipeee.logger import get_logger

logger = get_logger(__name__)


class SpecCacheLayer:
    def __init__(self, max_cache_len):
        logger.debug(f"Initializing SpecCacheLayer with max_cache_len: {max_cache_len}")
        self.max_cache_len = max_cache_len
        self.keys, self.values = None, None
        self.is_initialized = False

    def lazy_initialization(self, key_states: torch.Tensor):
        logger.debug("Lazy initialization of SpecCacheLayer")
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        B, H, _, D = self.keys.shape
        ul = key_states.shape[2]
        L = cache_position.shape[1]

        logger.debug(f"SpecCacheLayer.update: key_states shape={key_states.shape}, value_states shape={value_states.shape}, cache_position shape={cache_position.shape}")

        cache_position = cache_position[:, None, :, None].expand(B, H, L, D)

        self.keys.scatter_(dim=2, index=cache_position[:, :, -ul:], src=key_states)
        self.values.scatter_(dim=2, index=cache_position[:, :, -ul:], src=value_states)

        return self.keys.gather(dim=2, index=cache_position), self.values.gather(dim=2, index=cache_position)

    def get(
        self,
        cache_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logger.debug(f"SpecCacheLayer.get: cache_position={cache_position}")
        return self.keys[:, :, cache_position, :].unsqueeze(2), self.values[:, :, cache_position, :].unsqueeze(2)


class SpecCache:
    def __init__(
        self,
        max_cache_len,
    ):
        logger.info(f"Initializing SpecCache with max_cache_len: {max_cache_len}")
        self.layers = []
        self.max_cache_len = max_cache_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logger.debug(f"SpecCache.update: layer_idx={layer_idx}, key_states shape={key_states.shape}")
        while len(self.layers) <= layer_idx:
            logger.debug(f"Adding SpecCacheLayer for layer_idx: {len(self.layers)}")
            self.layers.append(SpecCacheLayer(self.max_cache_len))

        keys, values = self.layers[layer_idx].update(key_states, value_states, cache_position)

        return keys, values

    def get(
        self,
        cache_position: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        logger.debug(f"SpecCache.get: cache_position={cache_position}, num_layers={len(self.layers)}")
        return tuple(layer.get(cache_position) for layer in self.layers)
