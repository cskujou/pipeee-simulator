import torch

from pipeee.logger import get_logger
from pipeee.modeling_io import ModelInput, ModelInterim
from pipeee.worker.models.modeling_llama import LlamaForCausalLM

logger = get_logger(__name__)


class BaseModule:
    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class ModelRunner(BaseModule):
    def __init__(self, model: LlamaForCausalLM, forward_layers: tuple[int, int]):
        logger.info(f"Initializing ModelRunner with layers: {forward_layers}")
        self.model = model.model
        self.lm_head = model.lm_head
        self.norm = self.model.norm
        self.forward_layers = forward_layers
        logger.debug("ModelRunner initialized successfully")

    def create_mask(self, tree_mask, completed_length, dtype, device):
        if tree_mask is not None:
            num_uncomputed_tokens, num_computed_tokens = tree_mask.shape
            computed_length = completed_length + num_computed_tokens
            total_length = computed_length + num_uncomputed_tokens
            mask = torch.ones((1, 1, num_uncomputed_tokens, total_length), device=device, dtype=torch.bool)
            mask[:, 0, :, completed_length:computed_length] = tree_mask
            mask[:, 0, :, computed_length:] = torch.eye(num_uncomputed_tokens, device=device)
            min_dtype = torch.finfo(dtype).min
            mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        else:
            mask = torch.ones((1, 1, 1, completed_length + 1), device=device, dtype=dtype)
        return mask


    def preprocess(self, x: ModelInput):
        inputs_embeds: torch.Tensor = self.model.embed_tokens(x.input_ids)
        # cache_position = x.cache_position or torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        # position_ids = x.position_ids

        x.tree_mask = self.create_mask(
            tree_mask=x.tree_mask,
            completed_length=x.completed_length,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        position_embeddings = self.model.rotary_emb(inputs_embeds, x.position_ids)

        return ModelInterim(
            req_id=x.req_id,
            completed_length=x.completed_length,
            num_tokens=x.input_ids.shape[1],
            node_idx=x.node_idx,
            hidden_states=inputs_embeds,
            tree_mask=x.tree_mask,
            position_ids=x.position_ids,
            spec_cache_position=x.spec_cache_position,
            position_embeddings=position_embeddings,
            kv_cache=x.kv_cache,
            spec_cache=x.spec_cache,
        )

    def layer_forward(self, layer_idx: int, x: ModelInterim, mask) -> ModelInterim:
        layer = self.model.layers[layer_idx]
        x.hidden_states = layer(
            hidden_states=x.hidden_states,
            attention_mask=mask,
            position_ids=x.position_ids,
            kv_cache=x.kv_cache,
            spec_cache=x.spec_cache,
            spec_cache_position=x.spec_cache_position,
            position_embeddings=x.position_embeddings,
            completed_length=x.completed_length,
        )
        return x

    @torch.no_grad()
    def forward(self, x: ModelInput | ModelInterim) -> ModelInterim:
        logger.debug(f"ModelRunner forward starting with layers: {self.forward_layers}")
        if self.forward_layers[0] == 0 and isinstance(x, ModelInput):
            logger.debug("Preprocessing ModelInput")
            x = self.preprocess(x)

        for layer_idx in range(*self.forward_layers):
            logger.debug(f"Processing layer {layer_idx}")
            mask = x.tree_mask
            x = self.layer_forward(layer_idx, x, mask=mask)

        logger.debug("ModelRunner forward completed")
        return x

    @torch.no_grad()
    def get_logits(self, x: ModelInterim, logits_to_keep: int | slice) -> torch.Tensor:
        hidden_states = self.norm(x.hidden_states)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return logits
