from dataclasses import dataclass
from typing import Optional

import torch
from transformers import Cache

from pipeee.sched.spec_cache import SpecCache


@dataclass
class TokenizedRequest:
    req_id: int
    input_ids: torch.LongTensor = None
    attention_mask: torch.Tensor = None
    
@dataclass
class PrefillOutput:
    input_ids: list[int]
    first_token_id: int
    past_key_values: Optional[Cache] = None
    
@dataclass
class DecodeInput:
    req_id: int
    input_ids: list[int]
    first_token_id: int
    past_key_values: Optional[Cache] = None
    
    @staticmethod
    def from_prefill_output(prefill_output: PrefillOutput, req_id: int) -> "DecodeInput":
        return DecodeInput(
            req_id=req_id,
            input_ids=prefill_output.input_ids,
            first_token_id=prefill_output.first_token_id,
            past_key_values=prefill_output.past_key_values,
        )

# @dataclass
# class FlattenedRequestTree:
#     req_id: int = None
#     input_ids: torch.LongTensor = None
#     spec_tokens: torch.LongTensor = None
#     spec_path_indices: list[int] = None
#     spec_attn_masks: torch.Tensor = None

@dataclass
class ModelInput:
    req_id: int
    completed_length: Optional[int] = 0
    node_idx: Optional[list[int]] = None
    input_ids: Optional[torch.LongTensor] = None
    tree_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.LongTensor] = None
    spec_cache_position: Optional[torch.LongTensor] = None
    kv_cache: Optional[Cache] = None
    spec_cache: Optional[SpecCache] = None
    
    def from_request(request) -> "ModelInput":
        return ModelInput(
            req_id=request.request_id,
            input_ids=request.current_token_ids,
            attention_mask=request.current_attention_mask,
            position_ids=request.current_position_ids,
            cache_position=request.current_cache_position,
            kv_cache=request.kv_cache,
        )
    
@dataclass
class ModelInterim:
    req_id: int
    completed_length: Optional[int] = 0
    num_tokens: int = None
    node_idx: Optional[list[int]] = None
    hidden_states: torch.FloatTensor = None
    tree_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.LongTensor] = None
    spec_cache_position: Optional[torch.LongTensor] = None
    position_embeddings: Optional[torch.FloatTensor] = None
    kv_cache: Optional[Cache] = None
    spec_cache: Optional[SpecCache] = None
    
@dataclass
class WorkerOutput:
    req_id: int
    last_worker: bool = False
    token_ids: list[int] = None
    node_idx: Optional[list[int]] = None
    confidences: list[float] = None
    is_eos: list[bool] = None
    exit_layer_idx: int = None
    
    
@dataclass
class SampledOutput:
    req_id: int
    token_ids: list[str]
    confidences: list[float] = None
    position_ids: list[int] = None
    node_idx: int = None
    is_early_exited: bool = None
    is_eos: list[bool] = None
    exit_layer_idx: int = None