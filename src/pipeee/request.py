from typing import Optional

import torch

from pipeee.config import RequestConfig
from pipeee.logger import get_logger
from pipeee.modeling_io import DecodeInput, ModelInput
from pipeee.request_trie import EarlyExitedTokenTrie

logger = get_logger(__name__)


class Request:
    def __init__(
        self,
        req_id: int,
        prompt_token_ids: list[int],
        config: RequestConfig,
        device: torch.device = None,
    ):
        logger.info(f"Initializing Request {req_id} with prompt length: {len(prompt_token_ids)}")
        self.req_id: int = req_id
        self.prompt_token_ids: list[int] = prompt_token_ids
        self.last_token: Optional[int] = None
        self.config = config
        self._output_token_ids: list[int] = []
        self.device = device if device is not None else torch.device("cpu")

        max_position: int = config.max_length
        max_position = (
            min(max_position, len(prompt_token_ids) + config.max_new_tokens)
            if config.max_new_tokens is not None
            else max_position
        )
        self.trie = EarlyExitedTokenTrie(config.max_nodes_count, config.spec_topk, max_position)
        self.finished = False
        logger.debug(f"Request {self.req_id} initialized with max_position: {max_position}")

    @staticmethod
    def from_decode_input(decode_input: DecodeInput, config: RequestConfig) -> "Request":
        request = Request(
            req_id=decode_input.req_id,
            prompt_token_ids=decode_input.input_ids,
            device=config.device,
            config=config.request_config,
        )
        request.update(decode_input.first_token_id, position=len(request.prompt_token_ids))
        return request

    def add_tokens(
        self,
        parent_node_id: int,
        token_ids: list[int],
        exit_layer_idx: int,
        confidences: list[float],
        is_eos: list[bool],
    ):
        logger.debug(f"Request {self.req_id} adding {len(token_ids)} token(s) to parent node {parent_node_id}")
        result = self.trie.add_tokens(
            parent_node_id=parent_node_id,
            token_ids=token_ids,
            exit_layer_idx=exit_layer_idx,
            confidences=confidences,
            is_eos=is_eos,
        )
        logger.debug(f"Add tokens result for req_id={self.req_id}: {result}")
        return result

    @property
    def root_id(self) -> Optional[int]:
        return self.trie.root.node_id if self.trie.root is not None else None

    def update(
        self, token_id: int, root_node_id: Optional[int] = None, position: Optional[int] = None
    ) -> Optional[int]:
        logger.debug(f"Request {self.req_id} updating with token_id: {token_id}")
        # Two case to update trie
        # 1. No root, token_id is generated during the prefill stage.
        # 2. With root, but need to confirm that the next token is sampled from the current root.
        assert self.root_id is None or root_node_id == self.root_id, (
            f"Root node id mismatch for req_id={self.req_id}: expected {self.root_id}, got {root_node_id}"
        )

        self._output_token_ids.append(token_id)
        logger.debug(f"Output tokens for req_id={self.req_id}: {len(self._output_token_ids)}")
        root_cache_id = self.trie.update(token_id, position)
        logger.debug(f"Root cache id for req_id={self.req_id}: {root_cache_id}")
        return root_cache_id

    def create_model_input(self) -> ModelInput:
        logger.debug(f"Creating model input for req_id={self.req_id}")
        node_idx, token_ids, parent_indices, cache_ids, position_ids = self.trie.flatten()
        completed_length = len(self.prompt_token_ids) + len(self._output_token_ids) - 1

        # only root is uncomputed, return a single token input
        if len(token_ids) == 1:
            return ModelInput(
                req_id=self.req_id,
                node_idx=node_idx,
                completed_length=completed_length,
                input_ids=torch.tensor(token_ids).to(self.device),
                position_ids=torch.tensor([completed_length]).to(self.device),
                spec_cache_position=torch.tensor(cache_ids, dtype=torch.long).to(self.device),
            )

        num_flattened_nodes = len(parent_indices)
        num_uncomputed_nodes = len(token_ids)
        num_computed_nodes = num_flattened_nodes - num_uncomputed_nodes

        computed_tree_mask = torch.eye(num_computed_nodes, dtype=torch.bool).to(self.device)
        for i in range(1, num_computed_nodes):
            parent_idx = parent_indices[i]
            computed_tree_mask[i] |= computed_tree_mask[parent_idx]

        uncomputed_parent_indices = torch.tensor(parent_indices[-num_uncomputed_nodes:]).to(self.device)
        tree_mask = computed_tree_mask[uncomputed_parent_indices]
        token_ids = torch.tensor(token_ids, dtype=torch.long).to(self.device)
        position_ids = torch.tensor(position_ids, dtype=torch.long).to(self.device)
        return ModelInput(
            req_id=self.req_id,
            node_idx=node_idx,
            completed_length=completed_length,
            input_ids=token_ids,
            # spec_ids=token_ids[1:],
            position_ids=position_ids,
            spec_cache_position=torch.tensor(cache_ids, dtype=torch.long).to(self.device),
            tree_mask=tree_mask,
        )
