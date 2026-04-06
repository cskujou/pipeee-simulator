import heapq
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pipeee.confidence_updater import (
    BaseConfidenceUpdater,
    SimpleConfidenceUpdater,
)
from pipeee.utils.id_pool import IdPool


class FinishStatus(Enum):
    UNFINISHED = 0
    YIELD_EOS = 1
    EXCEED_MAX_LEN = 2


@dataclass
class TrieNode:
    node_id: Optional[int] = None
    cache_id: Optional[int] = None
    token_id: Optional[int] = None
    position: Optional[int] = None
    exit_layer_idx: Optional[int] = None
    confidence: Optional[float] = None
    is_eos: Optional[bool] = None  # TODO: whether this is needed?
    parent: Optional["TrieNode"] = None
    children: dict[int, "TrieNode"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}


class EarlyExitedTokenTrie:
    def __init__(
        self,
        max_nodes_count: int,
        speculative_sampling_num: int,
        max_position: int,
        confidence_updater: BaseConfidenceUpdater = None,
    ):
        assert max_nodes_count > 0, "max_nodes_count must be positive"
        assert speculative_sampling_num > 0, "speculative_sampling_num must be positive"
        max_nodes_count_with_extra = max_nodes_count * (1 + speculative_sampling_num)
        # 使用 IdPool 替代 Counter，管理 node_id 的分配和回收（无限模式，确保无 id 冲突）
        self.node_id_pool: IdPool = IdPool(initial_capacity=max_nodes_count_with_extra * 100)
        self.root: TrieNode = None
        self.max_position = max_position
        self._node_map: dict[int, TrieNode] = {}
        self._cache_id_pool: IdPool = IdPool(max_size=max_nodes_count + 10)
        self._uncomputed_nodes: set[int] = set()
        self._confidence_updater = confidence_updater if confidence_updater is not None else SimpleConfidenceUpdater()
        self.max_nodes_count = max_nodes_count

    def __len__(self) -> int:
        return len(self._node_map)

    def __getitem__(self, node_id: int) -> TrieNode:
        return self._node_map[node_id]

    def destroy_branch(self, node_id: int):
        if node_id not in self._node_map:
            return
        node = self._node_map.pop(node_id)
        if node.cache_id is not None:
            self._cache_id_pool.release(node.cache_id)
        if node_id in self._uncomputed_nodes:
            self._uncomputed_nodes.remove(node_id)
        # 释放 node_id 到池中，以便复用
        self.node_id_pool.release(node_id)
        for child in node.children.values():
            self.destroy_branch(child.node_id)
        node.children.clear()

    def is_root(self, node_id: Optional[int]) -> bool:
        return self.root is None or self.root.node_id == node_id

    def update(self, token_id: int, position: Optional[int] = None) -> Optional[int]:
        reset = False
        root_cache_id = None
        if self.root is None or token_id not in self.root.children:
            reset = True

        if reset:
            if self.root is not None:
                position = self.root.position + 1
                self._cache_id_pool.reset()
                # 释放所有节点的 node_id 到池中
                for node_id in list(self._node_map.keys()):
                    self.node_id_pool.release(node_id)
                self._node_map.clear()
                self._uncomputed_nodes.clear()
                root_cache_id = self.root.cache_id

            assert position is not None, "Position must be provided when resetting the trie."
            new_cache_id = self._cache_id_pool.acquire()
            self.root = TrieNode(
                node_id=self.node_id_pool.acquire(),
                cache_id=new_cache_id,
                token_id=token_id,
                position=position,
                exit_layer_idx=-1,
                confidence=1.0,
                is_eos=False,  # TODO: to verify whether this is needed
                parent=None,
            )
            self._node_map[self.root.node_id] = self.root
            self._uncomputed_nodes.add(self.root.node_id)
        else:
            root_cache_id = self.root.cache_id
            self._cache_id_pool.release(self.root.cache_id)
            cur_node = self.root
            self.root = cur_node.children.pop(token_id)
            self.root.parent = None
            self.root.confidence = 1.0
            self.destroy_branch(cur_node.node_id)
        return root_cache_id

    def add_token(
        self,
        token_id: int,
        position: int,
        exit_layer_idx: int,
        confidence: float,
        is_eos: bool,
        parent: TrieNode = None,
    ):
        if token_id in parent.children:
            self._confidence_updater.update_confidence(
                parent.children[token_id],
                confidence,
            )
            return -1

        node_id = self.node_id_pool.acquire()
        assert node_id not in self._node_map, "Node ID already exists in the trie."
        new_node = TrieNode(
            node_id=node_id,
            cache_id=None,
            token_id=token_id,
            position=position,
            exit_layer_idx=exit_layer_idx,
            confidence=confidence,
            is_eos=is_eos,
            parent=parent,
        )
        self._node_map[node_id] = new_node
        parent.children[new_node.token_id] = new_node
        if not is_eos:
            self._uncomputed_nodes.add(node_id)

    def add_tokens(
        self,
        parent_node_id: int,
        token_ids: list[int],
        exit_layer_idx: int,
        confidences: list[float],
        is_eos: list[bool],
    ) -> bool:
        if parent_node_id not in self._node_map:
            return FinishStatus.UNFINISHED
        parent_node = self._node_map[parent_node_id]
        position = parent_node.position + 1
        if position >= self.max_position:
            return FinishStatus.EXCEED_MAX_LEN

        for token_id, conf, eos in zip(token_ids, confidences, is_eos, strict=False):
            if eos:
                return FinishStatus.YIELD_EOS
            self.add_token(
                token_id=token_id,
                position=position,
                exit_layer_idx=exit_layer_idx,
                confidence=conf,
                is_eos=eos,
                parent=parent_node,
            )
        return FinishStatus.UNFINISHED

    def prune(self):
        if num_nodes := len(self._node_map) <= self.max_nodes_count:
            return
        cumulative_confidences: dict[int, float] = {}

        def traverse(node: TrieNode, cum_con: float):
            nonlocal cumulative_confidences
            cum_con *= node.confidence
            cumulative_confidences[node.node_id] = cum_con
            for child in node.children.values():
                traverse(child, cum_con)

        traverse(self.root, 1.0)

        # sort nodes by cumulative confidence
        k = self.max_nodes_count - num_nodes
        smallest_ids = heapq.nsmallest(k, cumulative_confidences.items(), key=lambda x: x[1])

        for node_id, _ in smallest_ids:
            node = self._node_map.pop(node_id)
            del node.parent.children[node.token_id]
            if node.cache_id is not None:
                self._cache_id_pool.release(node.cache_id)
            if node_id in self._uncomputed_nodes:
                self._uncomputed_nodes.remove(node_id)
            # 释放 node_id 到池中，以便复用
            self.node_id_pool.release(node_id)

    def flatten(self):
        if not self._uncomputed_nodes:
            return [], [], [], [], []

        # only root is uncomputed, return directly
        if len(self._uncomputed_nodes) == 1 and self.root.node_id in self._uncomputed_nodes:
            root_node = self.root
            self._uncomputed_nodes.clear()
            return [self.root.node_id], [root_node.token_id], [-1], [root_node.cache_id], [root_node.position]

        uncomputed_nodes = list(self._uncomputed_nodes)
        self._uncomputed_nodes.clear()
        nodes_to_include: set[int] = set()
        parent_map: dict[int, int] = {}
        node_ids: list[int] = []

        # for each uncomputed node, trace back to the root and collect all nodes along the path
        for node_id in uncomputed_nodes:
            current_id: int = node_id
            _node_ids: list[int] = []
            while current_id != -1 and current_id not in nodes_to_include:
                _node_ids.append(current_id)
                nodes_to_include.add(current_id)
                node = self._node_map[current_id]
                parent_id: int = node.parent.node_id if node.parent else -1
                parent_map[current_id] = parent_id
                current_id = parent_id
            node_ids.extend(reversed(_node_ids[1:]))  # exclude the uncomputed node itself

        node_ids.extend(uncomputed_nodes)
        token_ids: list[int] = [self._node_map[nid].token_id for nid in uncomputed_nodes]
        position_ids: list[int] = [self._node_map[nid].position for nid in uncomputed_nodes]

        assert node_ids[0] == self.root.node_id, "The first node must be the root"
        node_id_to_index: dict[int, int] = {node_id: idx for idx, node_id in enumerate(node_ids)}
        node_id_to_index[-1] = -1  # for root's parent

        cache_ids: list[int] = []
        parent_indices: list[int] = []

        for node_id in node_ids:
            node = self._node_map[node_id]
            if node.cache_id is None:
                node.cache_id = self._cache_id_pool.acquire()
            cache_ids.append(node.cache_id)

            parent_id = parent_map[node_id]
            parent_indices.append(node_id_to_index[parent_id])

        return uncomputed_nodes, token_ids, parent_indices, cache_ids, position_ids

    def next_token_candidates(self) -> tuple[list[int], list[float]]:
        tokens: list[int] = []
        confidences: list[float] = []
        for child in self.root.children.values():
            tokens.append(child.token_id)
            confidences.append(child.confidence)
        return tokens, confidences

    def to_json_tree(self) -> dict:
        if self.root is None:
            return {}

        def _node_to_dict(node: TrieNode) -> dict:
            token_str = str(node.token_id) if node.token_id is not None else "None|None|None"
            confidence_str = str(node.confidence) if node.confidence is not None else "None"
            key = f"{token_str}|{node.node_id}|{confidence_str}"

            children_list = []
            for token_id in sorted(node.children.keys()):
                child = node.children[token_id]
                children_list.append(_node_to_dict(child))

            return {key: children_list}

        return _node_to_dict(self.root)
