from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeee.request_trie import TrieNode
    


class BaseConfidenceUpdater(ABC):
    @abstractmethod
    def update_confidence(
        self,
        origin_node: "TrieNode",
        confidence: float
    ):
        raise NotImplementedError
    
    
class SimpleConfidenceUpdater(BaseConfidenceUpdater):
    def update_confidence(
        self,
        origin_node: "TrieNode",
        confidence: float
    ):
        origin_node.confidence = confidence