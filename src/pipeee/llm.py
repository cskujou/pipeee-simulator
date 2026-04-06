from pipeee.config import LLMConfig
from pipeee.engine import DecodeEngine, PrefillEngine


class LLM:
    def __init__(self, config: LLMConfig):
        self.prefill_engine = PrefillEngine(config)
        self.decode_engine = DecodeEngine(config)
        self.requests = []

    def add_request(self, requests: str | list[str]):
        self.requests.extend(requests if isinstance(requests, list) else [requests])
        self.prefill_engine.add_request(requests)

    def run(self, step=None):
        prefill_outputs = self.prefill_engine.step()
        self.decode_engine.add_request(prefill_outputs)
        if step is None:
            while True:
                self.decode_engine.step()
        else:
            for _ in range(step):
                self.decode_engine.step()