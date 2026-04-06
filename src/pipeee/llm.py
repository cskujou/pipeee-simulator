from collections.abc import Generator
from dataclasses import dataclass

from pipeee.config import LLMConfig
from pipeee.engine import DecodeEngine, PrefillEngine


@dataclass
class StreamOutput:
    req_id: int
    text: str
    finished: bool


class LLM:
    def __init__(self, config: LLMConfig):
        self.prefill_engine = PrefillEngine(config)
        self.decode_engine = DecodeEngine(config)
        self.requests = []
        self._tokenizer = self.prefill_engine.tokenizer
        self._last_output_counts: dict[int, int] = {}
        self._req_id_counter = 1

    def get_next_req_id(self) -> int:
        req_id = self._req_id_counter
        self._req_id_counter += 1
        return req_id

    def add_request(self, requests: str | list[str]):
        if isinstance(requests, str):
            requests = [requests]
        self.requests.extend(requests)
        req_ids = [self.get_next_req_id() for _ in requests]
        self.prefill_engine.add_request(requests, req_ids)

    def run(self, step=None):
        prefill_outputs = self.prefill_engine.step()
        self.decode_engine.add_request(prefill_outputs)
        if step is None:
            while True:
                self.decode_engine.step()
        else:
            for _ in range(step):
                self.decode_engine.step()

    def stream(self, max_steps: int = 100) -> Generator[StreamOutput, None, None]:
        prefill_outputs = self.prefill_engine.step()
        self.decode_engine.add_request(prefill_outputs)

        for po in prefill_outputs:
            self._last_output_counts[po.req_id] = 0

        for _ in range(max_steps):
            step_output = self.decode_engine.step()

            for req_id, output_token_ids in step_output.active_outputs.items():
                new_token_count = len(output_token_ids)
                last_count = self._last_output_counts.get(req_id, 0)

                if new_token_count > last_count:
                    new_tokens = output_token_ids[last_count:]
                    text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
                    self._last_output_counts[req_id] = new_token_count
                    yield StreamOutput(req_id=req_id, text=text, finished=False)

            for request in step_output.finished_requests:
                req_id = request.req_id
                yield StreamOutput(req_id=req_id, text="", finished=True)
                if req_id in self._last_output_counts:
                    del self._last_output_counts[req_id]

            if not step_output.active_outputs:
                break