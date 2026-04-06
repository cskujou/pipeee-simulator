import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeee.config import LLMConfig
from pipeee.executor import Executor
from pipeee.modeling_io import DecodeInput, PrefillOutput, TokenizedRequest
from pipeee.sched.scheduler import Scheduler
from pipeee.logger import get_logger

logger = get_logger(__name__)


count = 1

class PrefillEngine:
    def __init__(self, config: LLMConfig):
        logger.info(f"Initializing PrefillEngine with model: {config.model_path}, device: {config.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.device = config.device
        self.model = AutoModelForCausalLM.from_pretrained(config.model_path).to(self.device)
        self.scheduler = Scheduler(config)
        self.config = config

        self.requests = []
        logger.debug("PrefillEngine initialized successfully")
    

    def add_request(self, requests: str | list[str]):
        if isinstance(requests, str):
            requests = [requests]

        logger.info(f"Adding {len(requests)} request(s) to PrefillEngine")
        # requests: list[str]
        for i, request in enumerate(requests):
            inputs = self.tokenizer(request)
            self.requests.append(inputs)
            logger.debug(f"Added request {i+1}: {repr(request[:50])}{'...' if len(request) > 50 else ''}")
        logger.debug(f"Total requests in PrefillEngine: {len(self.requests)}")

            
    def step(self) -> list[PrefillOutput]:
        logger.info(f"Starting prefill step for {len(self.requests)} request(s)")
        prefill_outputs = []

        for i, inputs in enumerate(self.requests):
            logger.debug(f"Processing prefill request {i+1} with {len(inputs['input_ids'])} tokens")
            input_ids_tensor = torch.tensor([inputs['input_ids']], device=self.device)
            output = self.model(input_ids=input_ids_tensor)
            logits, past_key_values = output.logits, output.past_key_values

            topk_values, topk_token_ids = torch.topk(logits[:, -1, :], k=self.config.topk, dim=-1)
            # softmax and sample
            probabilities = torch.softmax(topk_values, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)
            next_token_id = topk_token_ids.gather(-1, next_token_id)

            prefill_outputs.append(
                PrefillOutput(
                    input_ids=inputs['input_ids'],
                    first_token_id=next_token_id.squeeze().item(),
                    past_key_values=past_key_values,
                )
            )
            logger.debug(f"Prefill request {i+1} completed, next token: {next_token_id.squeeze().item()}")

        logger.info(f"Prefill step completed, generated {len(prefill_outputs)} prefill outputs")
        return prefill_outputs

        
        
        
class DecodeEngine:
    def __init__(self, config: LLMConfig):
        logger.info(f"Initializing DecodeEngine with model: {config.model_path}, pp_size: {config.pp_size}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.executor = Executor(config)
        self.req_id_counter = 1
        self.scheduler = Scheduler(config)
        self.config = config
        logger.debug("DecodeEngine initialized successfully")

    def get_next_req_id(self) -> int:
        req_id = self.req_id_counter
        self.req_id_counter += 1
        logger.debug(f"Generated next request ID: {req_id}")
        return req_id


    def add_request(self, prefill_outputs: list[PrefillOutput]):
        logger.info(f"Adding {len(prefill_outputs)} prefill output(s) to DecodeEngine")
        for i, request in enumerate(prefill_outputs):
            req_id = self.get_next_req_id()
            self.scheduler.add_request(DecodeInput.from_prefill_output(request, req_id), self.config)
            logger.debug(f"Added prefill output {i+1} with req_id: {req_id}")

    def step(self):
        logger.info("Starting decode step")
        schedule_output = self.scheduler.schedule()
        req_count = len(schedule_output.requests) if schedule_output.requests else 0
        logger.debug(f"Scheduler output: {req_count} request(s) scheduled")

        executor_output = self.executor.execute_model_step(schedule_output)
        logger.debug(f"Executor output: {len(executor_output)} sampled output(s)")

        self.scheduler.update_from_output(executor_output)

        finished_requests = self.scheduler.pop_finished_requests()
        
        tree_json = self.scheduler.running[0].trie.to_json_tree() if self.scheduler.running else None
        
        global count
        with open(f"/home/kujou/tree/trie_debug_{count}.json", "w") as f:
            json.dump(tree_json, f, indent=2)
        count += 1
        
        logger.info(f"Decode step completed, {len(finished_requests)} request(s) finished")

        return finished_requests
