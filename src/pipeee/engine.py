from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeee.config import LLMConfig
from pipeee.executor import Executor
from pipeee.logger import get_logger
from pipeee.modeling_io import DecodeInput, PrefillOutput
from pipeee.sched.scheduler import Scheduler

logger = get_logger(__name__)


@dataclass
class StepOutput:
    finished_requests: list
    active_outputs: dict[int, list[int]]  # req_id -> output_token_ids

class PrefillEngine:
    def __init__(self, config: LLMConfig):
        logger.info(f"Initializing PrefillEngine with model: {config.model_path}, device: {config.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.device = config.device
        self.model = AutoModelForCausalLM.from_pretrained(config.model_path).to(self.device)
        self.scheduler = Scheduler(config)
        self.config = config

        self.requests = []
        self.request_reqs = []  # Store original request strings for chat template
        logger.debug("PrefillEngine initialized successfully")

    def add_request(self, requests: str | list[str], req_ids: list[int]):
        if isinstance(requests, str):
            requests = [requests]

        logger.info(f"Adding {len(requests)} request(s) to PrefillEngine")
        for req_id, request in zip(req_ids, requests, strict=True):
            # Apply chat template
            chat_messages = [{"role": "user", "content": request}]
            inputs = self.tokenizer.apply_chat_template(chat_messages, add_generation_prompt=True, return_tensors="pt")
            self.requests.append(inputs)
            self.request_reqs.append(req_id)
            logger.debug(f"Added request {req_id}: {repr(request[:50])}{'...' if len(request) > 50 else ''}")
        logger.debug(f"Total requests in PrefillEngine: {len(self.requests)}")

            
    def step(self) -> list[PrefillOutput]:
        logger.info(f"Starting prefill step for {len(self.requests)} request(s)")
        prefill_outputs = []

        for i, inputs in enumerate(self.requests):
            req_id = self.request_reqs[i]
            input_ids = inputs.squeeze(0).tolist()  # Remove batch dimension
            logger.debug(f"Processing prefill request {req_id} with {len(input_ids)} tokens")
            input_ids_tensor = inputs.to(self.device)
            output = self.model(input_ids=input_ids_tensor)
            logits, past_key_values = output.logits, output.past_key_values

            topk_values, topk_token_ids = torch.topk(logits[:, -1, :], k=self.config.topk, dim=-1)
            # softmax and sample
            probabilities = torch.softmax(topk_values, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)
            next_token_id = topk_token_ids.gather(-1, next_token_id)

            prefill_outputs.append(
                PrefillOutput(
                    req_id=req_id,
                    input_ids=input_ids,
                    first_token_id=next_token_id.squeeze().item(),
                    past_key_values=past_key_values,
                )
            )
            logger.debug(f"Prefill request {req_id} completed, next token: {next_token_id.squeeze().item()}")

        logger.info(f"Prefill step completed, generated {len(prefill_outputs)} prefill outputs")
        return prefill_outputs

        
        
        
class DecodeEngine:
    def __init__(self, config: LLMConfig):
        logger.info(f"Initializing DecodeEngine with model: {config.model_path}, pp_size: {config.pp_size}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.executor = Executor(config)
        self.scheduler = Scheduler(config)
        self.config = config
        logger.debug("DecodeEngine initialized successfully")

    def add_request(self, prefill_outputs: list[PrefillOutput]):
        logger.info(f"Adding {len(prefill_outputs)} prefill output(s) to DecodeEngine")
        for request in prefill_outputs:
            req_id = request.req_id
            self.scheduler.add_request(DecodeInput.from_prefill_output(request), self.config)
            logger.debug(f"Added prefill output with req_id: {req_id}")

    def step(self):
        logger.info("Starting decode step")
        schedule_output = self.scheduler.schedule()
        req_count = len(schedule_output.requests) if schedule_output.requests else 0
        logger.debug(f"Scheduler output: {req_count} request(s) scheduled")

        executor_output = self.executor.execute_model_step(schedule_output)
        logger.debug(f"Executor output: {len(executor_output)} sampled output(s)")

        self.scheduler.update_from_output(executor_output)

        finished_requests = self.scheduler.pop_finished_requests()

        active_outputs = {
            req_id: request._output_token_ids
            for req_id, request in self.scheduler.requests.items()
        }

        logger.info(f"Decode step completed, {len(finished_requests)} request(s) finished")

        return StepOutput(finished_requests=finished_requests, active_outputs=active_outputs)

        return finished_requests
