from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Optional

from transformers import DynamicCache

from pipeee.logger import get_logger
from pipeee.modeling_io import DecodeInput, ModelInput
from pipeee.request import Request
from pipeee.request_queue import RequestQueue
from pipeee.sched.scheduler_budget import SchedulerBudgetFactory
from pipeee.sched.spec_cache import SpecCache

logger = get_logger(__name__)


@dataclass
class SchedulerOutput:
    requests: Optional[list[ModelInput]]


class Scheduler:
    def __init__(self, config):
        logger.info(f"Initializing Scheduler with budget config: {config.sche_budget_config}")
        self.requests: dict[str, Request] = {}
        self.waiting = RequestQueue()
        self.running: list[Request] = []
        self.budget_factory = SchedulerBudgetFactory(config.sche_budget_config)

        self.kv_cache: dict[int, DynamicCache] = {}
        self.spec_kv_cache: dict[int, SpecCache] = defaultdict(lambda: SpecCache(config.request_config.max_nodes_count + 10))
        self.finished_requests: list[Request] = []
        logger.debug("Scheduler initialized successfully")

    def schedule(self) -> SchedulerOutput:
        logger.info(f"Starting scheduling: {len(self.running)} running, {self.waiting.size() if hasattr(self.waiting, 'size') else len(self.waiting)} waiting request(s)")
        scheduled_running_reqs: list[Request] = []
        scheduled_new_reqs: list[Request] = []
        budget = self.budget_factory()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and budget.pre_check():
            request = self.running[req_index]
            enough_budget = budget.post_check(request)
            if not enough_budget:
                logger.debug(f"Not enough budget to schedule running request: {request.req_id}")
                continue
            scheduled_running_reqs.append(request)
            budget.consume()
            logger.debug(f"Scheduled running request: {request.req_id}")
            req_index += 1

        # Next, schedule the WAITING requests.
        while self.waiting and budget.pre_check():
            request = self.waiting.peek_request()
            enough_budget = budget.post_check(request)
            if not enough_budget:
                logger.debug("Not enough budget to schedule waiting requests")
                break
            request = self.waiting.pop_request()
            self.running.append(request)
            scheduled_new_reqs.append(request)
            budget.consume()
            logger.debug(f"Scheduled new request: {request.req_id}")

        requests = []
        for req in chain(scheduled_running_reqs, scheduled_new_reqs):
            request = req.create_model_input()
            request.kv_cache = self.kv_cache[request.req_id]
            request.spec_cache = self.spec_kv_cache[request.req_id]
            requests.append(request)

        scheduler_output = SchedulerOutput(requests=requests)
        logger.info(f"Scheduling completed: {len(scheduled_running_reqs)} running, {len(scheduled_new_reqs)} new request(s)")
        return scheduler_output

    def add_request(self, decode_input: DecodeInput, config):
        logger.info(f"Adding request: req_id={decode_input.req_id}")
        self.kv_cache[decode_input.req_id] = decode_input.past_key_values
        request = Request.from_decode_input(decode_input, config=config)
        self.requests[request.req_id] = request
        self.waiting.add_request(request)
        logger.debug(f"Request added to waiting queue: req_id={request.req_id}")
        
    def transfer_completed_kv_cache(self, request, cache_position):
        logger.debug(f"Transferring completed KV cache for req_id={request.req_id}, cache_position={cache_position}")
        spec_cache = self.spec_kv_cache[request.req_id]
        completed_kv_cache = spec_cache.get(cache_position)
        kv_cache = self.kv_cache[request.req_id]
        for layer_idx, (key, value) in enumerate(completed_kv_cache):
            kv_cache.update(key, value, layer_idx)
        logger.debug(f"KV cache transferred successfully for req_id={request.req_id}")

    def update_from_output(self, output):
        logger.info(f"Updating scheduler from {len(output)} sampled output(s)")
        for sampled_output in output:
            req_id = sampled_output.req_id
            if req_id not in self.requests:
                logger.warning(f"Received output for unknown req_id: {req_id}")
                continue
            request = self.requests[req_id]

            if sampled_output.is_early_exited:
                logger.debug(f"Processing early exited output for req_id={req_id}")
                for i in range(len(sampled_output.token_ids)):
                    request.add_tokens(
                        parent_node_id=sampled_output.node_idx[i],
                        token_ids=sampled_output.token_ids[i],
                        exit_layer_idx=sampled_output.exit_layer_idx,
                        confidences=sampled_output.confidences[i],
                        is_eos=sampled_output.is_eos[i],
                    )
            else:
                logger.debug(f"Processing non-early exited output for req_id={req_id}")
                
                assert request.root_id is not None, f"Request {req_id} has no root_id set for non-early exited output"
                for i, nid in enumerate(sampled_output.node_idx):
                    if nid == request.root_id:
                        root_cache_id = request.update(
                            token_id=sampled_output.token_ids[i][0],
                            root_node_id=nid,
                        )
                        if root_cache_id is not None:
                            self.transfer_completed_kv_cache(request, root_cache_id)
                


    def pop_finished_requests(self) -> list[Request]:
        logger.info(f"Popping {len(self.finished_requests)} finished request(s)")
        for request in self.finished_requests:
            logger.debug(f"Cleaning up resources for req_id={request.req_id}")
            if request in self.requests:
                self.requests.pop(request.req_id)
            if request.req_id in self.kv_cache:
                del self.kv_cache[request.req_id]
            if request.req_id in self.spec_kv_cache:
                del self.spec_kv_cache[request.req_id]

        finished_requests = self.finished_requests
        self.finished_requests = []
        return finished_requests