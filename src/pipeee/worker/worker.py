import torch

from pipeee.buffer import Buffer
from pipeee.config import LLMConfig
from pipeee.logger import get_logger
from pipeee.modeling_io import ModelInterim, WorkerOutput
from pipeee.worker.model_runner import ModelRunner
from pipeee.worker.models.modeling_llama import LlamaForCausalLM

logger = get_logger(__name__)


class Worker:
    def __init__(
        self,
        worker_id: int,
        model: LlamaForCausalLM,
        input_buffer: Buffer,
        output_buffer: Buffer,
        eos_token_id: int,
        config: LLMConfig,
    ):
        logger.info(f"Initializing Worker {worker_id} (pp_size: {config.pp_size})")
        self.worker_id = worker_id
        self.world_size = config.pp_size
        self.last_worker = self.worker_id == self.world_size - 1
        self.model = model
        self.device = model.device

        self.forward_layers = self.assign_layers()
        self.model_runner = ModelRunner(self.model, self.forward_layers)
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer

        self.eos_token_id = eos_token_id

        self.k = config.topk if self.last_worker else config.request_config.spec_topk
        logger.debug(f"Worker {self.worker_id} initialized with layers: {self.forward_layers}, topk: {self.k}")

    def assign_layers(self) -> tuple[int, int]:
        total_layers = self.model.config.num_hidden_layers
        layers_per_worker = total_layers // self.world_size
        start_layer = self.worker_id * layers_per_worker
        if self.worker_id == self.world_size - 1:
            end_layer = total_layers
        else:
            end_layer = start_layer + layers_per_worker
        return (start_layer, end_layer)

    def send_to_next_worker(self, data):
        self.output_buffer.add_to_buffer(self.worker_id + 1, data)

    def receive_from_prev_worker(self):
        return self.input_buffer.get_from_buffer(self.worker_id)

    def send_to_executor(self, data):
        self.output_buffer.add_to_buffer(-1, data)

    def is_last_worker(self) -> bool:
        return self.forward_layers[1] == self.model.config.num_hidden_layers

    def sample_topk(self, logits):
        confidences, sampled_token_ids = torch.topk(logits, self.k, dim=-1)
        if self.last_worker:
            normalized_confidences = torch.softmax(confidences, dim=-1)
            next_token_id = torch.multinomial(normalized_confidences.squeeze(0), num_samples=1)
            sampled_token_ids = torch.gather(sampled_token_ids, -1, next_token_id.unsqueeze(0))
            confidences = torch.ones_like(sampled_token_ids, dtype=torch.float32)
        return confidences, sampled_token_ids

    def process(self):
        logger.debug(f"Worker {self.worker_id} starting to process inputs")
        count = 0
        while not self.input_buffer.is_buffer_empty(self.worker_id):
            input_data = self.receive_from_prev_worker()
            logger.debug(f"Worker {self.worker_id} processing req_id={input_data.req_id}")

            output_data: ModelInterim = self.model_runner.forward(input_data)

            if not self.is_last_worker():
                self.send_to_next_worker(output_data)

            logits = self.model_runner.get_logits(output_data, logits_to_keep=output_data.num_tokens)
            confidences, sampled_token_ids = self.sample_topk(logits)

            worker_output = WorkerOutput(
                req_id=output_data.req_id,
                last_worker=self.last_worker,
                token_ids=sampled_token_ids.squeeze(0).tolist(),
                node_idx=output_data.node_idx,
                confidences=confidences.squeeze(0).tolist(),
                is_eos=(sampled_token_ids.squeeze(0) == self.eos_token_id).tolist(),
                exit_layer_idx=self.forward_layers[1] - 1,
            )
            self.send_to_executor(worker_output)
            logger.debug(f"Worker {self.worker_id} sent output for req_id={output_data.req_id}: token_ids={worker_output.token_ids}")
            count += 1

        if count > 0:
            logger.info(f"Worker {self.worker_id} processed {count} input(s)")
