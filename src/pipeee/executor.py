from transformers import AutoTokenizer

from pipeee.buffer import Buffer
from pipeee.logger import get_logger
from pipeee.modeling_io import SampledOutput, WorkerOutput
from pipeee.worker.models.modeling_llama import LlamaForCausalLM
from pipeee.worker.worker import Worker

logger = get_logger(__name__)


class Executor:
    def __init__(
        self,
        config,
    ):
        logger.info(f"Initializing Executor with pp_size: {config.pp_size}, model: {config.model_path}")
        self.pp_size = config.pp_size
        self.model_path = config.model_path
        self.workers = []
        self.model = LlamaForCausalLM.from_pretrained(config.model_path, use_cache=False).to(config.device)
        self.model.eval()
        self.input_buffer = Buffer(num_buffers=config.pp_size + 1)
        self.output_buffer = Buffer(num_buffers=config.pp_size + 1)
        self.init_workers(config)
        logger.debug(f"Executor initialized successfully with {len(self.workers)} worker(s)")

    def init_workers(self, config):
        logger.debug(f"Initializing {self.pp_size} worker(s)")
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        eos_token_id = tokenizer.eos_token_id
        for i in range(self.pp_size):
            worker = Worker(
                worker_id=i,
                model=self.model,
                input_buffer=self.input_buffer,
                output_buffer=self.output_buffer,
                eos_token_id=eos_token_id,
                config=config,
            )
            self.workers.append(worker)
            logger.debug(f"Worker {i} initialized")

    def collect_worker_outputs(self) -> list[WorkerOutput]:
        outputs = []
        while not self.output_buffer.is_buffer_empty(-1):
            worker_output: WorkerOutput = self.output_buffer.get_from_buffer(-1)
            outputs.append(worker_output)
            logger.debug(f"Collected worker output: req_id={worker_output.req_id}, token_ids={worker_output.token_ids}")
        logger.debug(f"Total worker outputs collected: {len(outputs)}")
        return outputs

    def create_batch(self, schedule_output):
        logger.debug(f"Creating batch with {len(schedule_output.requests)} request(s)")
        for request in schedule_output.requests:
            request.input_ids.unsqueeze_(0).to(self.model.device)
            request.position_ids.unsqueeze_(0).to(self.model.device)
            request.spec_cache_position.unsqueeze_(0).to(self.model.device)
        return schedule_output

    def send_to_first_worker(self, batch):
        logger.debug(f"Sending batch to first worker with {len(batch.requests)} request(s)")
        # TODO: 暂时不去做 Continue Batching
        for request in batch.requests:
            self.input_buffer.add_to_buffer(0, request)

    def worker_step(self):
        logger.debug(f"Processing worker step for {len(self.workers)} worker(s)")
        # process each worker
        for worker in self.workers:
            worker.process()

        # send to next worker in pp group
        for i in range(self.pp_size):
            self.input_buffer.buffers[i] = self.output_buffer.buffers[i]
            self.output_buffer.buffers[i] = []

    def get_output(self, worker_outputs):
        output = []
        for worker_output in worker_outputs:
            output.append(
                SampledOutput(
                    req_id=worker_output.req_id,
                    token_ids=worker_output.token_ids,
                    confidences=worker_output.confidences,
                    node_idx=worker_output.node_idx,
                    is_early_exited=not worker_output.last_worker,
                    is_eos=worker_output.is_eos,
                    exit_layer_idx=worker_output.exit_layer_idx,
                )
            )
        return output

    def execute_model_step(self, schedule_output):
        logger.info(f"Executing model step with {len(schedule_output.requests) if schedule_output.requests else 0} request(s)")
        batch = self.create_batch(schedule_output)
        self.send_to_first_worker(batch)

        self.worker_step()

        worker_outputs = self.collect_worker_outputs()
        logger.debug(f"Model step completed, collected {len(worker_outputs)} worker output(s)")
        return self.get_output(worker_outputs)
        