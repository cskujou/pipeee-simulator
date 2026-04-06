from pipeee.llm import LLM, LLMConfig
from pipeee.logger import configure_logging

if __name__ == "__main__":
    configure_logging(
        # log_level="DEBUG",
        log_level="INFO",
    )
    config = LLMConfig.of(
        pp_size=3,
        model_path="HuggingFaceTB/SmolLM-135M-Instruct",
        scheduler_budget_cls="default",
        device="cuda",
        topk=5,
        spec_topk=2,
    )
    world_size = 4
    llm = LLM(config)

    requests = ["Hello, my name is", "The weather today is"]

    llm.add_request(requests)

    llm.run()
