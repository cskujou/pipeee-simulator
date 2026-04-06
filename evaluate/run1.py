import sys

from pipeee.llm import LLM, LLMConfig
from pipeee.logger import configure_logging

configure_logging(log_level="DEBUG", log_file="logs/run1_debug.log", disable_console=True)

RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def stream_requests(llm: LLM, requests: list[str]):
    request_texts = dict.fromkeys(range(len(requests)), "")
    finished = set()

    for output in llm.stream(max_steps=100):
        if output.finished:
            finished.add(output.req_id)
            if len(finished) == len(requests):
                break
            continue

        request_texts[output.req_id - 1] += output.text

        sys.stdout.write("\033[2J\033[H")  # Clear screen

        for i, (text, prompt) in enumerate(zip(request_texts.values(), requests, strict=True)):
            req_num = i + 1
            status = "FINISHED" if req_num in finished else "running"
            sys.stdout.write(f"{'=' * 60}\n")
            sys.stdout.write(f"[Request {req_num}] [{status}]\n")
            sys.stdout.write(f"{'=' * 60}\n\n")
            sys.stdout.write(f"{YELLOW}(User){RESET} {prompt}\n")
            sys.stdout.write(f"{YELLOW}(Assistant){RESET} {text}")
            if status == "running":
                sys.stdout.write(" ▌")
            sys.stdout.write("\n\n")

        sys.stdout.flush()

    sys.stdout.write("\033[2J\033[H")  # Clear screen

    for i, (text, prompt) in enumerate(zip(request_texts.values(), requests, strict=True)):
        req_num = i + 1
        status = "FINISHED" if req_num in finished else "running"
        sys.stdout.write(f"{'=' * 60}\n")
        sys.stdout.write(f"[Request {req_num}] [{status}]\n")
        sys.stdout.write(f"{'=' * 60}\n\n")
        sys.stdout.write(f"{YELLOW}(User){RESET} {prompt}\n")
        sys.stdout.write(f"{YELLOW}(Assistant){RESET} {text}")
        if status != "FINISHED":
            sys.stdout.write(f"\n\n{RED}(Stopped due to max_steps limit){RESET}")
        sys.stdout.write("\n\n")


if __name__ == "__main__":
    config = LLMConfig.of(
        pp_size=3,
        model_path="HuggingFaceTB/SmolLM-135M-Instruct",
        scheduler_budget_cls="default",
        device="cuda",
        topk=5,
        spec_topk=2,
        max_batch_size=4,
    )
    llm = LLM(config)

    requests = ["Hello, I'm Sun.", "What is the capital of China?", "Solve the equation x + 2 = 5."]

    llm.add_request(requests)

    stream_requests(llm, requests)