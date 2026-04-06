from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SchedulerBudgetConfig:
    scheduler_budget_cls: type | str
    max_batch_size: int
    memory_budget: Optional[int]
    compute_budget: Optional[int]


@dataclass
class RequestConfig:
    spec_topk: int
    eos_token_id: Optional[int]
    max_nodes_count: int
    max_length: int
    max_new_tokens: Optional[int] = None
    

@dataclass
class LLMConfig:
    pp_size: int
    model_path: str
    device: torch.device
    topk: int
    sche_budget_config: SchedulerBudgetConfig
    request_config: RequestConfig

    @staticmethod
    def of(
        pp_size: int,
        model_path: str,
        memory_budget: Optional[int] = None,
        compute_budget: Optional[int] = None,
        scheduler_budget_cls: type | str = "default",
        device: str | torch.device = "cuda",
        max_batch_size: int = 1,
        topk: int = 5,
        spec_topk: int = 3,
        max_nodes_count: Optional[int] = None,
        max_length: int = 2048,
        max_new_tokens: Optional[int] = None,
        eos_token_id: Optional[int] = None,

    ) -> "LLMConfig":
        if isinstance(device, str):
            device = torch.device(device)
        
        sche_budget_config = SchedulerBudgetConfig(
            scheduler_budget_cls=scheduler_budget_cls,
            max_batch_size=max_batch_size,
            memory_budget=memory_budget,
            compute_budget=compute_budget,
        )
        request_config = RequestConfig(
            spec_topk=spec_topk,
            max_nodes_count=max_nodes_count or (spec_topk+1) ** (pp_size-1),
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        
        return LLMConfig(
            pp_size=pp_size,
            model_path=model_path,
            device=device,
            topk=topk,
            sche_budget_config=sche_budget_config,
            request_config=request_config,
        )


