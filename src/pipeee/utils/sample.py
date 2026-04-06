from typing import Optional

import torch


def logits_to_next_tokens(
    next_token_logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    temperature = max(temperature, 1e-8)
    logits = next_token_logits / temperature

    # 数值稳定的softmax
    def stable_softmax(x: torch.Tensor) -> torch.Tensor:
        x_max = torch.max(x, dim=-1, keepdim=True).values
        x_exp = torch.exp(x - x_max)
        return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)

    probs = stable_softmax(logits)

    # 处理可能的数值异常，Nan替换为均匀分布
    probs = torch.nan_to_num(probs, nan=1.0 / probs.shape[-1])
    probs = probs / torch.sum(probs, dim=-1, keepdim=True)

    # top_k 过滤
    if top_k is not None and top_k > 0 and top_k < probs.shape[-1]:
        top_k_indices = torch.topk(probs, top_k, dim=-1).indices
        mask = torch.zeros_like(probs, dtype=torch.bool)
        for i in range(probs.shape[0]):
            mask[i, top_k_indices[i]] = True
        probs = probs * mask
        probs_sum = torch.sum(probs, dim=-1, keepdim=True)
        probs_sum = torch.where(probs_sum == 0, 1.0, probs_sum)
        probs = probs / probs_sum

    # 采样下一个token（保持原先的随机采样逻辑）
    batch_size = probs.shape[0]
    next_tokens = []
    for i in range(batch_size):
        token = torch.multinomial(probs[i], num_samples=1).item()
        next_tokens.append(token)

    next_tokens = torch.tensor(next_tokens, dtype=torch.int32).unsqueeze(-1)
    return next_tokens
