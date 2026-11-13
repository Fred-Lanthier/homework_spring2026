from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple
from itertools import batched

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    # Optional debug
    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
    # TODO(student): yield RolloutBatch minibatches of size minibatch_size.
    # Requirements:
    # - Let N = batch.input_ids.shape[0] be the number of sampled completions.
    # - If shuffle=True, permute indices with torch.randperm using the provided generator.
    # - Otherwise iterate in the original order 0, 1, ..., N-1.
    # - Slice ALL tensor fields consistently with the same minibatch indices.
    # - Keep task_names / completion_texts aligned with the same indices when present.
    # - If device is not None, move the minibatch to that device before yielding.
    N = batch.input_ids.shape[0]
    if shuffle:
        if generator is not None:
            iter_list = torch.randperm(N, device=generator.device, generator=generator).to(batch.input_ids.device)
        else:
            iter_list = torch.randperm(N).to(batch.input_ids.device)
    else:
        iter_list = torch.arange(N).to(batch.input_ids.device)

    for start in range(0, N, minibatch_size):
        if start + minibatch_size < N:
            idx = iter_list[start:start+minibatch_size]
        else:
            idx = iter_list[start:]

        mini_batch: RolloutBatch = RolloutBatch(
            input_ids=batch.input_ids[idx, :],
            attention_mask = batch.attention_mask[idx, :],
            completion_mask = batch.completion_mask[idx, :],
            old_logprobs = batch.old_logprobs[idx, :],
            ref_logprobs = batch.ref_logprobs[idx, :],
            rewards = batch.rewards[idx],
            advantages = batch.advantages[idx],
            task_names = None,
            completion_texts = None,
        )

        if batch.task_names is not None:
            mini_batch.task_names = []
            for i in idx.tolist():
                mini_batch.task_names.append(batch.task_names[i])

        if batch.completion_texts is not None:
            mini_batch.completion_texts = []
            for i in idx.tolist():
                mini_batch.completion_texts.append(batch.completion_texts[i])

            
        if device is not None:
            mini_batch = mini_batch.to(device)
        yield mini_batch
    
