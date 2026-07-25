"""
Scheduler module for Budget-Constrained Neural Decoder.

Implements the cumulative ΔQ maximization scheduler as specified in Section 6.4
of the design document.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class Scheduler(nn.Module):
    """
    The Scheduler turns predicted marginal quality gains into an execution depth.

    It selects the feasible depth that maximizes total predicted quality gain
    (the sum of marginal gains up to that depth) subject to the budget constraint.

    Decision rule:
        k* = argmax_{k in {0,...,K} : C_0 + sum_{t=1}^{k} C_t <= B * C_total} sum_{t=1}^{k} ΔQ(t)

    Note: k=0 represents executing no refinement stages (base reconstruction only, cost C_0).
          C_0 is the base reconstruction cost; stage_costs[0] = C_0.
    """

    def __init__(self, stage_costs: List[float]):
        """
        Args:
            stage_costs: List of K+1 costs [C_0, C_1, C_2, ..., C_K].
                        C_0 is the base reconstruction cost.
                        C_1..C_K are refinement stage costs, must satisfy C_1 >= C_2 >= ... >= C_K.
        """
        super().__init__()
        self.K = len(stage_costs) - 1
        self.register_buffer(
            "stage_costs", torch.tensor(stage_costs, dtype=torch.float32)
        )
        self.C_0 = stage_costs[0]
        self.C_total = sum(stage_costs)

    def select_depth(
        self, delta_q: torch.Tensor, budget: float
    ) -> torch.Tensor:
        """
        Select the execution depth given predicted marginal gains and budget.

        Args:
            delta_q: Tensor of shape [batch, K] containing predicted marginal
                     quality gains ΔQ(1) through ΔQ(K).
            budget: Runtime budget ratio B in (0, 1]. The maximum fraction of
                   full decoder compute available.

        Returns:
            Tensor of shape [batch] containing selected depth k* (0 to K).
            k=0 means base reconstruction only (no refinement stages).
            k=K means all K stages are executed.
        """
        if delta_q.dim() == 1:
            delta_q = delta_q.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = delta_q.shape[0]
        device = delta_q.device

        stage_costs = self.stage_costs.to(device)
        budget_limit = budget * self.C_total

        stage_costs_only = self.stage_costs[1:]
        cumulative_costs = stage_costs_only.cumsum(dim=0)
        full_costs = torch.zeros(self.K + 1, device=device)
        full_costs[0] = self.C_0
        full_costs[1:] = self.C_0 + cumulative_costs

        best_depths = torch.zeros(batch_size, dtype=torch.long, device=device)
        best_values = torch.full(
            (batch_size,), float("-inf"), device=device
        )

        for depth in range(self.K + 1):
            depth_cumulative_cost = full_costs[depth]
            feasible_mask = (depth_cumulative_cost <= budget_limit + 1e-9)
            if not isinstance(feasible_mask, torch.Tensor):
                feasible_mask = torch.tensor(feasible_mask, device=device)
            feasible_mask = feasible_mask.expand(batch_size)

            if depth == 0:
                depth_value = torch.zeros(batch_size, device=device)
            else:
                depth_value = delta_q[:, :depth].sum(dim=1)

            update_mask = feasible_mask & (depth_value > best_values)
            best_depths[update_mask] = depth
            best_values[update_mask] = depth_value[update_mask]

        if squeeze_output:
            return best_depths.squeeze(0)
        return best_depths

    def forward(
        self, delta_q: torch.Tensor, budget: float
    ) -> torch.Tensor:
        """Forward pass alias for select_depth."""
        return self.select_depth(delta_q, budget)


def cumulative_delta_q(delta_q: torch.Tensor) -> torch.Tensor:
    """
    Compute cumulative sum of ΔQ values.

    cumulative[k] = sum_{t=1}^{k} ΔQ(t) = Q(k) - Q(0)

    Args:
        delta_q: Tensor of shape [..., K] containing marginal gains.

    Returns:
        Tensor of shape [..., K] containing cumulative gains at each depth.
    """
    return torch.cumsum(delta_q, dim=-1)


def verify_stage_costs_monotonic(stage_costs: List[float]) -> bool:
    """
    Verify that refinement stage costs are non-increasing: C_1 >= C_2 >= ... >= C_K.
    C_0 (base reconstruction cost) is excluded from this check.

    Args:
        stage_costs: List of K+1 costs [C_0, C_1, ..., C_K].

    Returns:
        True if costs are valid, False otherwise.
    """
    for i in range(1, len(stage_costs) - 1):
        if stage_costs[i] < stage_costs[i + 1]:
            return False
    return True


if __name__ == "__main__":
    import torch.nn.functional as F

    K = 4
    stage_costs = [1.0, 4.0, 2.0, 2.0, 1.0]
    C_total = sum(stage_costs)
    C_0 = stage_costs[0]

    scheduler = Scheduler(stage_costs)

    print("=" * 60)
    print("Scheduler Unit Tests")
    print("=" * 60)

    print("\n1. Test: Budget allows all stages (B=1.0)")
    delta_q = torch.tensor([[3.0, 1.5, 1.0, 0.5]])
    budget = 1.0
    depth = scheduler.select_depth(delta_q, budget)
    print(f"   dQ = {delta_q.squeeze().tolist()}")
    print(f"   Budget B = {budget}")
    print(f"   Costs = {stage_costs} (C_total = {C_total}, C_0 = {C_0})")
    print(f"   Full costs = [1, 5, 7, 9, 10]")
    print(f"   Budget limit = {budget * C_total}")
    print(f"   Selected depth = {depth.item()}")
    print(f"   Expected: depth = 4 (all stages, since budget allows)")
    assert depth.item() == 4, f"Expected 4, got {depth.item()}"

    print("\n2. Test: Budget allows only base (B*C_total < C_0)")
    budget = 0.3
    depth = scheduler.select_depth(delta_q, budget)
    print(f"   Budget B = {budget}, limit = {budget * C_total}")
    print(f"   Selected depth = {depth.item()}")
    print(f"   Expected: depth = 0 (only base fits, cost C_0=1.0 <= 3.0)")
    assert depth.item() == 0, f"Expected 0, got {depth.item()}"

    print("\n3. Test: Budget allows base and first stage (B=0.5, limit=5.0)")
    budget = 0.5
    depth = scheduler.select_depth(delta_q, budget)
    print(f"   Budget B = {budget}, limit = {budget * C_total}")
    print(f"   Costs: [1, 5, 7, 9, 10]")
    print(f"   Selected depth = {depth.item()}")
    print(f"   Expected: depth = 1 (C_0+stage1=5.0 fits exactly at limit)")
    assert depth.item() == 1, f"Expected 1, got {depth.item()}"

    print("\n4. Test: Negative dQ at intermediate stage")
    delta_q_neg = torch.tensor([[3.0, -0.5, 1.5, 0.5]])
    budget = 1.0
    depth = scheduler.select_depth(delta_q_neg, budget)
    print(f"   dQ = {delta_q_neg.squeeze().tolist()}")
    print(f"   Cumulative dQ at each depth: [0, 3, 2.5, 4.0, 4.5]")
    print(f"   Selected depth = {depth.item()}")
    print(f"   Expected: depth = 4 (cumulative 4.5 > 4.0 > 3.0 > 2.5 > 0)")
    assert depth.item() == 4, f"Expected 4, got {depth.item()}"

    print("\n5. Test: Batch processing")
    delta_q_batch = torch.tensor([
        [3.0, 1.5, 1.0, 0.5],
        [1.0, 0.5, 0.3, 0.2],
        [4.0, 0.1, 0.1, 0.1],
    ])
    budgets = torch.tensor([0.3, 0.6, 0.9])
    depths = scheduler.select_depth(delta_q_batch, budgets)
    print(f"   Batch dQ shape: {delta_q_batch.shape}")
    print(f"   Budgets: {budgets.tolist()}")
    print(f"   Budget limits: {[b * C_total for b in budgets.tolist()]}")
    print(f"   Selected depths: {depths.tolist()}")
    expected = [0, 1, 3]
    assert depths.tolist() == expected, f"Expected {expected}, got {depths.tolist()}"

    print("\n6. Test: Edge case - budget at C_0/(C_0+C_full) boundary (B = 1/10 = 0.1)")
    budget = 0.1
    depth = scheduler.select_depth(delta_q, budget)
    print(f"   Budget = {budget:.6f}, limit = {budget * C_total}")
    print(f"   C_0 = {C_0}, C_total = {C_total}")
    print(f"   Selected depth = {depth.item()}")
    print(f"   Expected: depth = 0 (C_0=1.0 fits exactly at boundary)")

    print("\n7. Test: Resolution invariance (single sample as 1D tensor)")
    delta_q_1d = torch.tensor([3.0, 1.5, 1.0, 0.5])
    budget = 0.5
    depth = scheduler.select_depth(delta_q_1d, budget)
    print(f"   1D input dQ shape: {delta_q_1d.shape}")
    print(f"   Output depth: {depth.item()}")
    assert depth.item() == 1

    print("\n" + "=" * 60)
    print("All scheduler tests passed!")
    print("=" * 60)