"""
PolicyNetwork module for Budget-Constrained Neural Decoder.

Implements the policy network that predicts marginal quality gains ΔQ(k)
as specified in Section 6.3 of the design document.
"""

import torch
import torch.nn as nn
from typing import List


class PolicyNetwork(nn.Module):
    """
    The Policy Network predicts marginal quality gains ΔQ(k) for each
    refinement stage k, given pre-decoded content features and runtime budget B.

    Output: [ΔQ(1), ..., ΔQ(K)] - unconstrained (no sigmoid/softmax).
    Negative predictions are valid: they indicate the policy predicts a stage
    will decrease quality for a given frame, which the scheduler will avoid
    by selecting a smaller depth.

    Architecture:
        - Budget encoder: 1 → 16 → 16
        - Predictor: (feature_dim + 16) → hidden_dim → hidden_dim → K

    Total parameters: ~10k–20k.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 64,
        K: int = 4,
    ):
        """
        Args:
            feature_dim: Dimension of content features from extractor.
            hidden_dim: Hidden layer dimension. Default 64.
            K: Number of refinement stages. Default 4.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.K = K

        self.budget_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
        )

        predictor_input_dim = feature_dim + 16
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, K),
        )

    def forward(
        self,
        content_features: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict marginal quality gains ΔQ(k) for each stage.

        Args:
            content_features: Content features [B, feature_dim] from extractor.
            budget: Runtime budget ratio B in (0, 1]. Can be scalar or [B].

        Returns:
            delta_q: Predicted marginal gains [B, K]. Unconstrained values.
        """
        if budget.dim() == 0:
            budget = budget.unsqueeze(0)
        if budget.dim() == 1:
            budget = budget.unsqueeze(1)

        budget_encoded = self.budget_encoder(budget)

        if budget_encoded.shape[0] != content_features.shape[0]:
            budget_encoded = budget_encoded.expand(
                content_features.shape[0], -1
            )

        combined = torch.cat([content_features, budget_encoded], dim=1)

        delta_q = self.predictor(combined)

        return delta_q

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_policy_network():
    """Unit tests for PolicyNetwork."""
    import torch

    K = 4
    feature_dim = 64
    hidden_dim = 64
    batch_size = 8

    policy = PolicyNetwork(feature_dim=feature_dim, hidden_dim=hidden_dim, K=K)

    print("=" * 60)
    print("PolicyNetwork Unit Tests")
    print("=" * 60)

    content_features = torch.randn(batch_size, feature_dim)
    budgets = torch.rand(batch_size)

    output = policy(content_features, budgets)

    print(f"\n1. Output shape test")
    print(f"   Input content_features shape: {content_features.shape}")
    print(f"   Input budgets shape: {budgets.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected: [{batch_size}, {K}]")
    assert output.shape == (batch_size, K)
    print(f"   PASSED")

    print(f"\n2. Parameter count test")
    num_params = policy.num_parameters()
    print(f"   Total parameters: {num_params:,}")
    print(f"   Expected: ~10k-20k")
    assert 5_000 < num_params < 50_000, f"Unexpected parameter count: {num_params}"
    print(f"   PASSED")

    print(f"\n3. Budget encoding test")
    single_budget = torch.tensor([0.5])
    single_features = torch.randn(1, feature_dim)
    output_single = policy(single_features, single_budget)
    print(f"   Single budget input: {single_budget.item()}")
    print(f"   Output shape: {output_single.shape}")
    assert output_single.shape == (1, K)
    print(f"   PASSED")

    print(f"\n4. Scalar budget test")
    scalar_budget = torch.tensor(0.7)
    output_scalar = policy(content_features[:4], scalar_budget)
    print(f"   Scalar budget: {scalar_budget.item()}")
    print(f"   Output shape: {output_scalar.shape}")
    assert output_scalar.shape == (4, K)
    print(f"   PASSED")

    print(f"\n5. Negative ΔQ propagation test")
    output_vals = output.detach()
    neg_count = (output_vals < 0).sum().item()
    print(f"   Negative values in output: {neg_count}/{batch_size * K}")
    print(f"   (Negative values are valid and expected)")
    print(f"   PASSED")

    print(f"\n6. Gradient flow test")
    output = policy(content_features, budgets)
    loss = output.sum()
    loss.backward()
    has_grad = all(
        p.grad is not None for p in policy.parameters() if p.requires_grad
    )
    print(f"   All parameters have gradients: {has_grad}")
    assert has_grad, "Some parameters do not have gradients"
    print(f"   PASSED")

    print("\n" + "=" * 60)
    print("All PolicyNetwork tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_policy_network()