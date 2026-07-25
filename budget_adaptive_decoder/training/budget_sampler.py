"""
Budget sampling distribution for Phases 3 and 4.

Per design document §3 (Issue 3.3 fix), the budget sampling uses a mixture
distribution to ensure adequate training signal across deployment-relevant budgets:

- 50% Beta(2, 2)  - mass concentrated around B=0.5 (general training)
- 20% Beta(1, 1)  - uniform distribution across [0, 1] (broad exploration)
- 15% Beta(5, 2)  - favoring higher budgets (desktop deployment)
- 15% point masses at {0.3, 0.5, 0.9} for phone/mid/desktop deployment classes

This addresses the original Beta(2,2) caveat: ~33% mass in [0.4, 0.6]
but little mass near deployment-relevant budgets (phone -> 0.3, desktop -> 0.9).
"""

import torch
from typing import Tuple


class MixtureBudgetSampler:
    """Sample budget from a mixture distribution (Issue 3.3 fix)."""

    def __init__(
        self,
        beta_priors: Tuple[Tuple[float, float, float], ...] = (
            (2.0, 2.0, 0.50),
            (1.0, 1.0, 0.20),
            (5.0, 2.0, 0.15),
        ),
        point_masses: Tuple[Tuple[float, float], ...] = (
            (0.3, 0.05),
            (0.5, 0.05),
            (0.9, 0.05),
        ),
    ):
        """
        Args:
            beta_priors: List of (alpha, beta, weight) for each Beta component.
            point_masses: List of (value, weight) for discrete deployment budgets.
        """
        self.beta_priors = beta_priors
        self.point_masses = point_masses

        total = sum(w for _, _, w in beta_priors) + sum(w for _, w in point_masses)
        self.components = []
        for a, b, w in beta_priors:
            self.components.append(("beta", a, b, w / total))
        for v, w in point_masses:
            self.components.append(("point", v, w / total))

    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """Sample budgets for the batch.

        Args:
            batch_size: Number of samples.
            device: Torch device.

        Returns:
            Tensor of shape [batch_size] with values in (0, 1].
        """
        if device is None:
            device = torch.device("cpu")

        component_types = [c[0] for c in self.components]
        weights = torch.tensor([c[-1] for c in self.components], device=device)
        choices = torch.multinomial(weights, batch_size, replacement=True)

        samples = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            comp_idx = choices[i].item()
            comp = self.components[comp_idx]
            if comp[0] == "beta":
                _, a, b, _ = comp
                x = torch._standard_gamma(torch.tensor(a, device=device))
                y = torch._standard_gamma(torch.tensor(b, device=device))
                samples[i] = x / (x + y)
            else:
                _, v, _ = comp
                samples[i] = float(v)

        return samples
