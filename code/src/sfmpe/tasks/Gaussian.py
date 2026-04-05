import torch
from sfmpe.core.distributions import Distribution
from sfmpe.core.simulator import Simulator
from sfmpe.tasks.base_task import Task
from typing import Tuple


# -------------------- PRIOR --------------------

class GaussianPrior(Distribution):
    """
    Simple box-uniform distribution: independent uniforms per dimension
    """

    def __init__(self, low: torch.Tensor, high: torch.Tensor, device="cpu"):
        self.low = low.to(device)
        self.high = high.to(device)
        self.device = device

    def sample(self, size: int, **kwargs) -> torch.Tensor:
        u = torch.rand(size, self.low.shape[0], device=self.device)
        return self.low + (self.high - self.low) * u


# -------------------- SIMULATOR --------------------

class GaussianSimulator(Simulator):
    """
    Simulator for Gaussian time series:
    x_t ~ N(mu, sigma), i.i.d.
    """

    def __init__(self, n: int, device="cpu"):
        self.n = n
        self.device = device

    def simulate(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        theta: shape (batch, 2 * d)
               first d — mu
               next d — log_sigma
        returns:
            x: (batch, n, d)
        """
        d = theta.shape[1] // 2
        mu = theta[:, :d]
        log_sigma = theta[:, d:]
        sigma = torch.exp(log_sigma)

        batch_size = theta.shape[0]

        eps = torch.randn(batch_size, self.n, d, device=self.device)
        x = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps

        return x


# -------------------- TASK --------------------

class GaussianTask(Task):
    def __init__(
        self,
        dim: int = 1,
        n: int = 10,
        mu_range: Tuple[float, float] = (-5.0, 5.0),
        log_sigma_range: Tuple[float, float] = (-2.0, 1.0),
        device="cpu",
    ):
        self.dim = dim
        self.n = n
        self.mu_range = mu_range
        self.log_sigma_range = log_sigma_range

        super().__init__(device=device)

        self.theta_dim = 2 * dim
        self.data_dim = dim

    # -------- components --------

    def build_prior(self) -> Distribution:
        low = torch.tensor(
            [self.mu_range[0]] * self.dim +
            [self.log_sigma_range[0]] * self.dim
        )
        high = torch.tensor(
            [self.mu_range[1]] * self.dim +
            [self.log_sigma_range[1]] * self.dim
        )

        return UniformBox(low, high, device=self.device)

    def build_simulator(self) -> Simulator:
        return GaussianSimulator(n=self.n, device=self.device)

    def build_summary(self):
        """
        Summary: (mean, log variance)
        This is sufficient statistic for Gaussian (up to scaling).
        """
        def summary(x: torch.Tensor) -> torch.Tensor:
            # x: (batch, n, d)
            mean = x.mean(dim=1)
            var = x.var(dim=1, unbiased=False)

            # стабильность
            log_var = torch.log(var + 1e-8)

            return torch.cat([mean, log_var], dim=-1)

        return summary