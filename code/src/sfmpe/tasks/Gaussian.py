import torch
from sfmpe.core.distributions import Distribution
from sfmpe.core.simulator import Simulator
from sfmpe.tasks.base_task import Task
from sfmpe.utils.logger import setup_logging, Logger
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

    def sample(self, size, **kwargs) -> torch.Tensor:
        u = torch.rand(*size, self.low.shape[0], device=self.device)
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
        theta: shape (batch, 2)
               first    — mu
               second   — log_sigma
        returns:
            x: (batch, n)
        """
        mu = theta[..., 0]
        log_sigma = theta[..., 1]
        sigma = torch.exp(log_sigma)

        batch_size = theta.shape[:-1]

        eps = torch.randn(*batch_size, self.n, device=self.device)
        # print(eps.shape, mu.shape, sigma.shape)
        x = mu.unsqueeze(-1) + sigma.unsqueeze(-1) * eps
        # print(x.shape)
        return x


# -------------------- TASK --------------------

class GaussianTask(Task):
    def __init__(
        self,
        config: None,
        dim: int = 10,
        mu_range: Tuple[float, float] = (-5.0, 5.0),
        log_sigma_range: Tuple[float, float] = (-2.0, 1.0),
        device="cpu",
    ):
        if config is not None:
            
            self.prior_parameters = config["prior"]
            self.simulator_parameters = config["simulator"]
            self.summary_parameters = config["summary"]
            self.logger_config = config.get("logger")

            super().__init__()

            self.theta_dim = 2
            self.data_dim = self.dim
        else:
            self.dim = dim
            self.mu_range = mu_range
            self.log_sigma_range = log_sigma_range

            super().__init__(device=device)

            self.theta_dim = 2 * dim
            self.data_dim = dim

    # -------- components --------

    def build_prior(self) -> Distribution:
        self.mu_range = self.prior_parameters.get("mu_range", (-5.0, 5.0))
        self.log_sigma_range = self.prior_parameters.get("log_sigma_range", (-2.0, 1.0))

        low = torch.tensor([self.mu_range[0], self.log_sigma_range[0]])
        high = torch.tensor([self.mu_range[1], self.log_sigma_range[1]])

        return GaussianPrior(low, high, device=self.device)

    def build_simulator(self) -> Simulator:
        self.dim = self.simulator_parameters.get("n", 100)
        return GaussianSimulator(n=self.dim, device=self.device)

    def build_summary(self):
        """
        Summary: (mean, log variance)
        This is sufficient statistic for Gaussian (up to scaling).
        """
        if self.summary_parameters == "default":
            # def summary(x: torch.Tensor) -> torch.Tensor:
            #     # x: (batch, n, d)
            #     mean = x.mean(dim=1)
            #     var = x.var(dim=1, unbiased=False)

            #     # стабильность
            #     log_var = torch.log(var + 1e-8)

            #     return torch.cat([mean, log_var], dim=-1)

            return lambda x: x
        else:
            raise NotImplementedError(f"Summary {self.summary_parameters} is not implemented")
        
    def build_logger(self):
        if self.logger_config is None:
            return setup_logging(name="Gaussian", 
                                 level=Logger.INFO, 
                                 log_to_file=False)
        return setup_logging(name=self.logger_config["name"],
                             level=getattr(Logger, self.logger_config["level"]),
                             log_to_file=True,
                             log_file_path=self.logger_config["log_file_path"])