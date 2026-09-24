import torch
import torch.nn as nn


class SIGReg(nn.Module):
    def __init__(self, knots=17, num_slices=256):
        super().__init__()

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)

        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

        self.num_slices = num_slices

    def forward(self, proj):
        """
        proj shape can be:
            [batch, dim]
        or
            [views, batch, dim]
        """

        A = torch.randn(
            proj.size(-1),
            self.num_slices,
            device=proj.device,
            dtype=proj.dtype,
        )

        A = A / A.norm(p=2, dim=0, keepdim=True)

        x_t = (proj @ A).unsqueeze(-1) * self.t

        err = (
            (x_t.cos().mean(dim=-3) - self.phi).square()
            + x_t.sin().mean(dim=-3).square()
        )

        statistic = (err @ self.weights) * proj.size(-2)

        return statistic.mean()
