import torch
import torchsde


class TripleWell(object):
    noise_type = "diagonal"
    sde_type = "ito"

    state_space_size = 1
    noise_var = 0

    def drift(self, x):
        return -6 * (x**5) + 20 * (x ** 3) - 8 * x

    def f(self, t, y):
        x = torch.split(y, split_size_or_sections=(1), dim=1)
        f = self.drift(x[0])
        return torch.cat([f], dim=1)

    def g(self, t, y):
        x = torch.split(y, split_size_or_sections=(1), dim=1)
        f = self.noise_var * torch.ones_like(x[0])
        return torch.cat([f], dim=1)

    @torch.no_grad()
    def sample(self, batch_size, ts, normalize, device, bm=None):
        x0 = ((torch.randn(batch_size, 1) * 1.0) - 0.5).to(device)
        """Sample data for training. Store data normalization constants if necessary."""
        xs = torchsde.sdeint(self, x0, ts, bm=bm)
        # Hack: don't check normalize, we always normalize - sorry
        mean, std = torch.mean(xs[0, :, :], dim=(0, 1)), torch.std(
            xs[0, :, :], dim=(0, 1)
        )
        xs.sub_(mean).div_(std)
        return xs, mean, std
