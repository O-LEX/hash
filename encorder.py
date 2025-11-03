import torch
import torch.nn as nn

def prime_hash(indices):
    primes = torch.tensor([
        1958374283, 2654435761, 805459861,
        3674653429, 2097192037, 1434869437, 2165219737
    ], device=indices.device, dtype=torch.long)
    primes = primes[:indices.shape[-1]]

    h = torch.zeros(indices.shape[0], device=indices.device, dtype=torch.long)
    for i in range(indices.shape[-1]):
        h ^= indices[:, i] * primes[i]
    return h

def baseconvert_hash(indices):
    h = torch.zeros(indices.shape[0], device=indices.device, dtype=torch.long)
    for i in range(indices.shape[-1]):
        h += indices[:, i]
        h *= 2531011
    return h

class Encoder(nn.Module):
    def __init__(self, input_dim, num_levels=8, features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, per_level_scale=2.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2 ** log2_hashmap_size
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale

        self.embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(self.hashmap_size, features_per_level) * 1e-4)
            for _ in range(num_levels)
        ])

        if input_dim <= 7:
            self.hash = prime_hash
        else:
            self.hash = baseconvert_hash

    def forward(self, x):
        # x: [..., input_dim] in [0,1]
        outputs = []

        for l in range(self.num_levels):
            res = int(self.base_resolution * (self.per_level_scale ** l))
            pos = x * res

            pos0 = torch.floor(pos).long()
            frac = pos - pos0

            offsets = torch.stack(
                [torch.tensor([(i >> d) & 1 for d in range(self.input_dim)], device=x.device)
                 for i in range(2 ** self.input_dim)]
            )

            feats = torch.zeros((*x.shape[:-1], self.features_per_level), device=x.device)
            for i, off in enumerate(offsets):
                idx = pos0 + off
                h = self.hash(idx) % self.hashmap_size
                f = self.embeddings[l][h]

                w = torch.prod(torch.where(off.bool(), frac, 1 - frac), dim=-1, keepdim=True)
                feats += w * f

            outputs.append(feats)

        return torch.cat(outputs, dim=-1)

if __name__ == "__main__":
    enc = Encoder(input_dim=4, num_levels=4)
    x = torch.rand(10, 4, requires_grad=True)
    y = enc(x)
    loss = y.sum()
    loss.backward()
    print(x.grad)
