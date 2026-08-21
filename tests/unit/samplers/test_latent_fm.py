from types import SimpleNamespace

import torch

from samplers.base import SamplingRequest
from samplers.latent_fm import LatentFlowMatchingSampler
from samplers.runner import run_sampler

from .helpers import ModelBase


class LatentModel(ModelBase):
    sampling_family = "latent_fm"
    sampling_capabilities = frozenset({"latent_prior", "latent_flow_map"})
    config = SimpleNamespace(
        sampling=SimpleNamespace(
            steps=2,
            get=lambda key, default=None: getattr(
                LatentModel.config.sampling, key, default
            ),
        )
    )

    def prior_sample(self, batch, length):
        return torch.zeros(batch, length, 4)

    def sample_latent_flow_map(self, latent, *_args, **_kwargs):
        logits = torch.zeros(*latent.shape[:2], self.vocab_size)
        logits[..., 2] = 1
        return latent, logits


def test_sampler_steps_and_preserves_prompt():
    request = SamplingRequest(
        num_samples=1,
        num_steps=2,
        prompt_tokens=torch.tensor([[1, 0, 0]]),
        prompt_mask=torch.tensor([[1, 0, 0]]),
    )
    result = run_sampler(LatentFlowMatchingSampler(), LatentModel(), request)

    assert result.samples.tolist() == [[1, 2, 2]]
    assert result.metadata == {"nfe": 4, "decoder_nfe": 2, "encoder_nfe": 2}

