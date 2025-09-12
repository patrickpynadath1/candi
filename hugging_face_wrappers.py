import pdb
import torch
import torch.nn as nn
import transformers
from transformers import PreTrainedModel, PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import tqdm

from lm_eval.api.model import LM

class DiffusionConfig(PretrainedConfig):
    model_type = "diffusion_lm"

    def __init__(self, algo_name: str, **kwargs):
        super().__init__(**kwargs)
        self.algo_name = algo_name




class GenHFWrapper(PreTrainedModel):

    config_class = DiffusionConfig
    def __init__(self, config: DiffusionConfig, diffusion_model: nn.Module, timesteps: int = 8):
        super().__init__(config)
        self.diffusion_model = diffusion_model
        self.timesteps = timesteps


    def forward(self, input_ids, attention_mask=None, labels=None):
        # prep input

        with torch.no_grad():

            times = torch.linspace(1.0, 0.0, steps=self.timesteps).to(input_ids.device)  # linear scheduleA
            dalpha_t, cur_alpha_t = self.diffusion_model.noise(times)
            cur_alpha_t = cur_alpha_t[:, None]
            # dalpha_t = torch.ones(input_ids.shape[0], device=input_ids.device).float() * dalpha_t

            # cur_alpha_t = torch.ones(input_ids.shape[0], device=input_ids.device).float() * cur_alpha_t

            input_ids = input_ids.repeat(self.timesteps, 1)
            noisy_output = self.diffusion_model.q_xt(input_ids, cur_alpha_t)
            sigma = self.diffusion_model._sigma_from_alphat(cur_alpha_t)

            log_x_theta = self.diffusion_model.forward(noisy_output, sigma=sigma)

            cur_loss = self.diffusion_model.nll_per_token(
                log_x_theta=log_x_theta,
                xt=noisy_output, 
                x0=input_ids,
                alpha_t=cur_alpha_t,
                dalpha_t=dalpha_t,
                low_var=True,
            ).mean()

                
        log_x_theta = log_x_theta.mean(dim=0, keepdim=True)

        log_x_theta = log_x_theta[:, :-1, :].contiguous()
        return CausalLMOutputWithCrossAttentions(
            loss=cur_loss,
            logits=log_x_theta,
            hidden_states=None,
            attentions=None,
        )

class CandiHFWrapper(PreTrainedModel):

    config_class = DiffusionConfig
    def __init__(self, config: DiffusionConfig, diffusion_model: nn.Module, timesteps: int = 8):
        super().__init__(config)
        self.diffusion_model = diffusion_model
        self.timesteps = timesteps


    def forward(self, input_ids, attention_mask=None, labels=None):
        # prep input

        with torch.no_grad():

            times = torch.linspace(1.0, 0.0, steps=self.timesteps).to(input_ids.device)  # linear scheduleA
            dalpha_t, cur_alpha_t = self.diffusion_model.noise(times)

            # dalpha_t = torch.ones(input_ids.shape[0], device=input_ids.device).float() * dalpha_t

            # cur_alpha_t = torch.ones(input_ids.shape[0], device=input_ids.device).float() * cur_alpha_t

            input_ids = input_ids.repeat(self.timesteps, 1)
            noisy_output = self.diffusion_model.q_xt(input_ids, cur_alpha_t.unsqueeze(-1))

            log_x_theta = self.diffusion_model.forward(**noisy_output)


            cur_loss = self.diffusion_model.nll_per_token(
                log_x_theta=log_x_theta,
                alpha_t=cur_alpha_t[:, None],
                dalpha_t=dalpha_t,
                x0_tokens=input_ids,
                low_var=True,
                **noisy_output,
            ).mean()

                
        log_x_theta = log_x_theta.mean(dim=0, keepdim=True)
        log_x_theta = log_x_theta[:, 1:, :].contiguous()
        return CausalLMOutputWithCrossAttentions(
            loss=cur_loss,
            logits=log_x_theta,
            hidden_states=None,
            attentions=None,
        )
    


class CandiEvalWrapper(LM):
    def __init__(self, diffusion_model, tokenizer): 
        super().__init__()
        self.diffusion_model = diffusion_model
        self.tokenizer = tokenizer
        self.device = diffusion_model.device


    def loglikelihood(self, requests):
        
        to_return = []
        for r in tqdm.tqdm(requests): 
            input_str = r.args[0]
            target_str = r.args[1]

            input_ids = self.tokenizer(input_str, return_tensors="pt").input_ids.to(self.device)
            target_ids = self.tokenizer(target_str, return_tensors="pt").input_ids.to(self.device)

            time = 1 - input_ids.shape[1] / (input_ids.shape[1] + target_ids.shape[1])
            times = torch.tensor([time], device=input_ids.device)

            # noisy_target = self.diffusion_model.sample_prior(target_ids.shape[0], target_ids.shape[1], self.diffusion_model.vocab - 1)

            # noisy_target = torch.randn(target_ids.shape[0], target_ids.shape[1], self.diffusion_model.vocab_size - 1, device=input_ids.device) * noise[:, None, None]

            noisy_target = self.diffusion_model.prior_sample(target_ids.shape[0], target_ids.shape[1])

            clean_input = torch.nn.functional.one_hot(input_ids, num_classes=self.diffusion_model.vocab_size - 1).float().to(self.device)

            model_input = torch.cat([clean_input, noisy_target], dim=1)
            reveal_mask = torch.ones(input_ids.shape[1] + target_ids.shape[1]).to(self.device)

            reveal_mask[input_ids.shape[1]:] = 0.0

            reveal_mask = reveal_mask[None, :]
            sigma = self.diffusion_model.get_continuous_from_discrete_noise(torch.tensor([.999]).to(self.device))

            model_output = self.diffusion_model(xt=model_input, discrete_noise=times, continuous_noise=sigma, reveal_mask=reveal_mask)

            # none of the tasks of interest should the greddy flag
            # regardless, unfair towards diffusion models (since they use a fundementally different genrative scheme )
            is_greedy = 0

            log_p_theta = torch.gather(
                input=model_output[:, input_ids.shape[1]:, :], dim=-1, index=target_ids[:, :, None]
            ).squeeze(-1).sum().item()

            to_return.append((log_p_theta, is_greedy))
        return to_return

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Rolling loglikelihood not implemented for diffusion models yet.")
    

    def generate_until(self, requests):
        raise NotImplementedError("Generation not implemented for diffusion models yet.")


class DiffusionEvalWrapper(LM):
    def __init__(self, diffusion_model, tokenizer): 
        super().__init__()
        self.diffusion_model = diffusion_model
        self.tokenizer = tokenizer
        self.device = diffusion_model.device


    def loglikelihood(self, requests):
        
        to_return = []
        for r in tqdm.tqdm(requests): 
            input_str = r.args[0]
            target_str = r.args[1]

            input_ids = self.tokenizer(input_str, return_tensors="pt").input_ids.to(self.device)
            target_ids = self.tokenizer(target_str, return_tensors="pt").input_ids.to(self.device)

            times = torch.tensor([1.0], device=input_ids.device)

            # noisy_target = self.diffusion_model.sample_prior(target_ids.shape[0], target_ids.shape[1], self.diffusion_model.vocab - 1)
            _, alpha_t = self.diffusion_model.noise(times)
            sigma = self.diffusion_model._sigma_from_alphat(alpha_t)

            if hasattr(self.diffusion_model, 'mask_index'): 
                random = (torch.ones_like(target_ids).to(self.device) * self.diffusion_model.mask_index).to(input_ids.dtype)

            else: 
                random = torch.randint(0, self.diffusion_model.vocab_size - 1, target_ids.shape, device=input_ids.device)

            xt = torch.cat([input_ids, random], dim=1)
            model_output = self.diffusion_model(xt=xt, sigma=sigma[:, None])

            log_p_theta = torch.gather(
                input=model_output[:, input_ids.shape[1]:, :], dim=-1, index=target_ids[:, :, None]
            ).squeeze(-1).sum().item()

            to_return.append((log_p_theta, 0))
        return to_return

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Rolling loglikelihood not implemented for diffusion models yet.")
    

    def generate_until(self, requests):
        raise NotImplementedError("Generation not implemented for diffusion models yet.")

    
        
