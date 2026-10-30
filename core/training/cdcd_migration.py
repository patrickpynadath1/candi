"""Explicit, offline migration; never silently reset an old sampler on resume."""

import copy
from hydra.utils import instantiate
from omegaconf import OmegaConf

OLD_TARGET = 'core.time_sampling.loss_derivative.LossDerivativeTimeSampler'
NEW_TARGET = 'core.time_sampling.cdcd.CDCDTimeSampler'


def migrate_checkpoint(checkpoint, sampler_config):
    """Preserve model/optimizer/EMA, replace only legacy sampler buffers/config.

    Return a new checkpoint mapping. Save it to a NEW path; the source remains a
    reproducible legacy checkpoint. CDF warmup restarts independently of model
    global_step. External launch configs must also select the returned config.
    """
    settings = OmegaConf.to_container(OmegaConf.create(sampler_config), resolve=True)
    if settings.get('_target_') != NEW_TARGET:
        raise ValueError('migration requires a CDCD sampler config')
    state = checkpoint['state_dict']
    prefixes = [k[:-len('times')] for k in state if k.endswith('time_sampler.times')]
    if not prefixes:
        raise ValueError('checkpoint has no legacy loss-derivative sampler state')
    result = dict(checkpoint)
    result['state_dict'] = {k: v for k, v in state.items()
                            if not any(k.startswith(p) for p in prefixes)}
    for prefix in prefixes:
        if not all(prefix+k in state for k in ['cdf', 'density', 'losses']):
            raise ValueError('unrecognised legacy sampler state')
        fresh = instantiate(settings)
        result['state_dict'].update({prefix+k:v for k,v in fresh.state_dict().items()})
    hparams = copy.deepcopy(checkpoint['hyper_parameters'])
    config = OmegaConf.to_container(OmegaConf.create(hparams['config']), resolve=True)

    def replace(node):
        if isinstance(node, dict):
            if node.get('_target_') == OLD_TARGET:
                return dict(settings)
            return {k:replace(v) for k,v in node.items()}
        if isinstance(node, list):
            return [replace(v) for v in node]
        return node

    hparams['config'] = replace(config)
    result['hyper_parameters'] = hparams
    result['cdcd_migration'] = {'source_step': checkpoint.get('global_step'),
                                'reset_sampler_prefixes': prefixes,
                                'uniform_mixing': False}
    return result
