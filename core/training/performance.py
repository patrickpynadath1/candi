"""Opt-in execution optimizations which preserve model/checkpoint parameter names."""

from core.models.components import DiffusionBlocks
from core.models.dit import DiT
from core.modifiers.hybrid.model import HybridModel
from core.objectives.uniform import UniformDiffusionObjective
import torch


def restore_optimizer_execution(optimizer, *, fused):
    """Keep explicit execution options when resuming older AdamW state."""
    for group in optimizer.param_groups:
        group["fused"] = fused
        if fused:
            group["foreach"] = None
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                if torch.is_tensor(state.get("step")):
                    state["step"] = state["step"].to(parameter.device)


def configure_performance(module):
    settings = module.config.training.get("performance", {})
    if settings.get("hidden_only_bootstrap", False):
        for model in module.algorithm.modules():
            if isinstance(model, (DiT, HybridModel)):
                model.hidden_only_bootstrap = True
    checkpoint = settings.get("activation_checkpointing", False)
    if checkpoint:
        blocks = [
            m for m in module.algorithm.modules() if isinstance(m, DiffusionBlocks)
        ]
        if not blocks:
            raise ValueError("activation checkpointing requires DiffusionBlocks")
        for block in blocks:
            block.activation_checkpointing = True
    compilation = settings.get("compile", {})
    if compilation.get("enabled", False):
        if "optimize_ddp" in compilation:
            from torch._dynamo import config as dynamo_config

            # PyTorch 2.5's DDP graph partitioner cannot split custom autograd
            # operations such as the chunked simplex projection. Disable only
            # that partitioner when requested; Inductor and DDP stay enabled.
            optimize_ddp = compilation["optimize_ddp"]
            if not isinstance(optimize_ddp, bool):
                raise ValueError("compile.optimize_ddp must be a boolean")
            dynamo_config.optimize_ddp = optimize_ddp
        # Compile the backbone in place: replacing it with OptimizedModule would
        # introduce _orig_mod in checkpoints and optimizer parameter patterns.
        models = [
            m for m in module.algorithm.modules() if hasattr(m, "block_execution")
        ]
        if not models:
            raise ValueError("backbone compilation requires a transformer backbone")
        hybrid_backbones = {
            id(m.backbone)
            for m in module.algorithm.modules()
            if isinstance(m, HybridModel)
        }
        for model in models:
            options = dict(
                backend=compilation.get("backend", "inductor"),
                mode=compilation.get("mode", "default"),
                dynamic=compilation.get("dynamic", True),
                fullgraph=compilation.get("fullgraph", False),
            )
            if id(model) in hybrid_backbones:
                # Hybrid/insertion execution calls this method directly and
                # bypasses backbone.__call__, including during generation.
                model.forward_hidden = torch.compile(model.forward_hidden, **options)
            else:
                model.compile(**options)
        if compilation.get("objective", False):
            for objective in module.algorithm.modules():
                if isinstance(objective, UniformDiffusionObjective):
                    objective.compile(
                        backend=compilation.get("backend", "inductor"),
                        dynamic=True,
                        fullgraph=True,
                    )
