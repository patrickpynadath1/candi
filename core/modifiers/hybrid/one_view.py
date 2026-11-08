"""Expose a one-view recipe's view components under their usual names.

Single-process families (MDLM, Gaussian, simplex, ...) are one ``DiffusionView``
on the shared view path (see core/algorithms/mdlm.py).
"""


class OneViewAlgorithm:
    """Expose a one-view recipe's components under their standalone names.

    Mixed into a ``HybridDiffusion`` subclass that has exactly one view, so
    samplers, tests and modifiers can keep reading ``algo.process``,
    ``algo.schedule`` and ``algo.representation``. Assigning them replaces the
    view's component (``nn.Module`` routes module values before property
    setters run, hence ``__setattr__``).
    """

    view_components = ("process", "schedule", "representation")

    @property
    def process(self):
        return self.views[0].process

    @property
    def schedule(self):
        return self.views[0].schedule

    @property
    def representation(self):
        return self.views[0].representation

    def __setattr__(self, name, value):
        if name in self.view_components and "model" in self._modules:
            setattr(self.views[0], name, value)
        else:
            super().__setattr__(name, value)
