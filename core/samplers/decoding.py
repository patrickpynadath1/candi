"""Final token selection, separate from numerical integration.

``prediction`` and ``posterior_mean`` call a Gaussian-family algorithm at
``(state, time, path)``: a view-path recipe through its one view's point,
anything else through the single-state form ``(state, time, path)``.
"""

from core.modifiers.hybrid.views import ViewPoint
from core.samplers.view_calls import is_view_algorithm


def prediction(algo, state, time, path, endpoint="sampling"):
    call = getattr(algo, f"{endpoint}_prediction")
    if is_view_algorithm(algo):
        return call((state,), (ViewPoint(time, path),), time)
    return call(state, time, path)


def posterior_mean(algo, state, time, path):
    if is_view_algorithm(algo):
        probabilities = prediction(algo, state, time, path).float().exp()
        return algo.representation.posterior_mean(probabilities)
    return algo.posterior_mean(state, time, path)


class PosteriorDecoding:
    def __call__(self, algo, state, time):
        return prediction(algo, state, time, algo.schedule(time)).argmax(-1)


class RepresentationDecoding:
    def __call__(self, algo, state, time):
        clean_mean = posterior_mean(algo, state, time, algo.schedule(time))
        return algo.representation.decode(clean_mean)
