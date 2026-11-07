"""Call a view-path algorithm from a sampler written for one state.

Family samplers (Euler, DDPM, simplex bridge, spherical, blurring, analytic)
keep their own loops; on a one-view recipe they call the algorithm through
``predict`` instead of ``algo.sampling_prediction(state, time, ...)``.
"""


def is_view_algorithm(algo):
    return hasattr(algo, "views") and hasattr(algo, "points")


def points(algo, time):
    """One local point per view at global ``time`` (generation, not training)."""
    return algo.points(time)


def predict(algo, state, time, endpoint="sampling", **options):
    """``algo.<endpoint>_prediction`` on a one-view recipe at global ``time``."""
    call = getattr(algo, f"{endpoint}_prediction")
    return call((state,), points(algo, time), time, **options)
