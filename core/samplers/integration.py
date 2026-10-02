"""Interchangeable integration updates; the process supplies the vector field."""


class EulerUpdate:
    def __call__(self, field, state, time, next_time):
        return state + (next_time - time) * field(time, state)


class HeunUpdate:
    def __call__(self, field, state, time, next_time):
        delta = next_time - time
        first = field(time, state)
        second = field(next_time, state + delta * first)
        return state + 0.5 * delta * (first + second)
