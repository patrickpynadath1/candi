"""Final token selection, separate from numerical integration."""


class PosteriorDecoding:
    def __call__(self, algo, state, time):
        return algo.sampling_prediction(state, time, algo.schedule(time)).argmax(-1)


class RepresentationDecoding:
    def __call__(self, algo, state, time):
        clean_mean = algo.posterior_mean(state, time, algo.schedule(time))
        return algo.representation.decode(clean_mean)
