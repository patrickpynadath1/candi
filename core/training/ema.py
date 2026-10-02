import torch


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the result of
                `model.parameters()`.
            decay: The exponential decay.
            use_num_updates: Whether to use number of updates when computing
                averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def _trainable_parameters(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        if len(parameters) != len(self.shadow_params):
            raise ValueError(
                f"EMA parameter count mismatch: expected {len(self.shadow_params)}, "
                f"got {len(parameters)}"
            )
        return parameters

    def move_shadow_params_to_device(self, device):
        self.shadow_params = [i.to(device) for i in self.shadow_params]

    def update(self, parameters):
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = self._trainable_parameters(parameters)
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages.
        """
        parameters = self._trainable_parameters(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        parameters = self._trainable_parameters(parameters)
        self.collected_params = [param.detach().clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters.
        """
        parameters = self._trainable_parameters(parameters)
        if len(self.collected_params) != len(parameters):
            raise ValueError("EMA stored parameter count mismatch")
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
        self.collected_params = []

    def state_dict(self):
        return dict(
            decay=self.decay,
            num_updates=self.num_updates,
            shadow_params=[
                parameter.detach().clone() for parameter in self.shadow_params
            ],
        )

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        shadow_params = state_dict["shadow_params"]
        if len(shadow_params) != len(self.shadow_params):
            raise ValueError(
                f"EMA parameter count mismatch: expected {len(self.shadow_params)}, "
                f"got {len(shadow_params)}"
            )
        self.shadow_params = [parameter.detach().clone() for parameter in shadow_params]
