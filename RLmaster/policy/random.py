
from typing import Any, Dict, Optional, Union

import numpy as np

from tianshou.data import Batch
from tianshou.policy import BasePolicy


class RandomPolicy(BasePolicy):
    """A random agent used in multi-agent learning.

    It randomly chooses an action from the legal action.
    """

    def __init__(
        self,
        action_shape,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.action_shape = action_shape

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute the random action over the given batch data.

        """
        act = np.random.randint(0, self.action_shape, size=batch.obs.shape[0])
        return Batch(act=act)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Since a random agent learns nothing, it returns an empty dict."""
        return {}
