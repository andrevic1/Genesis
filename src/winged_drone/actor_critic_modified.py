import torch
import torch.nn.functional as F
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent
from torch.distributions import Normal
import math

class ActorCriticTanh(ActorCriticRecurrent):
    def __init__(self, *args, max_servo=0.34906585, max_throttle=1.0, **kw):
        super().__init__(*args, **kw)
        self.max_servo    = max_servo
        self.max_throttle = max_throttle
        self._LOG2        = math.log(2.)

    # ------------------------------------------------ helper
    def _scale(self, a):
        thr = 0.5 * (a[..., :1] + 1) * self.max_throttle
        srv = a[..., 1:] * self.max_servo
        return torch.cat([thr, srv], -1)

    def _inverse_scale(self, act):
        thr, srv = act[..., :1], act[..., 1:]
        a_thr = thr / self.max_throttle * 2 - 1
        a_srv = srv / self.max_servo
        return torch.cat([a_thr, a_srv], -1).clamp(-0.999999, 0.999999)

    # ------------------------------------------------ overrides
    def act(self, obs, deterministic=False, masks=None, hidden_states=None):
        inp = self.memory_a(obs, masks, hidden_states)
        self.update_distribution(inp.squeeze(0))

        z = self.distribution.mean if deterministic else self.distribution.rsample()
        a = torch.tanh(z)                         # (-1,1)
        act = self._scale(a)

        # log-prob stabile
        logp_corr = 2 * (self._LOG2 - z - F.softplus(-2*z))
        self._last_logp = (self.distribution.log_prob(z) + logp_corr).sum(-1, keepdim=True)

        return act

    def get_actions_log_prob(self, act):
        a  = self._inverse_scale(act)
        # atanh in forma stabile
        z  = 0.5 * (torch.log1p(a) - torch.log1p(-a))
        logp_corr = 2 * (self._LOG2 - z - F.softplus(-2*z))
        return (self.distribution.log_prob(z) + logp_corr).sum(-1)
