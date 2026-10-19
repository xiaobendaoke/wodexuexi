"""The frozen graph policy plus an observed action-effect residual scorer."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from marl_models.sf_hlm import mlp
from marl_models.sf_hlm_v2 import CompleteQuotaPPO

CONTRACT = 'complete_graph_observed_action_effects_8_v1'


class ActionEffectPPO(CompleteQuotaPPO):
    def __init__(self, num_uavs=5, hidden=128, learning_rate=3e-4):
        super().__init__(num_uavs, hidden, learning_rate)
        self.network.effect_head = mlp(8, 64, 1)
        # Identical initial base logits; feature contribution starts at zero.
        last = [m for m in self.network.effect_head.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def forward_states(self, states):
        logits, value = super().forward_states(states)
        effects = np.stack([s['effects'] for s in states]).astype(np.float32)
        if effects.shape != (*logits.shape, 8) or not np.isfinite(effects).all():
            raise ValueError('Action-effect input contract differs')
        logits = logits+self.network.effect_head(torch.from_numpy(effects)).squeeze(-1)
        mask = torch.from_numpy(np.stack([s['mask'] for s in states]))
        return logits.masked_fill(~mask, -1e9), value

    def save(self, folder, contract):
        path = Path(folder); path.mkdir(parents=True, exist_ok=True)
        torch.save(dict(network=self.network.state_dict(), optimizer=self.optimizer.state_dict(),
                        contract=contract, input=CONTRACT, num_uavs=self.num_uavs, hidden=self.hidden), path/'quota_effect.pt')

    def load(self, folder, contract):
        d = torch.load(Path(folder)/'quota_effect.pt', weights_only=False, map_location='cpu')
        if (d['contract'] != contract or d['input'] != CONTRACT or
                d['num_uavs'] != self.num_uavs or d['hidden'] != self.hidden):
            raise ValueError('Incompatible action-effect checkpoint')
        self.network.load_state_dict(d['network']); self.optimizer.load_state_dict(d['optimizer'])
