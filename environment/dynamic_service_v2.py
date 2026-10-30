"""Public-demand aggregation and motion feedback; inherited physics unchanged."""
import numpy as np
import config
from environment.dynamic_service import DynamicServiceEnv


def aggregate_demand(observation):
    """Share the union of current public records, without consulting the tape."""
    visible=np.asarray(observation['visible'],dtype=bool)
    ue=np.asarray(observation['ue']);nodes=np.asarray(observation['nodes'])
    union=visible.any(axis=0);out=np.zeros_like(ue)
    for u in np.flatnonzero(union):
        witness=int(np.flatnonzero(visible[:,u])[0])
        absolute=ue[witness,u,:2]+nodes[witness,:2]
        out[:,u,:2]=absolute-nodes[:,:2]
        out[:,u,2:]=ue[witness,u,2:]
    return dict(observation,ue=out,visible=np.broadcast_to(union,visible.shape).copy())


def boundary_actions(raw,positions):
    """Map a latent Normal action into the current one-step feasible box.

    The policy density is on raw, not the executed displacement. At a wall,
    zero latent mean maps into the interior instead of being clipped to hover.
    The inherited joint projector still enforces the speed ball and separation.
    """
    raw=np.asarray(raw,dtype=np.float64);positions=np.asarray(positions,dtype=np.float64)
    if raw.shape!=(5,2) or positions.shape!=(5,2) or not np.isfinite(raw).all():
        raise ValueError('five finite latent actions and positions required')
    speed=config.UAV_SPEED*config.TIME_SLOT_DURATION;gap=config.UAV_COVERAGE_RADIUS/2
    high=np.array([config.AREA_WIDTH,config.AREA_HEIGHT])-gap
    if not np.isfinite(positions).all() or np.any(positions<gap-1e-3) or np.any(positions>high+1e-3):
        raise ValueError('positions outside the physical flight region')
    lo=np.maximum(-1.,(gap-positions)/speed)
    hi=np.minimum(1.,(high-positions)/speed)
    return lo+(np.tanh(raw)+1)*.5*(hi-lo)


class DynamicServiceEnvV2(DynamicServiceEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.motion=np.zeros((5,6),np.float32)

    def raw_observation(self):
        return super().observe()

    def observe(self):
        obs=aggregate_demand(self.raw_observation())
        gap=config.UAV_COVERAGE_RADIUS/2;speed=config.UAV_SPEED*config.TIME_SLOT_DURATION
        high=np.array([config.AREA_WIDTH,config.AREA_HEIGHT])-gap
        # Four distances to walls in movement budgets, capped at one step.
        walls=np.c_[self.positions-gap,high-self.positions]/speed
        obs.update(motion=self.motion.copy(),walls=np.clip(walls,0,1).astype(np.float32))
        return obs

    def step(self,actions,assignment,decision_timeout=False):
        before=self.positions.copy();actions=np.asarray(actions,dtype=np.float64)
        bounded=np.clip(actions,-1,1);norm=np.linalg.norm(bounded,axis=1,keepdims=True)
        intended=bounded/np.maximum(norm,1e-12)*np.minimum(norm,1)
        if decision_timeout:intended*=0
        row=super().step(actions,assignment,decision_timeout=decision_timeout)
        actual=(self.positions-before)/(config.UAV_SPEED*config.TIME_SLOT_DURATION)
        residual=np.linalg.norm(actual-intended,axis=1)
        self.motion=np.c_[actual,intended,residual,residual>1e-4].astype(np.float32)
        row.update(boundary_uav_steps=int(((self.positions<51)|(self.positions>649)).any(axis=1).sum()),
                   hovering_uav_steps=int((np.linalg.norm(actual,axis=1)<.1/15).sum()),
                   projection_residual=float(residual.mean()),actual_step_m=float(np.linalg.norm(actual,axis=1).mean()*15))
        return row
