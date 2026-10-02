import numpy as np
import config
from environment.env import Env
np.random.seed(42)
e = Env()
e.reset()
rng = np.random.default_rng(781)
for step in range(3000):
    before = np.array([u.pos[:2].copy() for u in e.uavs])
    actions = rng.uniform(-1, 1, (config.NUM_UAVS, 2))
    e._apply_actions_to_env(actions)
    after = np.array([u.pos[:2] for u in e.uavs])
    distances = np.linalg.norm(after[:, None] - after[None, :], axis=-1)
    distances += np.eye(config.NUM_UAVS) * 1e6
    if distances.min() < config.MIN_UAV_SEPARATION - 1e-3:
        print('FAIL', step, float(distances.min()))
        print('POSITIONS', before.tolist())
        print('ACTIONS', actions.tolist())
        break
else:
    print('PASS: 3000 movement steps')
