# Quality-aware Constrained Lower-layer Attention-MAPPO

This document records the thesis-facing method after replacing the lower-level
request offloading classifier with a constrained lower-layer MAPPO policy.

## Method Story

The system uses a two-level decision structure for multi-UAV MEC:

- Upper layer: `attention_mappo` controls UAV trajectory and coverage.
- Lower layer: `constrained_attention_offload_mappo` decides where each service request is executed.

For every covered service request, the lower layer chooses one of three actions:

- `0`: local UAV execution
- `1`: cooperative UAV execution
- `2`: MBS offloading

The lower layer is no longer a supervised request classifier. It is a request-level
multi-agent reinforcement learning policy trained with MAPPO. Each UAV is a lower-layer
agent and makes request-level offloading decisions for the service requests currently
covered by that UAV.

## Quality-aware Cooperative Mask

The cooperative action is not always available. Before sampling the lower-layer action,
the environment builds a hard action mask. The mask disables `cooperative` when the
neighbor UAV is not a good execution target.

The cooperative action is masked when:

- no cooperative UAV is available;
- estimated cooperative latency is too high relative to the request deadline;
- estimated cooperative latency is too high relative to the best non-cooperative path;
- the cooperative UAV has too little compute share relative to the local UAV.

This is the code-level version of the paper idea: when the nearby UAV is in poor
condition, the lower-layer policy is not allowed to offload the request to it.

## Lagrangian Constraints

The lower layer also uses long-term system constraints. The default training objective
penalizes violations of:

- deadline satisfaction rate target: `OFFLOAD_DSR_TARGET = 0.234`
- MBS load ceiling: `OFFLOAD_MBS_LOAD_CEILING = 0.0589`

The default values correspond to a balanced setting: preserve most of the heuristic
deadline satisfaction while reducing MBS dependence.

During training, the script maintains two multipliers:

- `lambda_dsr`
- `lambda_mbs`

If DSR falls below the target, `lambda_dsr` increases. If MBS load exceeds the ceiling,
`lambda_mbs` increases. These multipliers dynamically strengthen the penalty terms in
the lower-layer reward.

## Main Entry

Smoke run:

```bash
python run_hierarchical_mappo_experiment.py --num_episodes 1 --timestamp smoke_constrained_attention_mappo
```

Thesis run:

```bash
python run_hierarchical_mappo_experiment.py --num_episodes 50 --timestamp constrained_attention_mappo_50ep
```

Optional overrides:

```bash
python run_hierarchical_mappo_experiment.py \
  --num_episodes 50 \
  --timestamp constrained_attention_mappo_balanced \
  --offload_model constrained_attention_offload_mappo \
  --constraint_mode lagrange \
  --dsr_target 0.234 \
  --mbs_load_ceiling 0.0589
```

## Recommended Comparisons

Use these policies for the final thesis comparison:

- `uncoordinated + heuristic`
- `attention + heuristic`
- `attention + oracle-guided classifier`
- `attention + constrained attention-MAPPO`

The oracle-guided classifier is retained as a baseline or teacher reference. It is no
longer the main lower-layer method.

## Logged Diagnostics

The hierarchical training log includes:

- `lambda_dsr`
- `lambda_mbs`
- `dsr_violation`
- `mbs_load_violation`
- `constraint_penalty`
- `coop_masked_count`
- `mbs_masked_count`

These fields should be reported together with reward, latency, energy, DSR, fairness,
offloading ratios, and MBS load ratio.
