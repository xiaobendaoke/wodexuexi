# Complexity and Deployment Analysis

## Core claim

The two-level design keeps the upper-layer UAV control action fixed at `2U = 10` continuous outputs for `U = 5`. Request-level offloading is handled by a separate three-class policy over `local`, `cooperative`, and `MBS`, so the upper-layer MARL action space does not grow with the number of service requests in a slot.

## Default configuration

- UAVs: `5`
- UEs: `100`
- Per-UAV movement action dimension: `2`
- Upper-layer joint movement action dimension: `10`
- Lower-layer request classes: `3`
- Single-UAV observation dimension: `325`
- Max UAV neighbors in observation: `4`
- Max associated UEs in observation: `30`

## Surrogate inference cost

- Checkpoint: `saved_offload_policies/offload_policy_surrogate_runtime.pt`
- Feature family: `full_features`
- Input dimension: `8`
- Parameters: `4931`
- Approximate linear MACs per request: `4800`

## Action-space comparison

| Service requests in one slot | Two-level output count | Three-way request assignments | Explicit UAV/MBS assignments |
| ---: | ---: | ---: | ---: |
| 1 | 13 | 3 | 6 |
| 5 | 25 | 243 | 7776 |
| 10 | 40 | 59049 | 60466176 |
| 20 | 70 | 3486784401 | 3656158440062976 |

## Paper wording

A safe statement is: the method is not a globally optimal monolithic joint optimizer. Its contribution is a deployable decomposition that fixes the upper-layer continuous action interface and moves request-dependent discrete choices into a lightweight online classifier plus safety reranking.
