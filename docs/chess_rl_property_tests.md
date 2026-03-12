# Chess RL Property Tests

## Overview

Property-based tests using proptest to verify invariants across many random seeds.

## Test Catalog

### Board Properties (`property/test_board_props.rs`, 4 tests)
| Property | Cases | Description |
|----------|-------|-------------|
| board_always_64_squares | 50 | Board length is always 64 regardless of seed |
| apply_move_preserves_length | 50 | Board stays 64 squares after any legal move |
| legal_moves_even_length | 50 | Move list always has even length (from/to pairs) |
| move_squares_in_range | 50 | All squares in move list are in [0, 63] |

### Move Generation Properties (`property/test_movegen_props.rs`, 3 tests)
| Property | Cases | Description |
|----------|-------|-------------|
| initial_20_moves_white | 30 | White always has 20 legal moves from initial position |
| opponent_has_moves_after_one_move | 30 | After one legal move, opponent has >0 moves |
| no_self_check_3_moves | 30 | Legal moves never leave own king in check (3-ply deep) |

### Agent Properties (`property/test_agent_props.rs`, 3 tests)
| Property | Cases | Description |
|----------|-------|-------------|
| action_idx_in_range | 20 | Action index always in [0, num_moves) |
| log_prob_non_positive | 20 | Log probability always <= 0 |
| forward_score_finite | 20 | Forward pass score never NaN/Inf |

### Determinism Properties (`property/test_determinism_props.rs`, 3 tests)
| Property | Cases | Description |
|----------|-------|-------------|
| rollout_deterministic | 30 | Same seed = identical rollout output |
| legal_moves_deterministic | 30 | Same seed = identical legal move list |
| training_deterministic | 30 | Same seed = identical training result |

## Running
```bash
cargo test --test test_chess_rl_hardening property
```
