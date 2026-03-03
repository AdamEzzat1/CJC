# Chess RL Benchmark: Design Document

## Architecture

### Three-Layer Design

1. **Chess Environment (CHESS_ENV)** -- Pure CJC, ~280 lines
   - Functional (immutable) board representation: flat array of 64 integers
   - Deterministic move generation: squares 0..63 in order, targets in order
   - Full legality checking via check detection after each candidate move
   - Feature encoding: board -> Tensor[1,64] normalized by piece type

2. **RL Agent (RL_AGENT)** -- Pure CJC, ~200 lines
   - Per-move scoring MLP: input[1,66] -> hidden[1,16] -> score[scalar]
   - Masked softmax over legal moves for action selection
   - Manual REINFORCE gradient computation with analytical backpropagation
   - No autograd dependency (manual chain rule through ReLU and matmul)

3. **Training Loop (TRAINING)** -- Pure CJC, ~180 lines
   - Self-play episode generation with trajectory recording
   - Per-step REINFORCE updates with configurable learning rate and baseline
   - Evaluation against uniform random opponent
   - All state passed explicitly (no global mutation)

### Piece Encoding

```
 0 = empty
 1 = white pawn,   -1 = black pawn
 2 = white knight,  -2 = black knight
 3 = white bishop,  -3 = black bishop
 4 = white rook,    -4 = black rook
 5 = white queen,   -5 = black queen
 6 = white king,    -6 = black king
```

### Network Architecture

```
Input:  [1, 66] = concat(board_features[64], [from_sq/63, to_sq/63])
Hidden: [1, 16] = relu(input @ W1[66,16] + b1[1,16])
Score:  scalar  = (hidden @ W2[16,1])[0,0]
```

For each legal move, the network computes a score. Softmax over all move scores gives the action probability distribution. `categorical_sample` selects the action.

### Gradient Computation

REINFORCE policy gradient for the selected action:

```
grad(log pi(a|s)) = sum_j (delta(j,a) - prob_j) * grad(score_j)
```

Where `grad(score_j)` is computed analytically:
- `d(score)/d(W2) = hidden_j^T`
- `d(score)/d(hidden) = W2^T`
- `d(hidden)/d(pre_relu) = relu_mask` (indicator of pre_relu > 0)
- `d(score)/d(b1) = W2^T * relu_mask`
- `d(score)/d(W1) = input_j^T @ (W2^T * relu_mask)`

## Design Decisions

### Why Manual Gradients (Not GradGraph)?

CJC has a GradGraph for automatic differentiation, but the chess RL benchmark uses manual analytical gradients because:
1. The REINFORCE gradient requires access to all move scores simultaneously (for the softmax correction term)
2. Manual gradients demonstrate CJC's numerical computing capabilities more thoroughly
3. The per-move scoring architecture doesn't fit cleanly into a single computational graph

### Why Functional Array Semantics?

CJC's `array_push(arr, val)` returns a new array rather than mutating in place. The chess code uses `arr = array_push(arr, val)` reassignment. This is consistent with CJC's immutable-first design and avoids aliasing bugs.

### Why Not `if` Expressions?

CJC's parser treats `if` as a statement, not an expression. The code uses `let x = default; if cond { x = other; }` instead of `let x = if cond { a } else { b };`. This was discovered during debugging and is a known parser limitation.

### Why No Semicolons After Block Statements?

Inside function bodies, CJC's parser does NOT consume a trailing `;` after `while {}`, `if {}`, or `for {}` blocks. Adding `;` after these causes parse errors. This is asymmetric with top-level code (where error recovery handles stray `;`).

## Dependencies

Only CJC builtins -- no external crates or runtime dependencies beyond:
- `abs`, `len`, `float`, `int`, `print` (pre-existing)
- `matmul`, `Tensor.from_vec`, `Tensor.randn`, `Tensor.zeros`, `.softmax()`, `.relu()`, `.get()`, `.transpose()` (pre-existing)
- `log`, `exp`, `categorical_sample` (added by this benchmark, general-purpose)

## Wiring Discipline

All new builtins follow CJC's three-layer wiring:
1. **cjc-runtime/builtins.rs**: Core implementation
2. **cjc-types/effect_registry.rs**: Effect classification (pure/nondet)
3. **cjc-eval/lib.rs** + **cjc-mir-exec/lib.rs**: Dispatch in both execution paths
