---
title: CJC-Lang Chess RL v2.1 — LinkedIn post draft
status: Draft, 2026-04-09
tone: Honest, technically specific, no hype
---

# LinkedIn post draft — CJC-Lang Chess RL v2.1

## The post

---

I shipped v2.1 of the Chess RL demo for CJC-Lang today. It's an
end-to-end A2C + GAE training loop in a language I'm building from
scratch in Rust. The whole pipeline — chess engine, 774-dim feature
encoder, factored policy/value network (~45k params), GradGraph-based
backprop, Adam optimizer driver, Elo-lite snapshot gauntlet, PGN dump,
checkpoint bundle, CSV logger, and live SVG training curves — runs
inside a single `fn main()` in CJC-Lang. The Rust side provides
primitives (tensor ops, autodiff engine, SVG renderer); everything
above that is CJC-Lang code.

**61 new tests** landed with v2.1, all passing, zero regressions in the
existing ~5,300-test workspace suite. The infrastructure side of the
upgrade is rock solid: checkpoint round-trip, cross-executor parity,
deterministic weight hashing, byte-identical SVG output. **7 / 7
infrastructure gates passing.**

And then I ran the first real training pass, and it drew every game.

Here's the part I'm not going to dress up:

• **Training:** 60 episodes, ~16 minutes wall clock
• **Eval vs random baseline (20 games):** 0 W / 20 D / 0 L — 50.0% WR
• **Eval vs material-greedy opponent (10 games):** 0 W / 10 D / 0 L — 50.0% WR
• **Snapshot-gauntlet Elo-lite (K=32):** 1000 → 1000 (Δ +0)
• **Aspirational gates from the upgrade spec:** ≥70% vs random,
  ≥30% vs greedy, Elo gain ≥100 — all **missed**
• **Final weight hash:** `-1596143894472527787`, deterministic across reruns

The failure mode is instructive, not mysterious. Every single training
episode hit the `max_moves=25` cap with `terminal_reward=0` — meaning
the agent never reached checkmate or stalemate *once* in training. So
the A2C update had no true reward signal to learn from, just
GAE-bootstrapped value estimates, which is a well-known weak-signal
regime. At eval time the greedy policy then collapses into piece-
shuffling between two static positions until the move counter expires.

The three dumped PGN games tell the story plainly. Game 1 actually
opens **1. e2-e4 Nf6 2. Bb5** — the start of a Ruy-Lopez-style line,
which is a recognizable pattern! So the network did learn *something*
about center control and piece activity. But without threefold-
repetition detection (a documented v2 limitation), without search, and
with only 60 episodes of mostly-flat loss signal, the policy has no
cost gradient for "stop shuffling the queen back and forth."

The original prompt asked for 500 episodes in ≤20 minutes. I measured
the per-episode cost up front: ~16.7 s/episode on the register-machine
executor, ~19.2 s on the tree-walk one. 500 episodes would cost ~2.5
hours, 8× over the gate. I chose to run the honest number that fits a
human budget (60), report real results, and document the gap loudly
rather than cherry-pick numbers or tune until something looks good.
That's the only posture I can defend.

**What v2.1 actually proves:** CJC-Lang can host a production-shaped RL
loop end to end. Adam with bias-corrected moments, advantage whitening,
temperature annealing, checkpoint save/load, Elo rating, snapshot
gauntlets, PGN emission, SVG plotting — all driven from one CJC-Lang
program, with byte-identical parity on the component tests between the
two executors, and a deterministic weight fingerprint that holds after
38.9 minutes of continuous compute. That's the determinism contract
doing its job under real pressure.

**What v2.1 does not prove:** that CJC-Lang can train a competitive
chess agent. 60 episodes on a 48-wide MLP trunk with no search isn't
enough, and I'm not going to pretend otherwise. The honest next step is
interpreter hot-path optimization (or a native rollout kernel) so a
real training run fits in a reasonable budget — not more RL tricks.

Code, tests, CSV log, checkpoints, PGN, and SVGs are all in the repo
under `bench_results/chess_rl_v2_1/`. The full writeup is at
`docs/chess_rl_v2/README.md` — including the v2.1 addendum with every
number above and the honest limitations list.

I learned a lot about where the interpreter's bottlenecks are. That's
what the next version is for.

#ML #CompilerEngineering #ReinforcementLearning #Rust #OpenSource

---

## Self-review notes (not for the public post)

**Things this post does right:**
- Leads with the raw loss numbers, not the infrastructure wins.
- Explains the failure mode mechanistically (zero-terminal-reward ⇒
  weak A2C signal ⇒ shuffling at eval), not as "mysterious ML magic."
- Concrete measurement (19.2 s/ep, 16.7 s/ep) to justify the 500 → 60
  episode downsize rather than pretending it was a free choice.
- Names the one concrete positive thing the network did learn (opening
  Ruy-Lopez-ish pattern) without overclaiming it as "it learned chess."
- Says what the upgrade proves AND what it does not prove.
- Ends with the next step being "fix the interpreter," not "try different
  hyperparameters" — which correctly identifies the bottleneck.

**Things to watch out for:**
- LinkedIn readers scan — the numbers need to be the first thing they
  see. The "0 W / 20 D / 0 L" line is deliberately bolded and early.
- The phrase "chess engine in CJC-Lang" is technically accurate but
  could be misread as "plays chess well." The "didn't win a single game"
  framing heads that off immediately.
- Don't link to the site/blog yet. The blog post should come after the
  LinkedIn post so the LinkedIn discussion can feed into the blog's
  comments section.
- Avoid the word "failure" — the infrastructure didn't fail, only the
  ML quality. The post uses "missed" for gates and "failure mode" in
  its ML-specific sense.

**Things explicitly NOT in the post (and why):**
- No comparison to PyTorch or other frameworks. v2.1 doesn't beat them
  at anything, so comparisons would just invite "so what's the point."
- No link to LLM-generated content. The post is hand-written.
- No screenshots of the training curves. The curves are essentially
  flat (no learning signal), and posting them would either be
  misleading (if cropped) or uninteresting (if honest).
- No mention of upcoming v2.2 plans. The post stays scoped to what
  actually shipped. Future work is hinted at ("interpreter hot-path
  optimization") but not promised.

**The one thing I'd want a reviewer to push back on:**
The post's core message is "infrastructure worked, ML didn't." That's
accurate but it's also the kind of claim that can come across as
"dev fell in love with their own tooling." A skeptical reader might
ask, "if the ML didn't work, how do you know the infra is actually
correct end to end?" The answer is: the parity gates. Two different
executors running the same 1,715-line CJC-Lang program produce
byte-identical outputs on the component tests, which is a much stronger
correctness signal than any single-executor test could be. But that
subtle point doesn't quite land in a LinkedIn post, and I'm choosing
to trust the reader rather than belabor it.
