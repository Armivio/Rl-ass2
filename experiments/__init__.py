"""Assignment-2 experiment harness (additive package).

This package orchestrates, evaluates, visualises and builds grids for the
DQN / Dueling-DQN comparison. It only *calls* the environment
(``world.continuous_env.ContinuousEnvironment``) and the agents
(``agents.dqn_agent.DQNAgent``, ``agents.DuelingDQN_agentv2.DuelingDQNAgent``);
it never modifies their internals.

Modules
-------
grid_analysis        BFS reachability, Manhattan stats, greedy-trap check, PNG render.
make_deceptive_grid  Build ``grid_configs/deceptive_grid.npy`` (local-optima layout).
adapters             Unify the two different agent APIs behind one interface.
eval_protocol        Deterministic per-grid eval start set + greedy evaluation.
run_comparison       Multi-seed runner: train both agents, write two tidy CSVs.
metrics              Success rate, AUC, episodes-to-threshold, composite + 95% CI.
plots                Learning curves (+CI bands), bar charts, AUC, steps distributions.
"""
