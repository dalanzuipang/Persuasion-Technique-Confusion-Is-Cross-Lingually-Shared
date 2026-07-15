# Prompts

## `prompt_a_23_agents.md`

Prompt-A is a GPT-4o-mini (`gpt-4o-mini-2024-07-18`) few-shot multi-agent
system with 23 independent expert agents. Each agent receives:

- the JRC definition of its technique
- curated example sentences extracted from the training data

Examples are **fixed at design time**; the system is *not* retrieval-augmented.

## `iter_ens_config.md`

Iter-Ens extends Prompt-A: three rounds at temperature τ = 0.4, aggregated by
vote threshold 0.34. This implements a **high-recall union aggregation** (a
technique is kept if it fires in ≥ 1 of 3 rounds), not strict majority voting.
