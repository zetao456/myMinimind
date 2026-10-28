"""Deterministic unequal rewards for pipeline tests, never for real training."""
def reward(prompts, completions, records):
    assert len(prompts) == len(completions) == len(records)
    return [float(i % 2 == 0) for i in range(len(completions))]
