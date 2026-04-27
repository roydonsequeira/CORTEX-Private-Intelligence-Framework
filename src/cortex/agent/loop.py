"""LATS loop stub — Language Agent Tree Search.

Full implementation in Phase 6.
Reference: Yao et al. "Language Agent Tree Search Unifies Reasoning, Acting, and
Planning in Language Models" (arXiv:2310.04406).
"""


class LATSLoop:
    """Language Agent Tree Search — beam search over the action space.

    Combines Monte Carlo Tree Search with LLM-based node expansion. At each node:
    expand → simulate → backpropagate. Terminates when a solution node is found
    or the simulation budget is exhausted.

    Used for complex multi-branch tasks where the flat ReAct loop fails. The
    AgentKernel falls back to LATSLoop automatically after two consecutive
    non-progress steps (as detected by the Reflector).

    Implemented in Phase 6.
    """
