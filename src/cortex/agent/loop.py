"""Language Agent Tree Search for complex multi-branch agent tasks."""

import json
import math
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from cortex.models.provider import GenerationConfig, Message
from cortex.models.router import ModelCapability, ModelRouter

if TYPE_CHECKING:
    from cortex.agent.kernel import AgentKernel, AgentState

logger = structlog.get_logger(__name__)
_EXPLORATION_CONSTANT = 1.414


class SearchNode(BaseModel):
    """A node in the LATS search tree."""

    id: str
    state: "AgentState"
    parent_id: str | None
    children: list[str] = Field(default_factory=list)
    value: float
    visit_count: int
    depth: int


class LATSLoop:
    """Language Agent Tree Search over possible action trajectories.

    This implementation follows the high-level structure from Yao et al.
    "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in
    Language Models" (arXiv:2310.04406): select a promising state, expand it
    with LLM-proposed actions, simulate branches, evaluate states, then
    backpropagate values through the tree. It terminates when a complete state
    is found or the simulation budget is exhausted.
    """

    def __init__(
        self,
        kernel: "AgentKernel",
        router: ModelRouter,
        max_depth: int = 5,
        n_branches: int = 3,
        simulation_budget: int = 10,
    ) -> None:
        self._kernel = kernel
        self._router = router
        self._max_depth = max_depth
        self._n_branches = n_branches
        self._simulation_budget = simulation_budget

    async def run(self, task: str, session_id: str) -> "AgentState":
        """Run LATS and return the best terminal state discovered."""
        from cortex.agent.kernel import AgentState

        SearchNode.model_rebuild(_types_namespace={"AgentState": AgentState})
        root = SearchNode(
            id=uuid4().hex,
            state=AgentState(session_id=session_id, user_input=task),
            parent_id=None,
            value=0.0,
            visit_count=1,
            depth=0,
        )
        nodes: dict[str, SearchNode] = {root.id: root}
        best = root.state
        best_value = -1.0
        budget_used = 0

        while budget_used < self._simulation_budget:
            selected = self._ucb1_select(nodes)
            if selected.depth >= self._max_depth:
                selected.value = await self._evaluate_state(selected.state)
                self._backpropagate(nodes, selected.id, selected.value)
                budget_used += 1
                continue

            branches = await self._expand(task, selected.state)
            if not branches:
                break

            for branch in branches[: self._n_branches]:
                if budget_used >= self._simulation_budget:
                    break
                child_state = await self._simulate(task, branch, session_id, selected.depth + 1)
                value = await self._evaluate_state(child_state)
                child = SearchNode(
                    id=uuid4().hex,
                    state=child_state,
                    parent_id=selected.id,
                    value=value,
                    visit_count=1,
                    depth=selected.depth + 1,
                )
                nodes[child.id] = child
                selected.children.append(child.id)
                self._backpropagate(nodes, child.id, value)
                if value > best_value:
                    best = child_state
                    best_value = value
                budget_used += 1
                if child_state.status == "complete":
                    return child_state

        if best_value < 0:
            root.state.status = "failed"
            root.state.final_answer = "LATS exhausted its budget without finding a solution."
            return root.state
        return best

    async def _evaluate_state(self, state: "AgentState") -> float:
        """Score an AgentState in [0, 1] using the REASONING model."""
        prompt = (
            "Score this agent state from 0.0 to 1.0 as JSON with keys "
            "goal_completion, factual_correctness, conciseness. Consider whether "
            "the final answer solves the original task.\n\n"
            f"Task: {state.user_input}\n"
            f"Status: {state.status}\n"
            f"Final answer: {state.final_answer or ''}"
        )
        response = await self._router.complete(
            ModelCapability.REASONING,
            [
                Message(role="system", content="You are a strict agent state evaluator."),
                Message(role="user", content=prompt),
            ],
            GenerationConfig(temperature=0.0, max_tokens=128),
        )
        try:
            data = json.loads(response.content)
            scores = [
                float(data.get("goal_completion", 0.0)),
                float(data.get("factual_correctness", 0.0)),
                float(data.get("conciseness", 0.0)),
            ]
            weighted = scores[0] * 0.5 + scores[1] * 0.35 + scores[2] * 0.15
            return max(0.0, min(1.0, weighted))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("lats_evaluation_parse_failed", error=str(exc), raw=response.content[:200])
            return 1.0 if state.status == "complete" else 0.0

    def _ucb1_select(self, nodes: dict[str, SearchNode]) -> SearchNode:
        """Select the most promising leaf node using UCB1."""
        candidates = [node for node in nodes.values() if not node.children]
        if not candidates:
            candidates = list(nodes.values())
        unvisited = [node for node in candidates if node.visit_count == 0]
        if unvisited:
            return max(unvisited, key=lambda node: node.value)

        def score(node: SearchNode) -> float:
            if node.parent_id and node.parent_id in nodes:
                parent_visits = max(1, nodes[node.parent_id].visit_count)
            else:
                parent_visits = max(1, sum(n.visit_count for n in nodes.values()))
            exploitation = node.value
            exploration = _EXPLORATION_CONSTANT * math.sqrt(
                math.log(parent_visits + 1) / max(1, node.visit_count)
            )
            return exploitation + exploration

        return max(candidates, key=score)

    async def _expand(self, task: str, state: "AgentState") -> list[str]:
        """Ask the LLM for alternative next branches."""
        response = await self._router.complete(
            ModelCapability.REASONING,
            [
                Message(
                    role="system",
                    content=(
                        "Generate alternative next steps for a Language Agent Tree Search. "
                        "Return ONLY a JSON array of concise branch descriptions."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"Original task: {task}\n"
                        f"Current status: {state.status}\n"
                        f"Current answer: {state.final_answer or ''}\n"
                        f"Return at most {self._n_branches} branches."
                    ),
                ),
            ],
            GenerationConfig(temperature=0.7, max_tokens=512),
        )
        try:
            branches = json.loads(response.content)
            if isinstance(branches, list):
                return [str(branch).strip() for branch in branches if str(branch).strip()]
        except json.JSONDecodeError:
            pass
        return [
            line.lstrip("0123456789. -").strip()
            for line in response.content.splitlines()
            if line.strip()
        ][: self._n_branches]

    async def _simulate(
        self, task: str, branch: str, session_id: str, depth: int
    ) -> "AgentState":
        """Simulate one branch with the ReAct loop while disabling nested LATS."""
        branch_task = f"{task}\n\nCandidate approach: {branch}"
        return await self._kernel.run(
            branch_task,
            session_id=f"{session_id}-lats-{depth}-{uuid4().hex[:8]}",
            _allow_lats=False,
        )

    def _backpropagate(self, nodes: dict[str, SearchNode], node_id: str, value: float) -> None:
        """Propagate a child value up through ancestors."""
        current_id: str | None = node_id
        while current_id is not None and current_id in nodes:
            node = nodes[current_id]
            total = node.value * node.visit_count + value
            node.visit_count += 1
            node.value = total / node.visit_count
            current_id = node.parent_id
