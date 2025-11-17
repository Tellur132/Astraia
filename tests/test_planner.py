from astraia.llm_interfaces import LLMObjective, LLMRunContext
from astraia.planner import create_planner_agent


SEARCH_SPACE = {
    "theta": {"type": "float", "low": -1.0, "high": 1.0},
    "depth": {"type": "int", "low": 1, "high": 3},
}


def make_context() -> LLMRunContext:
    return LLMRunContext(objectives=[LLMObjective(name="kl", direction="minimize")])


def test_planner_disabled_returns_none() -> None:
    planner = create_planner_agent(None, None, search_space=SEARCH_SPACE)
    assert planner is None


def test_rule_based_planner_emits_strategy() -> None:
    planner = create_planner_agent(
        {"backend": "rule", "enabled": True},
        None,
        search_space=SEARCH_SPACE,
    )
    assert planner is not None
    strategy = planner.generate_strategy(make_context())
    payload = strategy.to_payload()
    assert payload["objectives"] == ["kl"]
    assert "theta" in payload["parameter_focus"]


def test_llm_planner_falls_back_without_provider() -> None:
    planner = create_planner_agent(
        {"backend": "llm", "enabled": True, "prompt_template": "use context"},
        None,
        search_space=SEARCH_SPACE,
    )
    assert planner is not None
    strategy = planner.generate_strategy(make_context())
    assert strategy.source == "rule"
