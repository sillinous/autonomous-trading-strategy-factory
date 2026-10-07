import json

import numpy as np
import pandas as pd
import pytest

from atsf.factory import run_factory
from atsf.proposer import ClaudeProposer, admit, build_prompt


def data(n: int = 1600) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, n)))
    return pd.DataFrame({"open": close, "high": close * 1.004, "low": close * 0.996,
                         "close": close, "volume": 1e6}, index=pd.bdate_range("2016-01-01", periods=n))


GOOD = {"rationale": "Buy short-term dips inside an uptrend.", "strategy": {
    "name": "dip-buyer", "universe": ["WRONG"], "timeframe": "1h",
    "indicators": [{"name": "trend", "kind": "sma", "period": 100},
                   {"name": "dip", "kind": "rsi", "period": 3}],
    "entry": {"all": [{"left": "close", "comparator": ">", "right": "trend"},
                      {"left": "dip", "comparator": "<", "right": 20}]},
    "exit": {"any": [{"left": "dip", "comparator": ">", "right": 60}]},
    "position_sizing": {"method": "volatility_target", "value": 0.15, "max_position": 1.0},
    "risk": {"max_position": 1.0, "stop_loss": 0.07}}}


def reply(elements, prose="Here you go:\n```json\n{}\n```"):
    return {"content": [{"type": "text", "text": prose.format(json.dumps(elements))}],
            "usage": {"input_tokens": 10, "output_tokens": 20}}


def test_admits_valid_and_forces_symbol_and_timeframe():
    proposal = admit(GOOD, "SPY", data())
    assert proposal.accepted
    assert proposal.strategy.universe == ["SPY"] and proposal.strategy.timeframe == "1d"
    assert proposal.strategy.metadata["origin"] == "claude"


@pytest.mark.parametrize("bad,reason", [
    ("not a dict", "element is not"),
    ({"strategy": {"name": "x"}}, "invalid DSL"),
    ({"strategy": {**GOOD["strategy"], "indicators": [{"name": "t", "kind": "ichimoku", "period": 9}]}},
     "invalid DSL"),
    ({"strategy": {**GOOD["strategy"], "entry": {"all": [{"left": "close", "comparator": ">", "right": 1e9}]}}},
     "never enters"),
    ({"strategy": {**GOOD["strategy"], "entry": {"all": [{"left": "missing", "comparator": ">", "right": 1}]}}},
     "does not compile"),
])
def test_rejects_invalid_proposals_with_reasons(bad, reason):
    proposal = admit(bad, "SPY", data())
    assert not proposal.accepted and reason in proposal.reason


def test_proposer_round_trip_with_fake_transport():
    sent = {}

    def transport(body):
        sent.update(body)
        return reply([GOOD, GOOD, {"strategy": {"name": "broken"}}])

    batch = ClaudeProposer(transport=transport, model="test-model").propose(
        "SPY", data(), 3, history=[{"family": "macd_cross", "reasons": ["below buy-and-hold"]}])
    assert [p.accepted for p in batch.proposals] == [True, False, False]
    assert batch.proposals[1].reason == "duplicate proposal"
    prompt = sent["messages"][0]["content"]
    assert "macd_cross" in prompt and "bb_upper" in prompt and sent["model"] == "test-model"


def test_unparseable_output_is_an_error():
    proposer = ClaudeProposer(transport=lambda body: {"content": [{"type": "text", "text": "no"}]})
    with pytest.raises(RuntimeError, match="unparseable"):
        proposer.propose("SPY", data(), 2)


def test_requires_api_key_without_transport(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeProposer()


def test_prompt_states_the_evaluation_contract():
    prompt = build_prompt("SPY", data(), 4, None)
    assert "next bar" in prompt and "buy-and-hold" in prompt and "deflated" in prompt


def test_proposals_count_as_trials_in_the_factory():
    frame = data()
    accepted = admit(GOOD, "TEST", frame).strategy
    report = run_factory(frame, "TEST", strategies=[accepted], perturbation_samples=2)
    assert report.n_trials == 1 and report.rows[0].family.startswith("claude:")
