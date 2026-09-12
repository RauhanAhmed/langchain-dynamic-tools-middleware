"""AST-style scoring for BFCL, following the official possible-answer semantics.

Adapted from the Berkeley Function Calling Leaderboard evaluation
(ShishirPatil/gorilla, Apache 2.0). Each expected call is
{function_name: {param: [acceptable values]}}. A predicted call matches when
its name matches and every parameter equals one of the acceptable values, or
is omitted when the empty string is an acceptable value. Nested structures
use the same rule recursively. Extra parameters the answer does not allow
fail the match, exactly as the official checker does.
"""

from __future__ import annotations

from typing import Any

from benchmarks.bfcl.data import sanitize_name


def _is_empty_option(option: Any) -> bool:
    return isinstance(option, str) and option == ""


def _values_equal(predicted: Any, expected: Any) -> bool:
    """Compare one predicted value against one acceptable value."""
    if _is_empty_option(expected):
        return predicted is None or predicted == ""
    if isinstance(predicted, bool) or isinstance(expected, bool):
        return predicted is expected
    if isinstance(predicted, (int, float)) and isinstance(expected, (int, float)):
        return float(predicted) == float(expected)
    if isinstance(predicted, list) and isinstance(expected, list):
        return len(predicted) == len(expected) and all(
            _values_equal(p, e) for p, e in zip(predicted, expected, strict=True)
        )
    if isinstance(predicted, dict) and isinstance(expected, dict):
        # A nested dict in the answer is itself a param spec: {param: [options]}.
        if all(isinstance(value, list) for value in expected.values()):
            return _params_match(predicted, expected)
        return predicted == expected
    if isinstance(predicted, dict) or isinstance(expected, dict):
        return False
    return predicted == expected


def _value_matches_options(predicted: Any, options: list[Any]) -> bool:
    return any(_values_equal(predicted, option) for option in options)


def _params_match(predicted: dict[str, Any], expected_spec: dict[str, Any]) -> bool:
    """Compare predicted call arguments against a BFCL parameter spec."""
    for key, options in expected_spec.items():
        if key not in predicted:
            if any(_is_empty_option(option) for option in options):
                continue
            return False
        if not _value_matches_options(predicted[key], options):
            return False
    return all(key in expected_spec for key in predicted)


def call_matches(predicted: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Compare one predicted call {name, args} against one expected call."""
    if len(expected) != 1:
        return False
    name, spec = next(iter(expected.items()))
    if sanitize_name(str(predicted.get("name", ""))) != sanitize_name(name):
        return False
    return _params_match(predicted.get("args", {}), spec)


def score_calls(predicted: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> bool:
    """True when the predicted call set matches the expected call set.

    Both sides are unordered: each expected call must be matched by exactly
    one predicted call and vice versa.
    """
    if len(predicted) != len(ground_truth):
        return False
    used: set[int] = set()
    for call in predicted:
        for index, expected in enumerate(ground_truth):
            if index not in used and call_matches(call, expected):
                used.add(index)
                break
        else:
            return False
    return True


def score_entry(
    category: str,
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]] | None,
) -> bool:
    """Score one BFCL entry for a given category.

    For irrelevance the model must produce no call (ground truth is None).
    For relevance any call counts (also None). Everywhere else the predicted
    calls must match the expected set exactly.
    """
    if category == "irrelevance":
        return len(predicted) == 0
    if category == "relevance":
        return len(predicted) > 0
    if ground_truth is None:
        msg = f"Category {category!r} requires ground truth but none was loaded"
        raise ValueError(msg)
    return score_calls(predicted, ground_truth)
