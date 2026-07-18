from tasks.gsm8k import (
    code_answer,
    extract_code,
    gold_answer,
    numeric_answer,
    score_response,
)


def test_numeric_gsm8k_scoring():
    answer = "reasoning\n#### 18"
    assert gold_answer(answer) == 18
    assert numeric_answer("The final answer is $18.") == 18
    assert score_response("work... #### 18", answer)
    assert not score_response("answer: 17", answer)


def test_code_scoring_runs_in_subprocess():
    response = "```python\ndef simple_math_problem():\n    return 6 * 7\n```"
    assert extract_code(response).startswith("def simple_math_problem")
    assert code_answer(response, timeout=2) == 42
    assert score_response(response, "#### 42", method="code", timeout=2)
    with_math = "def simple_math_problem():\n    import math\n    return math.sqrt(81)"
    assert code_answer(with_math, timeout=2) == 9


def test_code_scoring_rejects_imports_and_timeouts():
    blocked = "def simple_math_problem():\n    import os\n    return 1"
    looping = "def simple_math_problem():\n    while True:\n        pass"
    assert code_answer(blocked, timeout=1) is None
    assert code_answer(looping, timeout=0.2) is None
