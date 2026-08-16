from unittest import mock

import tasks.gsm8k as gsm8k
from tasks.gsm8k import (
    _bubblewrap_command,
    code_answer,
    evaluate_codes,
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


def test_numeric_scoring_handles_signs_decimals_commas_and_last_answer():
    assert numeric_answer("estimate 1,000; final answer is -$1,234.50") == -1234.5
    assert numeric_answer("answer: 2 then corrected answer: 3.25") == 3.25
    assert gold_answer("work\n#### $12,345") == 12345
    assert score_response("result = 0.333", "#### 0.3334")


def test_numeric_scoring_rejects_numbers_too_large_for_float():
    assert numeric_answer("answer: " + "9" * 10000) is None


def test_code_scoring_runs_in_subprocess():
    response = "```python\ndef simple_math_problem():\n    return 6 * 7\n```"
    assert extract_code(response).startswith("def simple_math_problem")
    assert code_answer(response, timeout=2) == 42
    assert score_response(response, "#### 42", method="code", timeout=2)
    with_math = "def simple_math_problem():\n    import math\n    return math.sqrt(81)"
    assert code_answer(with_math, timeout=2) == 9


def test_batch_code_scoring_reuses_one_bubblewrap_worker():
    responses = [
        "def simple_math_problem():\n    return 6 * 7",
        "def simple_math_problem():\n    return 8 * 9",
    ]
    real_popen = gsm8k.subprocess.Popen
    with mock.patch.object(gsm8k.subprocess, "Popen", wraps=real_popen) as popen:
        evaluations = evaluate_codes(responses, timeout=2)
    assert [evaluation.result for evaluation in evaluations] == [42, 72]
    assert popen.call_count == 1


def test_code_scoring_uses_network_and_filesystem_isolated_bubblewrap():
    command = _bubblewrap_command()
    assert "--unshare-net" in command
    assert ["--tmpfs", "/tmp"] == command[
        command.index("--tmpfs") : command.index("--tmpfs") + 2
    ]
    assert "/home" not in command


def test_code_scoring_rejects_imports_and_timeouts():
    blocked = "def simple_math_problem():\n    import os\n    return 1"
    looping = "def simple_math_problem():\n    while True:\n        pass"
    assert code_answer(blocked, timeout=1) is None
    assert code_answer(looping, timeout=0.2) is None


def test_code_scoring_blocks_indirect_imports_and_private_attribute_access():
    indirect_import = "def simple_math_problem():\n    return __import__('os').getcwd()"
    private_attribute = "def simple_math_problem():\n    return (1).__class__"
    assert code_answer(indirect_import, timeout=1) is None
    assert code_answer(private_attribute, timeout=1) is None
