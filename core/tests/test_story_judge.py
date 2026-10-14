"""Judge scores must preserve alignment and expose failures instead of zero scores."""

import json
import sys

import pytest

from core.evaluation.story_judge import (
    RUBRICS, judge_requests, load_records, main, make_requests, parse_score,
)


@pytest.mark.parametrize("text", ["4", "[RESULT] 0", "[RESULT] 6", "[RESULT] 4.5",
                                  "[RESULT] 4/5", "[RESULT] 4 trailing",
                                  "[RESULT] 2 [RESULT] 4", "[RESULT] 44",
                                  "So the score is 4. [RESULT] N, where N is an integer 1–5.", None])
def test_parser_rejects_missing_ambiguous_and_invalid_scores(text):
    with pytest.raises(ValueError):
        parse_score(text)


@pytest.mark.parametrize("score", range(1, 6))
def test_parser_accepts_feedback_and_integer(score):
    assert parse_score(f"Feedback: The story is readable.\n[RESULT] {score}\n") == score


@pytest.mark.parametrize("ids", [[None], [""], ["a", "a"]])
def test_input_rejects_missing_and_duplicate_ids(tmp_path, ids):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps({"records": [
        {"id": value, "prompt": "Once", "continuation": "upon a time"} for value in ids
    ]}))
    with pytest.raises(ValueError, match="unique, nonempty"):
        load_records(path)


def test_failure_denominators_and_alignment():
    records = [{"id": str(i), "prompt": "Once", "continuation": "a story."} for i in range(2)]
    requests = make_requests(records)
    calls = []

    def provider(prompts):
        calls.append(prompts)
        return ["Feedback [RESULT] 4", "not a score", "Feedback [RESULT] 2", "[RESULT] 5"]

    result = judge_requests(requests, provider, 4)
    assert len(calls) == 2
    assert [item["record_id"] for item in result["records"]] == ["0"] * 4 + ["1"] * 4
    assert result["summary"]["grammar"] == {"mean": 4, "scored": 2, "failed": 0, "total": 2}
    assert result["summary"]["coherence"] == {"mean": None, "scored": 0, "failed": 2, "total": 2}
    assert result["records"][1]["score"] is None


def test_provider_errors_are_explicit():
    requests = make_requests([{"id": "a", "prompt": "", "continuation": ""}])
    result = judge_requests(requests, lambda batch: [], 4)
    assert all(item["status"] == "provider_error" for item in result["records"])
    assert all(value["failed"] == 1 for value in result["summary"].values())


def test_cli_prepare_and_score_offline(tmp_path, monkeypatch):
    samples, prepared, responses, scored = [tmp_path / name for name in (
        "samples.jsonl", "prepared.json", "responses.jsonl", "scored.json")]
    samples.write_text(json.dumps({"id": "story:1", "prompt": "Once upon a time,",
                                   "continuation": "a child made a friend."}) + "\n")
    monkeypatch.setattr(sys, "argv", ["story_judge", "--input", str(samples),
                                     "--output", str(prepared), "--prepare-only"])
    main()
    requests = json.loads(prepared.read_text())["requests"]
    assert len(requests) == len(RUBRICS)
    # Deliberately reverse results: matching must be by ID, not file position.
    responses.write_text("\n".join(json.dumps({"id": request["id"], "output": "[RESULT] 3"})
                                    for request in reversed(requests)))
    monkeypatch.setattr(sys, "argv", ["story_judge", "--input", str(samples),
                                     "--output", str(scored), "--responses", str(responses)])
    main()
    result = json.loads(scored.read_text())
    assert all(value["mean"] == 3 for value in result["summary"].values())
    assert result["metadata"]["backend"] == "offline_responses"
    assert len(result["metadata"]["input_sha256"]) == 64


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unknown"])
def test_offline_cli_rejects_misaligned_ids(tmp_path, monkeypatch, kind):
    record = {"id": "a", "prompt": "", "continuation": "Story"}
    samples, responses = tmp_path / "samples.json", tmp_path / "responses.jsonl"
    samples.write_text(json.dumps([record]))
    rows = [{"id": request["id"], "output": "[RESULT] 3"} for request in make_requests([record])]
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["id"] = "unknown"
    responses.write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr(sys, "argv", ["story_judge", "--input", str(samples),
                                     "--output", str(tmp_path / "result.json"),
                                     "--responses", str(responses)])
    with pytest.raises(ValueError):
        main()


def test_local_backend_does_not_duplicate_special_tokens_or_decode_prompt():
    import torch
    from core.evaluation.story_judge import LocalJudge

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            assert [message['role'] for message in messages] == ['system', 'user']
            return '<s>[INST]' + messages[0]['content'] + '\n' + messages[1]['content'] + '[/INST]'

        def __call__(self, texts, **kwargs):
            assert kwargs['add_special_tokens'] is False
            assert kwargs['truncation'] is False
            return Batch(input_ids=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                         attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]))

        def batch_decode(self, ids, **kwargs):
            assert ids.tolist() == [[4, 5], [4, 5]]
            return ['[RESULT] 4', '[RESULT] 5']

    class Model:
        device = 'cpu'

        def generate(self, **kwargs):
            assert kwargs['do_sample'] is False
            return torch.tensor([[0, 1, 2, 4, 5], [1, 2, 3, 4, 5]])

    judge = LocalJudge.__new__(LocalJudge)
    judge.torch = torch
    judge.tokenizer = Tokenizer()
    judge.model = Model()
    judge.max_new_tokens = 5
    judge.max_input_tokens = 10
    assert judge(['a', 'b']) == ['[RESULT] 4', '[RESULT] 5']
    judge.max_input_tokens = 2
    with pytest.raises(ValueError, match='no truncation'):
        judge(['a', 'b'])


def test_judge_preserves_algorithm_battery_provenance(tmp_path, monkeypatch):
    source, output = tmp_path / 'results.json', tmp_path / 'judge.json'
    benchmark = {'algorithm_id': 'method-hash', 'battery_id': 'battery-hash', 'task_id': 'tinystories'}
    source.write_text(json.dumps({'benchmark': benchmark, 'records': [
        {'id': 'story', 'prompt': 'Once', 'continuation': 'a child made a friend.'}]}))
    monkeypatch.setattr(sys, 'argv', ['story_judge', '--input', str(source),
                                    '--output', str(output), '--prepare-only'])
    main()
    assert json.loads(output.read_text())['metadata']['benchmark'] == benchmark
