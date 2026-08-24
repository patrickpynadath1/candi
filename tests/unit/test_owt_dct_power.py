import torch

from scripts.analyze_owt_dct_power import (
    accumulate_mode_power,
    corpus_centroid,
    dct_matrix,
    packed_token_sequences,
    save_plot,
    summarize,
)


class TinyTokenizer:
    eos_token_id = 9

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [int(piece) for piece in text.split()]}


def test_packing_inserts_eos_and_carries_across_documents():
    sequences = list(packed_token_sequences(["1 2", "3 4 5"], TinyTokenizer(), 4, 2))
    assert sequences == [[1, 2, 9, 3]]


def test_dct_is_orthonormal():
    basis = dct_matrix(16, device=torch.device("cpu"), dtype=torch.float64)
    torch.testing.assert_close(basis.T @ basis, torch.eye(16, dtype=torch.float64))


def test_mode_power_obeys_parseval_and_detects_constant_mode():
    embedding = torch.nn.Embedding.from_pretrained(
        torch.tensor([[1.0, -2.0], [0.0, 0.0]])
    )
    stats = accumulate_mode_power([[0, 0, 0, 0]], embedding, 4, 1, torch.device("cpu"))
    result = summarize(stats, embedding_dim=2)
    # Four positions each have squared norm five.
    assert abs(result["mean_power"].sum() - 20.0) < 1e-5
    assert abs(result["mean_power"][0] - 20.0) < 1e-5
    assert result["mean_power"][1:].max() < 1e-10


def test_random_embedding_baseline_is_reported():
    learned = torch.nn.Embedding.from_pretrained(torch.tensor([[2.0], [0.0]]))
    random = torch.nn.Embedding.from_pretrained(torch.tensor([[1.0], [0.0]]))
    stats = accumulate_mode_power(
        [[0, 0, 0, 0]],
        learned,
        4,
        1,
        torch.device("cpu"),
        random_embedding=random,
    )
    result = summarize(stats, embedding_dim=1)
    assert abs(result["learned_over_random_power"][0] - 4.0) < 1e-6


def test_corpus_centroid_is_token_weighted_and_removes_global_mean():
    embedding = torch.nn.Embedding.from_pretrained(
        torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    )
    sequences = [[0, 0, 0, 1]]
    centroid = corpus_centroid(sequences, embedding)
    torch.testing.assert_close(centroid, torch.tensor([2.0, 4.0]))
    centered = embedding(torch.tensor(sequences)) - centroid
    torch.testing.assert_close(centered.mean(dim=(0, 1)), torch.zeros(2))


def test_plot_writes_svg(tmp_path):
    import numpy as np

    path = tmp_path / "power.svg"
    arrays = {
        "mean_power": np.array([4.0, 2.0, 1.0]),
        "random_mean_power": np.array([1.0, 1.0, 1.0]),
        "learned_over_random_power": np.array([4.0, 2.0, 1.0]),
        "cumulative_power_fraction": np.array([4 / 7, 6 / 7, 1.0]),
    }
    save_plot(path, arrays, {"num_sequences": 2, "sequence_length": 3})
    contents = path.read_text()
    assert contents.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "Learned / random power" in contents
