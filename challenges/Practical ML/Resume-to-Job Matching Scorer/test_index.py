import numpy as np
from index import JobIndex


def _unit(rows):
    m = np.array(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_search_returns_the_nearest_job_first_with_cosine_similarity(tmp_path):
    vecs = _unit([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]])
    idx = JobIndex.build(tmp_path, ["a", "b", "c"], vecs)
    hits = idx.search(_unit([[1, 0.05, 0]]), k=2)[0]
    assert [j for j, _ in hits] == ["a", "c"]
    assert hits[0][1] > hits[1][1]
    assert hits[0][1] == np.float32(hits[0][1]) and 0.9 < hits[0][1] <= 1.0


def test_batch_search_equals_one_query_at_a_time(tmp_path):
    rng = np.random.default_rng(0)
    vecs = _unit(rng.normal(size=(30, 8)))
    idx = JobIndex.build(tmp_path, [f"j{i}" for i in range(30)], vecs)
    queries = _unit(rng.normal(size=(5, 8)))
    batch = idx.search(queries, k=4)
    single = [idx.search(q[None, :], k=4)[0] for q in queries]
    assert [[j for j, _ in h] for h in batch] == [[j for j, _ in h] for h in single]


def test_k_larger_than_the_index_returns_everything_and_reopen_works(tmp_path):
    vecs = _unit([[1, 0], [0, 1]])
    JobIndex.build(tmp_path, ["a", "b"], vecs)
    reopened = JobIndex.open(tmp_path)
    assert len(reopened.search(_unit([[1, 1]]), k=10)[0]) == 2
