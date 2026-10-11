import numpy as np

from ba3.acquisition.policies import Query, select_query
from ba3.data.episodes import build_episodes, chronological_split
from ba3.models.boltzmann import exact_observed_fisher, fisher_reliability


def test_fisher_is_nonnegative_and_bounded_reliability():
    degrees = [np.array([1.0, 2.0, 4.0]), np.array([2.0, 3.0, 4.0])]
    value = exact_observed_fisher(1.0, [2, 1], degrees)
    assert value >= 0.0
    assert 0.0 <= fisher_reliability(value) <= 1.0


def test_active_policy_respects_cost():
    queries = [Query("C1", "auth", 1.0), Query("C2", "flow", 5.0)]
    posterior = np.array([0.5, 0.5])
    outcomes = lambda query: [np.array([0.9, 0.1]), np.array([0.1, 0.9])]
    chosen = select_query(queries, outcomes, posterior, "entropy", np.random.default_rng(0))
    assert chosen.host == "C1"


def test_chronological_split_has_no_overlap(tmp_path):
    import gzip

    path = tmp_path / "redteam.txt.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("1,U1,C9,C1\n2,U1,C9,C2\n1000,U2,C8,C3\n1001,U2,C8,C4\n")
    episodes = build_episodes(path, gap_seconds=10)
    splits = chronological_split(episodes, fractions=(0.5, 0.0, 0.5))
    train_ids = {item.episode_id for item in splits["train"]}
    test_ids = {item.episode_id for item in splits["test"]}
    assert train_ids.isdisjoint(test_ids)
