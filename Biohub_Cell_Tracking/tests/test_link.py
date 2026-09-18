import numpy as np

from biohub_tracking.config import LinkingConfig
from biohub_tracking.link import link_frames


SCALE = (1.625, 0.40625, 0.40625)


def test_greedy_and_hungarian_link_nearest():
    prev = np.array([[10, 10, 10], [10, 40, 10]], dtype=float)
    curr = np.array([[10, 11, 10], [10, 41, 10]], dtype=float)
    greedy = link_frames(prev, curr, LinkingConfig(method="greedy"), SCALE)
    hung = link_frames(prev, curr, LinkingConfig(method="hungarian"), SCALE)
    assert set(greedy) == {(0, 0), (1, 1)}
    assert set(hung) == {(0, 0), (1, 1)}


def test_max_distance_blocks_far_links():
    prev = np.array([[0, 0, 0]], dtype=float)
    curr = np.array([[0, 200, 200]], dtype=float)
    links = link_frames(prev, curr, LinkingConfig(method="hungarian", max_distance_um=15.0), SCALE)
    assert links == []
