from __future__ import annotations

import time

from biohub_tracking import kaggle_api


def test_rate_limit_cooldown_is_bounded_and_silent_of_secrets():
    kaggle_api._cool_until = 0.0
    kaggle_api._n_rate_limited = 0
    wait = kaggle_api._note_rate_limited("60")
    assert wait == 60.0
    assert kaggle_api._n_rate_limited == 1
    assert kaggle_api._cool_until > time.time()
    capped = kaggle_api._note_rate_limited("999")
    assert capped == 120.0
    floor = kaggle_api._note_rate_limited("1")
    assert floor == 30.0
    kaggle_api._cool_until = 0.0
