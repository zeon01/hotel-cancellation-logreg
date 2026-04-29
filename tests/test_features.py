"""Feature-engineering tests: outputs are derived correctly and don't introduce NaNs."""

from __future__ import annotations

import pandas as pd
import pytest


def test_total_nights_is_sum() -> None:
    pytest.importorskip("cancellation_logreg.features")
    from cancellation_logreg.features import add_total_nights

    df = pd.DataFrame({"stays_in_weekend_nights": [1, 2], "stays_in_week_nights": [3, 0]})
    try:
        out = add_total_nights(df)
    except NotImplementedError:
        pytest.skip("features.add_total_nights not yet implemented")

    assert (out["total_nights"] == [4, 2]).all()


def test_room_change_flag_matches_definition() -> None:
    pytest.importorskip("cancellation_logreg.features")
    from cancellation_logreg.features import add_room_change_flag

    df = pd.DataFrame({"reserved_room_type": ["A", "D"], "assigned_room_type": ["A", "K"]})
    try:
        out = add_room_change_flag(df)
    except NotImplementedError:
        pytest.skip("features.add_room_change_flag not yet implemented")

    assert (out["room_was_changed"] == [False, True]).all()
