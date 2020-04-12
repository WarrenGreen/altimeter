from altimeter.kalman_filter import update


def test_update():
    assert update(20, 9, 30, 3) == (27.5, 2.25)
