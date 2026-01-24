from desktop_agent.vision import ScreenRegion, map_point_preview_to_real, map_point_real_to_preview


def test_point_mapping_roundtrip():
    region = ScreenRegion(left=100, top=50, width=1000, height=500)
    preview_size = (200, 100)

    x, y = 600, 300
    px, py = map_point_real_to_preview(x, y, region, preview_size)
    x2, y2 = map_point_preview_to_real(px, py, region, preview_size)

    # Allow a small rounding difference.
    assert abs(x2 - x) <= 5
    assert abs(y2 - y) <= 5
