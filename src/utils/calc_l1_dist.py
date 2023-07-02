from src.utils.constants import HEIGHT, WIDTH


def _c(dist, type):
    assert type in ("height", "width")
    ret = dist
    if type == "height":
        if abs(dist) > HEIGHT/2:
            if dist > 0:
                ret = dist - HEIGHT
            else:
                ret = dist + HEIGHT
    else:
        if abs(dist) > WIDTH/2:
            if dist > 0:
                ret = dist - WIDTH
            else:
                ret = dist + WIDTH
    return ret

def l1_dist(a, b):
    return abs(_c(a[0] - b[0], "height")) + abs(_c(a[1] - b[1], "width"))