import sys

from fontTools.pens.pointInsidePen import PointInsidePen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont

GLYPH_EM = 12.0


def outline(path, char):
    font = TTFont(path)
    glyphs = font.getGlyphSet()
    glyph = glyphs[font.getBestCmap()[ord(char)]]
    pen = RecordingPen()
    glyph.draw(pen)
    edges, contour, start, cur = [], -1, None, None
    for op, args in pen.value:
        if op == "moveTo":
            contour += 1
            start = cur = args[0]
        elif op == "lineTo":
            edges.append((contour, cur, args[0]))
            cur = args[0]
        elif op in ("qCurveTo", "curveTo"):
            cur = args[-1]
        elif op in ("closePath", "endPath"):
            if cur is not None and cur != start:
                edges.append((contour, cur, start))
            cur = None
    return font["head"].unitsPerEm, contour + 1, edges, glyphs, glyph


def inked(glyphs, glyph, point):
    pen = PointInsidePen(glyphs, point, evenOdd=False)
    glyph.draw(pen)
    return pen.getResult()


def facing_pairs(edges, glyphs, glyph):
    out = []
    for a in range(len(edges)):
        for b in range(a + 1, len(edges)):
            (ca, p, q), (cb, r, s) = edges[a], edges[b]
            for axis in (0, 1):
                other = 1 - axis
                if p[axis] != q[axis] or r[axis] != s[axis] or p[axis] == r[axis]:
                    continue
                lo = max(min(p[other], q[other]), min(r[other], s[other]))
                hi = min(max(p[other], q[other]), max(r[other], s[other]))
                if hi <= lo:
                    continue
                mid = [0.0, 0.0]
                mid[axis] = 0.5 * (p[axis] + r[axis])
                mid[other] = 0.5 * (lo + hi)
                side = "ink" if inked(glyphs, glyph, tuple(mid)) else "paper"
                out.append((abs(p[axis] - r[axis]), side, "xy"[axis], p[axis], r[axis], hi - lo, ca, cb))
    return sorted(out)


def main(argv):
    path, char = argv[1], argv[2]
    count = int(argv[3]) if len(argv) > 3 else 10
    upem, contours, edges, glyphs, glyph = outline(path, char)
    print(f"{path} {char!r}: {upem} units per em, {contours} contours, {len(edges)} straight edges")
    for gap, side, axis, a, b, facing, ca, cb in facing_pairs(edges, glyphs, glyph)[:count]:
        print(
            f"{side:5} {axis}={a:g} and {b:g}: {gap:g} font units = {gap / upem * GLYPH_EM:.4f} glyph units, "
            f"facing over {facing:g} (contours {ca}, {cb})"
        )


if __name__ == "__main__":
    main(sys.argv)
