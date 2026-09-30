import math
import os
import sys

FONTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fonts"))
CALIBRATION_FONT = "LightsStemCalibration.ttf"
CALIBRATION_UPEM = 2048
CALIBRATION_STEMS = {"I": (256, 0, 544, 1440), "l": (256, 0, 320, 1440)}
CALIBRATION_RING = (700, 700, 600, 380, 8)
CALIBRATION_CROSS = ((200, 900, 1800, 1100), (900, 200, 1100, 1800))
MODES = ("coretext", "coretext-path", "coretext-smoothed-dark", "coretext-smoothed-light", "freetype", "freetype-light")
HEADER = ("name", "font", "glyph_id", "em_px", "phase_index", "phase", "origin_x", "origin_y", "width", "height")


def calibration_font(path):
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib.tables._g_l_y_f import flagOverlapSimple

    def rects(*boxes):
        pen = TTGlyphPen(None)
        for x0, y0, x1, y1 in boxes:
            pen.moveTo((x0, y0))
            pen.lineTo((x0, y1))
            pen.lineTo((x1, y1))
            pen.lineTo((x1, y0))
            pen.closePath()
        return pen.glyph()

    def ring(cx, cy, outer, inner, segments):
        pen = TTGlyphPen(None)
        for radius, turn in ((outer, -1.0), (inner, 1.0)):
            reach = radius / math.cos(math.pi / segments)
            at = lambda r, k: (round(cx + r * math.cos(turn * 2.0 * math.pi * k / segments)), round(cy + r * math.sin(turn * 2.0 * math.pi * k / segments)))
            pen.moveTo(at(radius, 0))
            for k in range(segments):
                pen.qCurveTo(at(reach, k + 0.5), at(radius, k + 1))
            pen.closePath()
        return pen.glyph()

    builder = FontBuilder(CALIBRATION_UPEM, isTTF=True)
    builder.setupGlyphOrder([".notdef", "I", "l", "o", "plus"])
    builder.setupCharacterMap({ord("I"): "I", ord("l"): "l", ord("o"): "o", ord("+"): "plus"})
    glyphs = {".notdef": TTGlyphPen(None).glyph()}
    glyphs.update({name: rects(box) for name, box in CALIBRATION_STEMS.items()})
    glyphs["o"] = ring(*CALIBRATION_RING)
    glyphs["plus"] = rects(*CALIBRATION_CROSS)
    glyphs["plus"].flags[0] |= flagOverlapSimple
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics({".notdef": (800, 0), "I": (800, 256), "l": (576, 256), "o": (1400, 100), "plus": (2000, 200)})
    builder.setupHorizontalHeader(ascent=1900, descent=-500)
    builder.setupNameTable({"familyName": "Lights Stem Calibration", "styleName": "Regular"})
    builder.setupOS2(sTypoAscender=1900, sTypoDescender=-500, usWinAscent=1900, usWinDescent=500)
    builder.setupPost()
    builder.updateHead(created=0, modified=0)
    builder.save(path)
    print(f"wrote {path}")


def read_manifest(directory):
    with open(os.path.join(directory, "manifest.tsv"), encoding="utf-8") as f:
        lines = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    if tuple(lines[0]) != HEADER:
        raise SystemExit(f"manifest header {lines[0]} is not {list(HEADER)}")
    rows = []
    for cells in lines[1:]:
        row = dict(zip(HEADER, cells))
        rows.append(
            {
                "name": row["name"],
                "font": row["font"],
                "glyph_id": int(row["glyph_id"]),
                "em_px": int(row["em_px"]),
                "origin": (float(row["origin_x"]), float(row["origin_y"])),
                "size": (int(row["width"]), int(row["height"])),
            }
        )
    return rows


class CoreText:
    def __init__(self, smoothed, dark_on_light, path=False, subpixel=True, quantised=False):
        import CoreText as ct
        import Quartz
        from Foundation import NSData

        self.ct, self.q, self.nsdata = ct, Quartz, NSData
        self.smoothed, self.dark_on_light, self.path = smoothed, dark_on_light, path
        self.subpixel, self.quantised = subpixel, quantised
        self.graphics_fonts = {}

    def graphics_font(self, file):
        if file not in self.graphics_fonts:
            data = open(os.path.join(FONTS, file), "rb").read()
            provider = self.q.CGDataProviderCreateWithCFData(self.nsdata.dataWithBytes_length_(data, len(data)))
            self.graphics_fonts[file] = self.q.CGFontCreateWithDataProvider(provider)
        return self.graphics_fonts[file]

    def render(self, row):
        q = self.q
        width, height = row["size"]
        font = self.ct.CTFontCreateWithGraphicsFont(self.graphics_font(row["font"]), row["em_px"], None, None)
        context = q.CGBitmapContextCreate(None, width, height, 8, width, q.CGColorSpaceCreateDeviceGray(), q.kCGImageAlphaNone)
        paper, ink = (1.0, 0.0) if self.dark_on_light else (0.0, 1.0)
        q.CGContextSetGrayFillColor(context, paper, 1.0)
        q.CGContextFillRect(context, q.CGRectMake(0, 0, width, height))
        q.CGContextSetGrayFillColor(context, ink, 1.0)
        q.CGContextSetAllowsAntialiasing(context, True)
        q.CGContextSetShouldAntialias(context, True)
        q.CGContextSetAllowsFontSmoothing(context, self.smoothed)
        q.CGContextSetShouldSmoothFonts(context, self.smoothed)
        q.CGContextSetAllowsFontSubpixelPositioning(context, self.subpixel)
        q.CGContextSetShouldSubpixelPositionFonts(context, self.subpixel)
        q.CGContextSetAllowsFontSubpixelQuantization(context, self.quantised)
        q.CGContextSetShouldSubpixelQuantizeFonts(context, self.quantised)
        if self.path:
            q.CGContextTranslateCTM(context, *row["origin"])
            q.CGContextAddPath(context, self.ct.CTFontCreatePathForGlyph(font, row["glyph_id"], None))
            q.CGContextFillPath(context)
        else:
            self.ct.CTFontDrawGlyphs(font, [row["glyph_id"]], [q.CGPointMake(*row["origin"])], 1, context)
        image = q.CGBitmapContextCreateImage(context)
        raw = bytes(q.CGDataProviderCopyData(q.CGImageGetDataProvider(image)))
        stride = q.CGImageGetBytesPerRow(image)
        out = bytearray()
        for m in range(height):
            line = raw[m * stride : m * stride + width]
            out += bytes(255 - v for v in line) if self.dark_on_light else line
        return bytes(out)


class FreeType:
    def __init__(self, light_hinting):
        import freetype

        self.ft = freetype
        self.flags = (freetype.FT_LOAD_TARGET_LIGHT if light_hinting else freetype.FT_LOAD_NO_HINTING) | freetype.FT_LOAD_NO_BITMAP
        self.faces = {}

    def face(self, file):
        if file not in self.faces:
            self.faces[file] = self.ft.Face(os.path.join(FONTS, file))
        return self.faces[file]

    def render(self, row):
        ft = self.ft
        width, height = row["size"]
        face = self.face(row["font"])
        face.set_char_size(0, row["em_px"] * 64, 72, 72)
        ox, oy = row["origin"]
        ix, iy = math.floor(ox), math.floor(oy)
        delta = ft.Vector(int(round((ox - ix) * 64)), int(round((oy - iy) * 64)))
        face.set_transform(ft.Matrix(0x10000, 0, 0, 0x10000), delta)
        face.load_glyph(row["glyph_id"], self.flags)
        face.glyph.render(ft.FT_RENDER_MODE_NORMAL)
        bitmap, left, top = face.glyph.bitmap, face.glyph.bitmap_left, face.glyph.bitmap_top
        buffer = bitmap.buffer
        out = bytearray(width * height)
        for r in range(bitmap.rows):
            for c in range(bitmap.width):
                v = buffer[r * bitmap.pitch + c]
                i, m = ix + left + c, height - iy - top + r
                if 0 <= i < width and 0 <= m < height:
                    out[m * width + i] = v
                elif v:
                    raise SystemExit(f"{row['name']} at {row['em_px']} px/em: FreeType drew ink outside the {width}x{height} viewport")
        return bytes(out)


def raster(mode):
    return {
        "coretext": lambda: CoreText(smoothed=False, dark_on_light=False),
        "coretext-path": lambda: CoreText(smoothed=False, dark_on_light=False, path=True),
        "coretext-smoothed-dark": lambda: CoreText(smoothed=True, dark_on_light=True),
        "coretext-smoothed-light": lambda: CoreText(smoothed=True, dark_on_light=False),
        "freetype": lambda: FreeType(light_hinting=False),
        "freetype-light": lambda: FreeType(light_hinting=True),
    }[mode]()


def render(directory, modes):
    rows = read_manifest(directory)
    for mode in modes:
        drawer = raster(mode)
        path = os.path.join(directory, f"{mode}.pgm")
        with open(path, "wb") as out:
            for row in rows:
                width, height = row["size"]
                out.write(b"P5\n%d %d\n255\n" % (width, height) + drawer.render(row))
        print(f"{mode}: {len(rows)} images -> {path}")


def stem_box(name, em_px, origin, size):
    x0, y0, x1, y1 = (c * em_px / CALIBRATION_UPEM for c in CALIBRATION_STEMS[name])
    x0, x1, y0, y1 = x0 + origin[0], x1 + origin[0], y0 + origin[1], y1 + origin[1]
    width, height = size
    overlap = lambda a0, a1, b0, b1: max(0.0, min(a1, b1) - max(a0, b0))
    return [[overlap(i, i + 1, x0, x1) * overlap(height - 1 - m, height - m, y0, y1) for i in range(width)] for m in range(height)]


def as_rows(image, size):
    width, height = size
    return [[image[m * width + i] / 255.0 for i in range(width)] for m in range(height)]


def centroid(rows):
    ink = sum(map(sum, rows))
    x = sum((i + 0.5) * v for row in rows for i, v in enumerate(row)) / ink
    y = sum((m + 0.5) * v for m, row in enumerate(rows) for v in row) / ink
    return x, y


def probe():
    import random

    ladder = (8, 10, 12, 14, 16, 20, 24, 32, 48, 64, 96, 128)
    seed = 173
    arms = [
        ("coretext as rendered", lambda: CoreText(False, False), False),
        ("coretext, sub-pixel quantisation on", lambda: CoreText(False, False, quantised=True), False),
        ("coretext, sub-pixel positioning off", lambda: CoreText(False, False, subpixel=False), False),
        ("freetype, origin on its 1/64 px grid", lambda: FreeType(False), True),
        ("freetype, arbitrary origin", lambda: FreeType(False), False),
    ]
    def stem_steps(drawer, em, ox, oy):
        size = (math.ceil(544 * em / CALIBRATION_UPEM + ox + 3), math.ceil(1440 * em / CALIBRATION_UPEM + oy + 3))
        row = {"name": "calibration-stem", "font": CALIBRATION_FONT, "glyph_id": 1, "em_px": em, "origin": (ox, oy), "size": size}
        drawn, truth = as_rows(drawer.render(row), size), stem_box("I", em, (ox, oy), size)
        (cx, cy), (bx, by) = centroid(drawn), centroid(truth)
        return 255.0 * max(abs(a - b) for p, q in zip(drawn, truth) for a, b in zip(p, q)), (cx - bx, cy - by)

    print(f"placements: calibration stem 'I', {len(ladder)} rungs, seed {seed}, 240 origins drawn in [2, 3) px on each axis")
    print("arm\tworst_steps\tp90_steps\tmedian_steps\tworst_centroid_px\tmean_centroid_dx_px\tworst_at")
    for label, make, on_grid in arms:
        rng, drawer, worst, shifts = random.Random(seed), make(), [], []
        for _ in range(240):
            em = rng.choice(ladder)
            ox, oy = 2.0 + rng.random(), 2.0 + rng.random()
            if on_grid:
                ox, oy = round(ox * 64) / 64, round(oy * 64) / 64
            steps, shift = stem_steps(drawer, em, ox, oy)
            worst.append((steps, f"{em} px/em origin ({ox:.5f}, {oy:.5f})"))
            shifts.append(shift)
        worst.sort()
        far = max(max(abs(dx), abs(dy)) for dx, dy in shifts)
        print(
            f"{label}\t{worst[-1][0]:.2f}\t{worst[int(0.9 * 239)][0]:.2f}\t{worst[120][0]:.2f}\t{far:.4f}\t"
            f"{sum(dx for dx, _ in shifts) / 240:+.5f}\t{worst[-1][1]}"
        )
    print("edge band: coretext as rendered, the stem's bottom edge (or its left edge) a distance above (or right of) a pixel boundary")
    gaps = (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.03, 0.06)
    print("em_px\tedge\t" + "\t".join(f"{g}" for g in gaps))
    for em in (10, 16, 32):
        drawer = CoreText(False, False)
        for edge in ("bottom", "left"):
            cells = [stem_steps(drawer, em, 2.2023 if edge == "bottom" else 2.0 + g, 2.0 + g if edge == "bottom" else 2.5)[0] for g in gaps]
            print(f"{em}\t{edge}\t" + "\t".join(f"{c:.1f}" for c in cells))
    import ctypes

    import freetype
    from freetype import raw

    handle = freetype.get_handle()
    darkening = []
    for module in ("autofitter", "cff", "type1", "t1cid"):
        flag = ctypes.c_ubyte(255)
        error = raw.FT_Property_Get(handle, module.encode(), b"no-stem-darkening", ctypes.byref(flag))
        darkening.append(f"{module} {flag.value if error == 0 else 'error %d' % error}")
    interpreter = ctypes.c_uint(0)
    raw.FT_Property_Get(handle, b"truetype", b"interpreter-version", ctypes.byref(interpreter))
    print(f"freetype defaults: no-stem-darkening {', '.join(darkening)}; truetype interpreter-version {interpreter.value}")
    print("dilation: font smoothing on, origin (2.3, 2.6), the stem's width and height gain in px, one full row and one full column")
    print("em_px\tdark_on_light_width\tdark_on_light_height\tlight_on_dark_width\tlight_on_dark_height")
    for em in ladder + (200,):
        cells = []
        for dark in (True, False):
            origin = (2.3, 2.6)
            size = (math.ceil(544 * em / CALIBRATION_UPEM + 7), math.ceil(1440 * em / CALIBRATION_UPEM + 7))
            row = {"name": "calibration-stem", "font": CALIBRATION_FONT, "glyph_id": 1, "em_px": em, "origin": origin, "size": size}
            drawn, truth = as_rows(CoreText(True, dark).render(row), size), stem_box("I", em, origin, size)
            m = size[1] // 2
            i = int(400 * em / CALIBRATION_UPEM + origin[0])
            full = max(r[i] for r in truth) > 0.999
            height = sum(r[i] for r in drawn) - sum(r[i] for r in truth) if full else float("nan")
            cells += [sum(drawn[m]) - sum(truth[m]), height]
        print(f"{em}\t" + "\t".join(f"{c:.4f}" for c in cells))


def versions():
    import platform
    from importlib.metadata import version

    import freetype

    print(
        f"macOS {platform.mac_ver()[0]}, FreeType {'.'.join(map(str, freetype.version()))} "
        f"(freetype-py {version('freetype-py')}), pyobjc {version('pyobjc-core')}"
    )


def main(argv):
    if len(argv) >= 3 and argv[1] == "font":
        calibration_font(argv[2])
    elif len(argv) >= 3 and argv[1] == "render":
        modes = argv[3].split(",") if len(argv) > 3 else list(MODES)
        unknown = [m for m in modes if m not in MODES]
        if unknown:
            raise SystemExit(f"unknown modes {unknown}; known {list(MODES)}")
        versions()
        render(argv[2], modes)
    elif len(argv) >= 2 and argv[1] == "probe":
        versions()
        probe()
    else:
        raise SystemExit("usage: glyph_oracle.py font <out.ttf> | render <dir> [mode,mode,...] | probe")


if __name__ == "__main__":
    main(sys.argv)
