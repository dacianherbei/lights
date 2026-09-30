import zlib, struct, base64, os

def read_png(path):
    d = open(path, 'rb').read()
    assert d[:8] == b'\x89PNG\r\n\x1a\n', path
    i, idat, w = 8, b'', None
    while i < len(d):
        ln = struct.unpack('>I', d[i:i+4])[0]; typ = d[i+4:i+8]; data = d[i+8:i+8+ln]
        if typ == b'IHDR': w, h, bd, ct = struct.unpack('>IIBB', data[:10])
        elif typ == b'IDAT': idat += data
        i += 12 + ln
    raw = zlib.decompress(idat); ch = {0:1, 2:3, 4:2, 6:4}[ct]; stride = w*ch
    out, prev, pos = [], bytearray(stride), 0
    for _ in range(h):
        f = raw[pos]; pos += 1
        line = bytearray(raw[pos:pos+stride]); pos += stride
        for x in range(stride):
            a = line[x-ch] if x >= ch else 0
            b = prev[x]
            c = prev[x-ch] if x >= ch else 0
            if f == 1: line[x] = (line[x]+a) & 255
            elif f == 2: line[x] = (line[x]+b) & 255
            elif f == 3: line[x] = (line[x]+(a+b)//2) & 255
            elif f == 4:
                p = a+b-c; pa, pb, pc = abs(p-a), abs(p-b), abs(p-c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x]+pr) & 255
        out.append(bytes(line)); prev = line
    return w, h, ch, out

def write_gray_png(w, h, rows):
    def chunk(t, data):
        c = t + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    raw = b''.join(b'\x00' + bytes(r) for r in rows)
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 9))
            + chunk(b'IEND', b''))

def maxdiff(truth, other):
    (w, h, c1, r1), (w2, h2, c2, r2) = truth, other
    assert (w, h) == (w2, h2)
    field = []
    peak = 0
    for y in range(h):
        row = []
        a, b = r1[y], r2[y]
        for x in range(w):
            m = 0
            for k in range(min(3, c1, c2)):
                m = max(m, abs(a[x*c1+k] - b[x*c2+k]))
            row.append(m)
            if m > peak: peak = m
        field.append(row)
    return field, peak

def encode(field, w, h, norm):
    if norm <= 0: norm = 1
    rows = [bytes(min(255, int(round(v * 255.0 / norm))) for v in r) for r in field]
    return "data:image/png;base64," + base64.b64encode(write_gray_png(w, h, rows)).decode()
