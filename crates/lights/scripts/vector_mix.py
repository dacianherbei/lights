import argparse
import re
import sys

# Lane specs that mean "several lanes at once". `.s`, `.d`, `.h`, `.b` with no
# count are SINGLE-element forms -- `mov.s v0[1], w8` moves one lane -- and they
# are not width. Keeping them out of WIDE is the difference between counting
# arithmetic and counting register traffic; the first version of this script
# scored `mov.s` and `ld1.s` as compute and inflated splat by 18%.
WIDE = r"(?:16b|8b|4h|8h|2s|4s|2d)"
LANE_ANY = r"(?:16b|8b|4h|8h|2s|4s|1d|2d|[bhsd])"
MNEMONIC_WIDE = re.compile(r"^[a-z][a-z0-9]*\.(" + WIDE + r")$")
MNEMONIC_LANE_ANY = re.compile(r"^([a-z][a-z0-9]*)\.(" + LANE_ANY + r")$")
OPERAND_VECTOR = re.compile(r"\bv\d+(?:\.(?:" + LANE_ANY + r"))?\b")
SCALAR_FP_REG = re.compile(r"\b[sdh]\d+\b")

# A vector instruction that only moves or permutes bits is not the payoff the
# design claims. `mov.16b` between two vector registers is the commonest false
# positive and it is what the 2026-09-09 reading of this counted.
VECTOR_MOVES = {
    "mov", "movi", "mvni", "fmov", "dup", "ins", "umov", "smov",
    "ldr", "str", "ldp", "stp", "ld1", "ld2", "ld3", "ld4",
    "st1", "st2", "st3", "st4", "ldur", "stur",
}
VECTOR_SHUFFLES = {
    "ext", "zip1", "zip2", "uzp1", "uzp2", "trn1", "trn2",
    "rev16", "rev32", "rev64", "tbl", "tbx", "xtn", "xtn2",
}

FP_MNEMONICS = re.compile(
    r"^(fadd|fsub|fmul|fdiv|fmla|fmls|fmadd|fmsub|fnmadd|fnmsub|fneg|fabs|"
    r"fsqrt|fmax|fmin|fmaxnm|fminnm|fcmp|fcmpe|fccmp|fcsel|fcvt|fcvtz|fcvtzs|"
    r"fcvtzu|fcvtas|fcvtau|fcvtms|fcvtmu|fcvtns|fcvtnu|fcvtps|fcvtpu|"
    r"scvtf|ucvtf|frinta|frinti|frintm|frintn|frintp|frintx|frintz|"
    r"faddp|fmaxv|fminv|fmulx|frecpe|frecps|frsqrte|frsqrts)"
)


def split_base(mnemonic):
    """Return (base mnemonic, lane spec or None, is a MULTI-lane form)."""
    hit = MNEMONIC_LANE_ANY.match(mnemonic)
    if hit:
        return hit.group(1), hit.group(2), bool(MNEMONIC_WIDE.match(mnemonic))
    return mnemonic, None, False


def classify(line):
    body = line.strip()
    if not body or body.startswith(".") or body.startswith(";") or body.endswith(":"):
        return None
    parts = body.split(None, 1)
    mnemonic = parts[0].lower()
    operands = parts[1] if len(parts) > 1 else ""
    base, lane, wide = split_base(mnemonic)
    is_vector = lane is not None or bool(OPERAND_VECTOR.search(operands))
    if is_vector:
        if base in VECTOR_MOVES:
            return "vector_move"
        if base in VECTOR_SHUFFLES:
            return "vector_shuffle"
        if not wide:
            # A single-lane form doing arithmetic is scalar work in a vector
            # register, not width.
            return "vector_narrow"
        return "vector_wide4" if lane in ("4s", "16b", "8h") else "vector_wide2"
    if FP_MNEMONICS.match(base) and SCALAR_FP_REG.search(operands):
        return "scalar_fp"
    if base in ("ldr", "str", "ldp", "stp", "ldur", "stur") and SCALAR_FP_REG.search(operands):
        return "scalar_fp_mem"
    return "other"


def bodies(path, needle):
    text = open(path, encoding="utf-8", errors="replace").read().split("\n")
    out, name, acc = [], None, []
    for line in text:
        if line.startswith("_") and line.rstrip().endswith(":"):
            if name is not None:
                out.append((name, acc))
            name, acc = line.rstrip()[:-1], []
            continue
        if name is not None:
            acc.append(line)
    if name is not None:
        out.append((name, acc))
    return [(n, b) for n, b in out if needle in n]


def tally(body):
    counts = {}
    for line in body:
        kind = classify(line)
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asm")
    ap.add_argument("--needle", default="")
    ap.add_argument("--min-instructions", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    found = bodies(args.asm, args.needle)
    if not found:
        print(f"refusing: no symbol in {args.asm} contains {args.needle!r}")
        return 3

    rows, shown = [], 0
    for name, body in found:
        counts = tally(body)
        total = sum(counts.values())
        if total < args.min_instructions:
            continue
        rows.append((name, total, counts))

    if not rows:
        print(f"refusing: {len(found)} symbols matched but none has "
              f"{args.min_instructions} instructions")
        return 3

    def wide(c):
        return c.get("vector_wide2", 0) + c.get("vector_wide4", 0)

    if args.list:
        print(f"{'instrs':>7} {'w4':>4} {'w2':>5} {'narrow':>7} {'shuf':>5} "
              f"{'move':>5} {'scalarfp':>9}  symbol")
        for name, total, c in sorted(rows, key=lambda r: -r[1]):
            print(f"{total:7d} {c.get('vector_wide4',0):4d} {c.get('vector_wide2',0):5d} "
                  f"{c.get('vector_narrow',0):7d} {c.get('vector_shuffle',0):5d} "
                  f"{c.get('vector_move',0):5d} "
                  f"{c.get('scalar_fp',0)+c.get('scalar_fp_mem',0):9d}  {name[:52]}")
            shown += 1
            if shown >= 20:
                print("  ...")
                break

    w4 = sum(c.get("vector_wide4", 0) for _, _, c in rows)
    w2 = sum(c.get("vector_wide2", 0) for _, _, c in rows)
    nar = sum(c.get("vector_narrow", 0) for _, _, c in rows)
    shuf = sum(c.get("vector_shuffle", 0) for _, _, c in rows)
    mov = sum(c.get("vector_move", 0) for _, _, c in rows)
    sf = sum(c.get("scalar_fp", 0) + c.get("scalar_fp_mem", 0) for _, _, c in rows)
    zero = sum(1 for _, _, c in rows if wide(c) == 0)
    print(f"{len(rows)} bodies matching {args.needle!r} "
          f"(>= {args.min_instructions} instructions)")
    print(f"  vector arithmetic: 4-wide {w4}, 2-wide {w2}")
    print(f"  not arithmetic   : single-lane {nar}, shuffle {shuf}, move {mov}")
    print(f"  scalar fp        : {sf}")
    denom = w4 + w2 + sf
    if denom:
        print(f"  share of fp work that is WIDE vector: "
              f"{100.0 * (w4 + w2) / denom:.1f}%  (4-wide alone: "
              f"{100.0 * w4 / denom:.1f}%)")
    print(f"  bodies with ZERO wide vector arithmetic: {zero} of {len(rows)}")
    if sf == 0 and w4 + w2 == 0:
        print("refusing: no floating-point work of either kind was found — "
              "the needle is probably wrong")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
