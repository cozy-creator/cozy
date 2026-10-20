"""Build-only TensorFS decode cores, moved from Runtime xs049/adapters.

Source: Runtime 326e72c20ab5c4be110739ab9bd23783223825fb, decode_cores/decode_schemes.

Schemes and validation live in Rust's checked Plan; these kernels execute that plan.
No framework is imported, and deployed decoding loads the emitted CUDA binaries.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _word(words, at, mask, WORD):
    at = at * (WORD // 8)
    value = tl.load(words + at, mask=mask, other=0).to(tl.uint32)
    if WORD >= 16:
        value = value | (tl.load(words + at + 1, mask=mask, other=0).to(tl.uint32) << 8)
    if WORD == 32:
        value = value | (tl.load(words + at + 2, mask=mask, other=0).to(tl.uint32) << 16)
        value = value | (tl.load(words + at + 3, mask=mask, other=0).to(tl.uint32) << 24)
    return value


@triton.jit
def _stored(e, params):
    cols = tl.load(params + 2)
    height = tl.load(params + 3)
    if tl.load(params + 6):
        matrix = height * cols
        e = (e // matrix) * matrix + (e % height) * cols + (e % matrix) // height
    return e


@triton.jit
def _logical(e, params):
    cols = tl.load(params + 2)
    height = tl.load(params + 3)
    if tl.load(params + 6):
        matrix = height * cols
        e = (e // matrix) * matrix + (e % cols) * height + (e % matrix) // cols
    return e


@triton.jit
def _grid(i, j, index, params, valid):
    rows, cols, height = tl.load(params + 1), tl.load(params + 2), tl.load(params + 3)
    kind, size, layout = tl.load(params + 17), tl.load(params + 18), tl.load(params + 19)
    if kind == 0:
        at = i * 0
    elif kind == 1:
        at = i
    elif kind == 2:
        at = j
    elif kind == 3:
        width = (cols + size - 1) // size
        if tl.load(params + 20):
            block = tl.load(index + j, mask=valid, other=0).to(tl.int64)
        else:
            block = j // size
        if layout == 0:
            at = i * width + block
        elif layout == 1:
            at = block * rows + i
        else:
            matrix, row = i // height, i % height
            tiles = (width + 3) // 4
            span = ((height + 127) // 128) * 128 * tiles * 4
            at = matrix * span + ((row // 128) * tiles + block // 4) * 512
            at = at + (row % 32) * 16 + (row % 128 // 32) * 4 + block % 4
    else:
        br, bc = tl.load(params + 21), tl.load(params + 22)
        at = ((i // height) * ((height + br - 1) // br) + (i % height) // br)
        at = at * ((cols + bc - 1) // bc) + j // bc
    return at


@triton.jit
def _unpack(words, slots, e, params, valid):
    rows, cols = tl.load(params + 1), tl.load(params + 2)
    i, j = e // cols, e % cols
    if tl.load(params + 7):
        i, j, rows, cols = e * 0, e, rows * 0 + 1, rows * cols
    return _code(words, slots, i, j, rows, cols, tl.load(params + 16),
                 tl.load(params + 15).to(tl.int32), valid,
                 tl.load(params + 8).to(tl.int32), tl.load(params + 9).to(tl.int32), tl.load(params + 10).to(tl.int32),
                 tl.load(params + 11).to(tl.int32), tl.load(params + 12).to(tl.int32), tl.load(params + 13).to(tl.int32), tl.load(params + 14).to(tl.int32))


@triton.jit
def _value(q, e, scale, term, table, factor, post, index, error, params, valid):
    cols, height = tl.load(params + 2), tl.load(params + 3)
    i, j = e // cols, e % cols
    n = tl.load(params + 24)
    if n:
        bad = valid & ((q < 0) | (q >= n))
        tl.atomic_or(error + q * 0, 1, mask=bad, sem="relaxed")
        value = tl.load(table + q, mask=valid & ~bad, other=0.0)
    else:
        value = q.to(tl.float32)
    at = _grid(i, j, index, params, valid)
    s = tl.load(scale + at, mask=valid, other=0.0)
    mode = tl.load(params + 25)
    g = tl.full(s.shape, 1.0, tl.float32)
    if mode:
        g_at = tl.where(tl.load(params + 26) == 1, 0, i // height)
        g = tl.load(factor + g_at, mask=valid, other=1.0)
        if mode == 2:
            s = tl.math.div_rn(s, g)
    kind = tl.load(params + 23)
    if kind == 1:
        value = (value - tl.load(term + at, mask=valid, other=0.0)) * s
    elif kind == 2:
        value = value * s + tl.load(term + at, mask=valid, other=0.0)
    else:
        value = value * s
    if mode == 1:
        value = value * g
    return _by_post(post, j, value, valid, tl.load(params + 27))


@triton.jit
def codes(out, words, slots, scale, term, table, factor, post, index, error, params,
          OUT: tl.constexpr, BLOCK: tl.constexpr):
    begin, count = tl.load(params + 4), tl.load(params + 5)
    dst = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = dst < count
    e = _stored(begin + dst, params)
    q = _unpack(words, slots, e, params, valid)
    v = _value(q, e, scale, term, table, factor, post, index, error, params, valid)
    _store(out, dst, v, valid, OUT)


@triton.jit
def structs(out, words, slots, scale, term, table, factor, post, index, error, params,
            DECODE: tl.constexpr, OUT: tl.constexpr, BLOCK: tl.constexpr):
    begin, count = tl.load(params + 4), tl.load(params + 5)
    dst = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = dst < count
    e = _stored(begin + dst, params)
    n, size = tl.load(params + 29), tl.load(params + 30)
    v = DECODE(words + (e // n) * size, e % n, valid)
    v = _by_post(post, e % tl.load(params + 2), v, valid, tl.load(params + 27))
    _store(out, dst, v, valid, OUT)


@triton.jit
def unpack(out, words, slots, scale, term, table, factor, post, index, error, params,
           OUT: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = e < tl.load(params)
    tl.store(out + e, _unpack(words, slots, e, params, valid), mask=valid)


@triton.jit
def requantize(out, words, slots, scale, term, table, factor, post, index, error, params,
               OUT: tl.constexpr, BLOCK: tl.constexpr):
    """Stored4/6-bit levels -> relative-scaled, ties-even/saturated exact int8 values."""
    e = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = e < tl.load(params)
    cols, bits, group = tl.load(params + 2), tl.load(params + 8), tl.load(params + 18)
    i, j = e // cols, e % cols
    width = tl.where(bits == 4, cols // 2, (cols // 4) * 3)
    low = (_word(words, i * width + j // 2, valid, 8) >> (4 * (j % 2)).to(tl.uint32)) & 15
    q = low
    if bits == 6:
        high = (_word(words, i * width + cols // 2 + j // 4, valid, 8)
                >> (2 * (j % 4)).to(tl.uint32)) & 3
        q = low | (high << 4)
    if tl.load(params + 24):
        value = tl.load(table + q, mask=valid, other=0.0)
    else:
        value = q.to(tl.float32) - (1 << (bits - 1)).to(tl.float32)
    relative = tl.load(scale + i * (cols // group) + j // group, mask=valid, other=0.0)
    value = value * relative
    bad = valid & (value != value)
    tl.atomic_or(error + e * 0, 2, mask=bad, sem="relaxed")
    rounded = libdevice.nearbyint(tl.where(bad, 0.0, value))
    q8 = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int32)
    tl.store(out + e, q8, mask=valid)


@triton.jit
def hadamard(out, words, slots, scale, term, table, factor, post, index, error, params,
             OUT: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = e < tl.load(params)
    stride = tl.load(params + 31)
    base = e // (4 * stride) * (4 * stride) + e % stride
    a = _word(words, base, valid, 32).to(tl.int32, bitcast=True)
    b = _word(words, base + stride, valid, 32).to(tl.int32, bitcast=True)
    c = _word(words, base + 2 * stride, valid, 32).to(tl.int32, bitcast=True)
    d = _word(words, base + 3 * stride, valid, 32).to(tl.int32, bitcast=True)
    opposite = _word(words, base + (3 - (e // stride) % 4) * stride, valid, 32).to(tl.int32, bitcast=True)
    tl.store(out + e, a + b + c + d - 2 * opposite, mask=valid)


@triton.jit
def rotated(out, words, slots, scale, term, table, factor, post, index, error, params,
            OUT: tl.constexpr, BLOCK: tl.constexpr):
    begin, count = tl.load(params + 4), tl.load(params + 5)
    dst = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = dst < count
    e = _stored(begin + dst, params)
    q = _word(words, e, valid, 32).to(tl.int32, bitcast=True)
    v = _value(q, e, scale, term, table, factor, post, index, error, params, valid)
    _store(out, dst, v, valid, OUT)


@triton.jit
def _hadamard_prefix(n, j, block, root):
    """Exact integer sum H_B[0:n,j] in log4(B), with no logical host expansion."""
    span, weight = block, root.to(tl.int32)
    sign = tl.full(n.shape, 1, tl.int32)
    total = tl.full(n.shape, 0, tl.int32)
    left = n
    while span > 1:
        span = span // 4
        weight = weight // 2
        a, b = left // span, (j // span) % 4
        total = total + sign * (a.to(tl.int32) - 2 * ((3 - b) < a).to(tl.int32)) * weight
        sign = tl.where(a + b == 3, -sign, sign)
        left = left % span
    return tl.where(n == block, root.to(tl.int32), total)


@triton.jit
def grouped_rotated(out, words, slots, scale, term, table, factor, post, index, error, params,
                    OUT: tl.constexpr, BLOCK: tl.constexpr):
    """Integer Hadamard result x channel scale, ordered correction, then unit/round_L."""
    begin, count = tl.load(params + 4), tl.load(params + 5)
    dst = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = dst < count
    e = _stored(begin + dst, params)
    cols, group, block = tl.load(params + 2), tl.load(params + 18), tl.load(params + 28)
    row, col = e // cols, e % cols
    q = _word(words, e, valid, 32).to(tl.int32, bitcast=True)
    value = q.to(tl.float32) * tl.load(factor + row, mask=valid, other=0.0)
    if tl.load(params + 29):
        base = col // block * block
        first, last = base // group, (base + block + group - 1) // group
        z = tl.full((BLOCK,), 0.0, tl.float32)
        k = 0
        while k < (block + group - 1) // group + 1:
            g = first + k
            active = valid & (g < last)
            lo = tl.minimum(tl.maximum(g * group - base, 0), block)
            hi = tl.minimum(tl.maximum((g + 1) * group - base, 0), block)
            root = tl.load(params + 30)
            m = _hadamard_prefix(hi, col % block, block, root) - _hadamard_prefix(lo, col % block, block, root)
            correction = tl.load(term + g * tl.load(params + 1) + row, mask=active, other=0.0)
            part = correction * m.to(tl.float32)
            # Do not add a synthetic leading/trailing zero: order and signed zero matter.
            z = tl.where(active, tl.where(k == 0, part, z + part), z)
            k += 1
        value = value + z
    value = _by_post(post, col, value, valid, tl.load(params + 27))
    _store(out, dst, value, valid, OUT)


@triton.jit  # type: ignore[untyped-decorator]
def _code(  # type: ignore[no-untyped-def]
    words,
    slots,
    i,
    j,
    rows,
    cols,
    nw,
    offset,
    valid,
    BITS,
    WORD,
    TRANSPOSED,
    AXIS,
    FILL,
    ORDER,
    SIGNED,
):
    """Code q[i, j] of the view (spec §6 bit packing)."""
    if TRANSPOSED:
        x = j
        y = i
        nb = rows
    else:
        x = i
        y = j
        nb = cols
    if AXIS == 0:
        t = x
        lane = y
    else:
        t = y
        lane = x
    PER = WORD // BITS
    MASK = (1 << BITS) - 1
    if FILL == 1:
        bit = t * BITS
        k = bit // WORD
        off = (bit % WORD).to(tl.uint32)
        if AXIS == 0:
            at = k * nb + lane
            nxt = at + nb
        else:
            at = lane * nw + k
            nxt = at + 1
        spill = valid & (off + BITS > WORD)
        lo = _word(words, at, valid, WORD) >> off
        hi = _word(words, nxt, spill, WORD) << tl.where(spill, WORD - off, 0)
        raw = (lo | hi) & MASK
    else:
        if FILL == 0:
            k = t // PER
            e = t % PER
        else:
            k = t % nw
            e = t // nw
        if ORDER == 0:
            slot = e
        elif ORDER == 1:
            slot = PER - 1 - e
        else:
            slot = tl.load(slots + e, mask=valid, other=0).to(tl.int64)
        if AXIS == 0:  # noqa: SIM108 - a constexpr branch, not a select
            at = k * nb + lane
        else:
            at = lane * nw + k
        raw = (_word(words, at, valid, WORD) >> (slot * BITS).to(tl.uint32)) & MASK
    q = raw.to(tl.int32)
    if SIGNED:  # noqa: SIM108 - a constexpr branch, not a select
        q = q - ((q >> (BITS - 1)) << BITS)
    else:
        q = q - offset
    return q

@triton.jit  # type: ignore[untyped-decorator]
def _by_post(post, j, v, valid, POST):  # type: ignore[no-untyped-def]
    """`v x post`: one factor (1), one per column (2), or none (0)."""
    if POST == 1:
        v = v * tl.load(post)
    elif POST == 2:
        v = v * tl.load(post + j, mask=valid, other=1.0)
    return v

@triton.jit  # type: ignore[untyped-decorator]
def _store(out, e, v, valid, OUT: tl.constexpr):  # type: ignore[no-untyped-def]
    """round_L into the destination's bits: nearest even, NaN canonical."""
    nan = v != v
    if OUT == 0:
        bits = tl.where(nan, 0x7FC00000, v.to(tl.int32, bitcast=True))
    elif OUT == 1:
        b = v.to(tl.uint32, bitcast=True)
        r = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
        bits = tl.where(nan, 0x7FC0, r).to(tl.uint16).to(tl.int16, bitcast=True)
    else:
        bits = tl.where(nan, 0x7E00, v.to(tl.float16).to(tl.int16, bitcast=True))
    tl.store(out + e, bits, mask=valid)

@triton.jit
def u8_at(base, at, mask):  # type: ignore[no-untyped-def]
    """A struct's byte at `at`, as int32."""
    return tl.load(base + at, mask=mask, other=0).to(tl.int32)

@triton.jit
def i8_at(base, at, mask):  # type: ignore[no-untyped-def]
    return tl.load(base + at, mask=mask, other=0).to(tl.int8, bitcast=True).to(tl.int32)

@triton.jit
def f16_at(base, at, mask):  # type: ignore[no-untyped-def]
    """A struct's little-endian F16 at byte `at`, as f32 (exact)."""
    lo = tl.load(base + at, mask=mask, other=0).to(tl.uint16)
    hi = tl.load(base + at + 1, mask=mask, other=0).to(tl.uint16)
    return (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

@triton.jit
def _u32(base, at, mask):  # type: ignore[no-untyped-def]
    b0 = u8_at(base, at, mask)
    b1 = u8_at(base, at + 1, mask)
    b2 = u8_at(base, at + 2, mask)
    b3 = u8_at(base, at + 3, mask)
    return (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).to(tl.uint32, bitcast=True)

@triton.jit
def _pow2(x):  # type: ignore[no-untyped-def]
    """2^x exactly, x in -149..127."""
    bits = tl.where(
        x >= -126, (x + 127) << 23, 1 << (tl.minimum(tl.maximum(x, -149), -127) + 149)
    )
    return bits.to(tl.float32, bitcast=True)

@triton.jit
def _kvalue(q):  # type: ignore[no-untyped-def]
    """ggml's kvalues_fp4[q]: E2M1 doubled; code 8 is +0."""
    m = q & 7
    mag = tl.where(m < 4, m, (2 + (m & 1)) << tl.maximum((m >> 1) - 1, 0))
    return tl.where(q >= 8, -mag, mag).to(tl.float32)

@triton.jit
def _nibble(base, at, n, el, mask):  # type: ignore[no-untyped-def]
    """q4_0's order: element j of 2n in byte at + j % n, low nibble first half."""
    byte = u8_at(base, at + el % n, mask)
    return tl.where(el < n, byte & 15, byte >> 4)

@triton.jit
def _q4_0(base, el, mask):  # type: ignore[no-untyped-def]
    return (_nibble(base, 2, 16, el, mask) - 8).to(tl.float32) * f16_at(base, 0, mask)

@triton.jit
def _q4_1(base, el, mask):  # type: ignore[no-untyped-def]
    q = _nibble(base, 4, 16, el, mask).to(tl.float32)
    return q * f16_at(base, 0, mask) + f16_at(base, 2, mask)

@triton.jit
def _q5_0(base, el, mask):  # type: ignore[no-untyped-def]
    high = ((_u32(base, 2, mask) >> el.to(tl.uint32)) & 1).to(tl.int32)
    q = _nibble(base, 6, 16, el, mask) + 16 * high
    return (q - 16).to(tl.float32) * f16_at(base, 0, mask)

@triton.jit
def _q5_1(base, el, mask):  # type: ignore[no-untyped-def]
    high = ((_u32(base, 4, mask) >> el.to(tl.uint32)) & 1).to(tl.int32)
    q = (_nibble(base, 8, 16, el, mask) + 16 * high).to(tl.float32)
    return q * f16_at(base, 0, mask) + f16_at(base, 2, mask)

@triton.jit
def _q8_0(base, el, mask):  # type: ignore[no-untyped-def]
    return i8_at(base, 2 + el, mask).to(tl.float32) * f16_at(base, 0, mask)

@triton.jit
def _low2(base, at, el, mask):  # type: ignore[no-untyped-def]
    """q2_k/q3_k: element t's 2-bit code in qs at `at`."""
    h = el // 128
    j = (el % 128) // 32
    half = (el % 32) // 16
    return (u8_at(base, at + 32 * h + 16 * half + el % 16, mask) >> (2 * j)) & 3

@triton.jit
def _q2_k(base, el, mask):  # type: ignore[no-untyped-def]
    q = _low2(base, 16, el, mask).to(tl.float32)
    sc = u8_at(base, el // 16, mask)
    dl = f16_at(base, 80, mask) * (sc & 15).to(tl.float32)
    ml = f16_at(base, 82, mask) * (sc >> 4).to(tl.float32)
    return dl * q - ml

@triton.jit
def _q3_k(base, el, mask):  # type: ignore[no-untyped-def]
    q = _low2(base, 32, el, mask)
    at = 16 * ((el % 32) // 16) + el % 16
    high = (u8_at(base, at, mask) >> (el // 32)) & 1
    b = el // 16
    raw = u8_at(base, 96 + b % 8, mask)
    lo = tl.where(b < 8, raw & 15, raw >> 4)
    sc = (lo | (((u8_at(base, 104 + b % 4, mask) >> (2 * (b // 4))) & 3) << 4)) - 32
    dl = f16_at(base, 108, mask) * sc.to(tl.float32)
    return dl * (q - 4 + 4 * high).to(tl.float32)

@triton.jit  # type: ignore[untyped-decorator]
def _k_quant(base, el, mask, QS: tl.constexpr, QH: tl.constexpr):  # type: ignore[no-untyped-def]
    """q4_k (QH 0) and q5_k: 8 sub-blocks of 32, 6-bit scale and min (get_scale_min_k4)."""
    a = el // 32
    ai = a % 4
    x = u8_at(base, 4 + ai, mask)
    y = u8_at(base, 8 + ai, mask)
    z = u8_at(base, 12 + ai, mask)
    c = tl.where(a < 4, x & 63, (z & 15) | ((x >> 6) << 4))
    e = tl.where(a < 4, y & 63, (z >> 4) | ((y >> 6) << 4))
    byte = u8_at(base, QS + 32 * (a // 2) + el % 32, mask)
    q = tl.where(a % 2 == 0, byte & 15, byte >> 4)
    if QH:
        q = q + 16 * ((u8_at(base, 16 + el % 32, mask) >> a) & 1)
    dl = f16_at(base, 0, mask) * c.to(tl.float32)
    ml = f16_at(base, 2, mask) * e.to(tl.float32)
    return dl * q.to(tl.float32) - ml

@triton.jit
def _q4_k(base, el, mask):  # type: ignore[no-untyped-def]
    return _k_quant(base, el, mask, 16, 0)

@triton.jit
def _q5_k(base, el, mask):  # type: ignore[no-untyped-def]
    return _k_quant(base, el, mask, 48, 1)

@triton.jit
def _q6_k(base, el, mask):  # type: ignore[no-untyped-def]
    h = el // 128
    j = (el % 128) // 32
    lane = el % 32
    byte = u8_at(base, 64 * h + 32 * (j % 2) + lane, mask)
    low = tl.where(j < 2, byte & 15, byte >> 4)
    q = (low | (((u8_at(base, 128 + 32 * h + lane, mask) >> (2 * j)) & 3) << 4)) - 32
    sc = i8_at(base, 192 + 8 * h + lane // 16 + 2 * j, mask).to(tl.float32)
    return f16_at(base, 208, mask) * sc * q.to(tl.float32)

@triton.jit
def _mxfp4_struct(base, el, mask):  # type: ignore[no-untyped-def]
    d = _pow2(u8_at(base, 0, mask) - 128)
    return _kvalue(_nibble(base, 1, 16, el, mask)) * d

@triton.jit
def _ue4m3(b):  # type: ignore[no-untyped-def]
    """ggml's UE4M3: no sign, no NaN, 0x7F zero."""
    e = (b >> 3) & 15
    m = (b & 7).to(tl.float32)
    v = tl.where(e == 0, m * _pow2(e * 0 - 9), (m + 8.0) * _pow2(e - 10))
    return tl.where(b == 0x7F, 0.0, v)

@triton.jit
def _nvfp4_struct(base, el, mask):  # type: ignore[no-untyped-def]
    k = el // 16
    d = _ue4m3(u8_at(base, k, mask)) * 0.5
    return _kvalue(_nibble(base, 4 + 8 * k, 8, el % 16, mask)) * d

@triton.jit
def _q1_0(base, el, mask):  # type: ignore[no-untyped-def]
    bit = (u8_at(base, 2 + el // 8, mask) >> (el % 8)) & 1
    d = f16_at(base, 0, mask)
    return tl.where(bit == 1, d, -d)

@triton.jit
def _q2_0(base, el, mask):  # type: ignore[no-untyped-def]
    q = (u8_at(base, 2 + el // 4, mask) >> (2 * (el % 4))) & 3
    return (q - 1).to(tl.float32) * f16_at(base, 0, mask)


@triton.jit
def cast_metadata(out, words, slots, scale, term, table, factor, post, index, error, params,
                  OUT: tl.constexpr, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
    """Normalize only stored factor cells to f32; no logical weight expansion."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(params)
    mode = tl.load(params + 1)
    mask = i < n
    if mode == 0:
        lo = tl.load(words + 2 * i, mask, 0).to(tl.uint32)
        hi = tl.load(words + 2 * i + 1, mask, 0).to(tl.uint32)
        value = ((lo | (hi << 8)) << 16).to(tl.float32, bitcast=True)
    elif mode == 1:
        lo = tl.load(words + 2 * i, mask, 0).to(tl.uint16)
        hi = tl.load(words + 2 * i + 1, mask, 0).to(tl.uint16)
        value = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        code = tl.load(words + i, mask, 0).to(tl.int32)
        value = tl.load(table + code, mask, 0)
    tl.store(out + i, value, mask)
