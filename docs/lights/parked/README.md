# Parked work: written, correct, and waiting for the change it tests

A patch here is a regression test that was written TEST-FIRST by a burn-down context, seen
red as the close bar requires, and then left uncommitted when that context's budget ran
out. It is parked rather than deleted because the test is the expensive half: a test
written after a fix and never seen red is not evidence, so throwing it away would mean
paying for it twice.

Apply it as the FIRST act of the entry that owns it, confirm it is still red, then build
the change:

    git apply docs/lights/parked/<name>.patch

## l177_coverage_plane_test.patch

Owner: **L-177**. Adds `tests/image.rs::a_coverage_plane_holds_one_scalar_per_pixel_so_a_1080p_f32_layer_is_33_177_600_bytes`,
which asserts that a 1080p `RasterLayer<f32>` holds 33,177,600 bytes — one `Rgb<f32>`
colour plane plus a one-`f32`-per-pixel coverage plane — and counts the allocations of each
size that `from_coverage8` and `Framebuffer::resolve_premultiplied` make.

It fails on the tree as it stands, reading 49,766,400 bytes, because the coverage plane
stores `Rgb::splat(covered)`: three channels carrying one scalar, 8 of every 12 bytes
saying nothing. That is 15.82 MiB of a 47.46 MiB f32 frame at 1080p, and 63 MiB at a 2×
Retina surface.

It was parked on 2026-09-30 so the first public commit of this crate would pass its own
suite rather than ship a known red. Nothing about the test is wrong; the change it tests
has not been built. Blast radius measured the same day: 41 references to `.coverage` across
13 files, the heaviest being `tests/draw_list.rs` (9), `tests/interpenetration.rs` (6) and
`tests/primitive_pass.rs` (5).
