# The platform contract: what a host implementation of lights must provide

BURNDOWN L-187, cluster `platform-host`, written 2026-09-29. This is the contract the first
host (L-186, macOS) is built against. It is also the contract a Windows or Linux
implementation will have to meet if someone ever asks for one. It is **not** an
abstraction. It says what a host must do and what bytes lights hands it, and it predicts
nothing about how a second platform will do it.

`tests/platform_contract.rs` works out every `frame.` line below from what the crate
actually does, and it fails if this document says anything different. It also requires the
`deferred.` and `withdrawn.` lines to stay, with exactly these keys. Change a line only in
the same commit as the code or the ruling that moved it.

## Two assumptions this entry carried, both withdrawn

    withdrawn.dependency_audit: a Linux and Windows dependency audit, filed for a portable host abstraction whose shared part could be listed in advance; the PM ruled on 2026-09-29 that each platform gets an implementation written from scratch, so a list per platform would be a prediction and this document carries none
    withdrawn.optical_contract: the contract as what optical requires, measured from optical's library; withdrawn the same day by the mission ruling (ad3663a8), under which lights is not the renderer under optical and optical requires nothing of it

The second withdrawal shows which way the tree moved, so here is the list it retired. L-187
used to require these eleven services, all measured from optical's library. Only three
still stand, and only in a smaller form:

| optical required | in this contract | why |
|---|---|---|
| marks | no | a widget library's drawing vocabulary; the demo draws scenes |
| a text measure and draw | no: L-197 chose the host's own text (2026-09-29) | see *The caption and the frame time* below |
| a layout allocation | no | no widgets |
| an interaction and hit test | no | raw pointer events are enough for every scene |
| an identity and persistent state | no | no widgets |
| pointer and key events with modifiers | **yes, without modifiers** | orbit is a drag; the selector is a key or a click |
| a cursor | no | no scene changes the cursor |
| a clipboard | no | no scene copies or pastes |
| an input method | no | no scene takes text |
| a repaint request | **yes, as a new frame after input** | orbit under the pointer |
| a surface with a scale factor | **yes** | a frame is rendered in pixels, and its time must be the time of the frame shown |

## The only consumer: the interactive demo

The contract covers what L-197 needs and nothing more. The PM scoped L-197's close bar on
2026-09-29 as: *a window that opens, three scenes, a selector, a caption per scene, orbit
under the pointer, and a frame time on screen*. The three scenes are orbit (drag to turn),
method comparison (splat against ray-cast on one scene) and surfel lighting (the 900-surfel
shell with a movable light). **A service joins this contract when a scene needs it, and
not before.** Every row below names the part of the close bar that requires it.

## Where a host lives

Not in lights. L-121 (PM, 2026-09-16) keeps lights headless: a render is a pure function
of a scene, and that is why it is bit-identical across thread counts.
`tests/layering.rs::lights_takes_no_window_gpu_input_or_frame_clock_dependency` rejects any
window, GPU, input or frame-clock dependency by name. The host is a separate crate that
depends on lights, and that part is settled. How lights itself is cut is a different
question, and it is deferred (below).

**Where platform code lives: in the host crate, and nowhere in lights (L-193, 2026-09-29).**
`tests/layering.rs::platform_code_lives_in_the_host_crate_and_never_in_lights` reads every
Rust file of lights and its manifest. It refuses a binding to an Apple framework or to the
Objective-C runtime, a selector or message send, an architecture intrinsic, inline
assembly, a CPU feature probe or gate, a `cfg` on a target, an OS-only module of std, and
a `[target.]` table. It proves it can see all of that by finding it in `crates/lights-mac`,
and every dependency the host takes that lights does not must be one it refuses. That
second check is the gap this entry found, kept as a test. The L-121 list names `metal` but
none of the five `objc2` crates the host really takes, and with `objc2-app-kit`, a window
binding, planted in lights' manifest, the L-121 guard stayed green.

Measured on 2026-09-29, lights holds none of it. Its only OS-only lines are three hardware
labels in report code, in `src/bin/lights_bench.rs`, `src/bin/lights_ui_frame.rs` and
`examples/renderer_arms.rs`, which run `sysctl` and `sw_vers`. The guard lists them by file,
so a port finds every one of them in one place. An architecture intrinsic has no home in
lights. It cannot land without a `cfg` arm for a platform that does not exist, or else a
core that stops compiling off aarch64, and the portable arithmetic is the definition of
correct that `tests/determinism.rs` pins, so a first one needs a PM ruling. L-193 adds no
trait, no `cfg` arm, no backend name and no directory. The directory it first proposed,
`crates/lights/src/mac/`, was filed six hours before L-186 gave the host a crate of its own.
Outside the scan, because it is tooling and not the crate's code, `scripts/glyph_oracle.py`
draws its oracle glyphs through CoreText and Quartz; what it drew is committed as fixtures.

## The three services

| service | the minimum | required by |
|---|---|---|
| **a window** | one window that opens with a size and a title, and ends the process when closed | "a window that opens" |
| | its size in pixels when it opens and again after every resize, before the next frame | every scene renders at the surface's pixel size |
| | the scale factor between the host's pointer coordinates and those pixels | the frame time must be for the frame actually shown; a scene that reads a position rather than a delta (a light placed under the pointer) maps it through this |
| **an event stream a hand can drive** | pointer press, motion and release, each with a position | orbit: "drag to turn"; the surfel scene's movable light |
| | a key press that says which key | the selector, if it is keyed |
| | close and resize, as events | the window row above |
| **a way to present a frame lights rendered** | take a finished frame in the byte layout below and show it at the surface's pixel size | every scene |
| | a new frame after any event that changes what is shown, and after a resize | "orbit under the pointer" |

The host does **not** provide a clock. The frame time on screen is measured by the demo
with `std::time::Instant` around its own render. That is standard library, not a
frame-clock dependency, and it needs nothing from the platform.

A resize rebuilds the `Renderer`. `Renderer::new` takes the size, and nothing resizes one
in place, so the framebuffer is allocated again at the new size.

### Not in the contract until a scene needs it

Modifiers, cursor shapes, the clipboard, an input method, scroll or zoom, touch and
gestures, more than one window, and redrawing with no input. L-186's close bar was written
against optical's six services. It still asks for modifiers, a cursor and a clipboard
round trip. This contract does not require them, and L-186 was amended to say so on the
same day (2026-09-29).

### The caption and the frame time: a fork that L-197 decides

Both have to appear on screen, and a host can cover either branch:

- **lights draws them into the frame.** The host provides nothing more. lights then needs
  text, which is the largest unbuilt piece (L-183, L-184, both blocked on this choice).
- **the host shows them.** The host provides one line of text it owns. The smallest is
  the window title, which every desktop platform's window has.

Neither branch is assumed here. L-197 records this as open "to be decided with evidence".

**DECIDED by L-197 (2026-09-29): the host shows them.** Measured from the tree that day:
lights has no text path at all. `DrawShape` has no text variant, no module rasterises a
glyph from a font, and `ttf-parser` is a dev-dependency, so no shipped build can read one
(L-183, L-184). The host's AppKit shows text for 152 lines in `crates/lights-mac`
(`src/bin/gallery/chrome.rs`) and no new crate. The caption and the frame time are
`NSTextField`s in a 92 pt strip above the render area, and the selector is an
`NSSegmentedControl` beside the keys 1 to 3. The chrome belongs to the gallery binary and
is not a host API, so this contract gains no service from it, and another platform's
gallery writes its own. The window title was the smallest branch, and it was not enough:
a caption of three lines does not fit in it. The reasons, and what is not verified, are in
`crates/lights-mac/src/bin/gallery/chrome_rs_comment/`.

## The frame handoff, in bytes and strides

This is the only part of the contract that can be written without naming a platform API:

    frame.bytes_per_pixel: 3
    frame.channel_order: R G B
    frame.alpha: none, resolve composites over the background it is given
    frame.row_stride: 3 x width, no padding
    frame.row_0: the edge CameraSpec::up points at
    frame.transfer: none, the bytes are linear-coded
    frame.quantisation: truncation, floor(255 x clamp(v, 0, 1))
    frame.float_layout: none, Vec3 is repr(Rust)

What those lines describe, and where they come from:

- **The one byte-level handoff is `Image`.** A warm path produces it this way:
  `Renderer::render_frame`, then `Renderer::resolve(background)`, which returns a
  `FloatImage<T>`, then `FloatImage::to_clamped_u8()`, which returns an `Image`.
  `Framebuffer::resolve(background)` goes straight to an `Image`. `Image::pixels` is a
  `Vec<[u8; 3]>`, so `image.pixels.as_flattened()` is the frame as one borrowed `&[u8]`,
  with no copy. Pixel (x, y) starts at byte `3 * (y * width + x)`. `Image::as_bytes()`
  returns the same bytes, copied.
- **Row 0 is the top.** The camera frame is x right, y **down**, z forward
  (`src/camera_rs_comment/conventions.md`), so the edge that `CameraSpec::up` points at is
  row 0. A host that flips rows shows the scene upside down.
- **Linear-coded, truncated.** No transfer function is applied anywhere on lights' path,
  and the caller owns the transfer (L-109, `src/image_rs_comment/colour.md`). A display
  that treats the bytes as sRGB-encoded shows them darker than they were lit. The demo
  page's PNGs are the same linear-coded bytes, so a host that presents them unchanged
  matches the page. Encoding for the display is the host's choice, and what it costs is
  not measured.
- **No alpha in this form.** The frame is opaque over the background passed to `resolve`.
  ~~`[u8; 4]` appears nowhere in `src/image.rs`. The expansion a GPU surface needs is
  unwritten and untimed (L-150).~~ WITHDRAWN 2026-09-29: L-186 wrote the expansion in the
  host (`crates/lights-mac`), and L-190 wrote the four-byte form into lights itself as a
  target the host owns, `Framebuffer::resolve_bgra8`, pinned below under macOS. Both are
  timed in `l190-fused-bgra8-handoff` (`docs/lights/RESEARCH.json`).
- **This handoff allocates.** `resolve` returns a new `FloatImage` and `to_clamped_u8` a
  new `Image`, 6,220,800 bytes at 1920x1080. The pack is a serial iterator:
  7.444 ms median in release at 1080p when L-150 measured it on 2026-09-16
  (`l150-cpu-pack-term` in `docs/lights/RESEARCH.json`). Its own unchanged command read
  4.643-4.658 ms on 2026-09-29, and the record's `unreproducible` audit note carries the
  pair (L-190). ~~**lights cannot fill a buffer the host owns.**~~
  WITHDRAWN 2026-09-29 by L-190: `Framebuffer::resolve_bgra8` resolves straight into a
  borrowed byte slice at the caller's row stride and allocates nothing. It is the macOS
  target below, and it is not a second name for this form: its bytes are four per pixel.
- **The float handoff has no byte layout.** `FloatImage<T>::pixels` is `Vec<Rgb<T>>`, and
  `Rgb<T>` is `Vec3<T>`, which carries no `repr` attribute. The language therefore does not
  fix its field order in memory, and a host reads it through `.x`, `.y` and `.z` or `get`,
  never through a pointer cast. The same holds for the premultiplied pair from
  `Renderer::resolve_premultiplied()`. Its coverage is one scalar stored in three channels
  (L-177).
- **Deterministic.** For one scene, camera, settings and precision, the frame is
  bit-identical at 1, 2, 4 and 18 threads (`tests/determinism.rs::thread_count_bit_identical`).
  A host can check its present path against a headless render of the same frame.

## Per platform: the natural native stack, as a note

These notes name each platform's own stack. They are not a plan and not a dependency list.
Which bytes a platform wants is found by writing its implementation, and nothing below
predicts it.

**macOS, the first implementation (L-186), BUILT as `crates/lights-mac`.** AppKit owns the window and the event stream:
a window and its view, the event objects for pointer and keys, and the backing scale factor.
Core Animation's `CAMetalLayer` presents one drawable per frame at the display's refresh,
and Metal owns the texture. OpenGL was named and withdrawn: Apple deprecated it in 2018, it
stops at 4.1, and it is not a first-class path on Apple Silicon (L-186). Memory is unified,
so a buffer in shared storage can be written by the CPU, which is what L-190's fused path
uses. Metal has no three-byte pixel format, so an `Image` must be expanded before it is
shown; that expansion was L-150's missing term, and `Host::present(&Image)` performs it.
`Host::present_filled` skips it: lights resolves straight into the host's buffer, below.
Measured on this machine with `sysctl` on
2026-09-29: 6 performance cores (`hw.perflevel0`), 12 efficiency cores (`hw.perflevel1`),
16384-byte pages and 128-byte cache lines. So a render pool of 18 threads
(`SharedSettings::threads`) puts two thirds of every frame on efficiency cores (L-190).
L-190 owns the macOS byte format, and it is **not** portable.

### The macOS byte target (L-190), not portable

`Framebuffer::resolve_bgra8(background, bytes, stride)` resolves the frame over `background`
into a byte slice the caller owns. On macOS the caller is `crates/lights-mac`, and the slice
is the contents of its shared-storage `MTLBuffer`, which is blitted into the drawable. The
layer's pixel format is `BGRA8Unorm`. This test-pinned table is what lands in that slice:

    bgra8.bytes_per_pixel: 4
    bgra8.channel_order: B G R A
    bgra8.alpha: 255, the frame is opaque over the background resolve is given
    bgra8.row_stride: the caller's, at least 4 x width
    bgra8.padding: bytes past 4 x width in a row are left as they were
    bgra8.colour: the frame's R G B bytes, byte for byte

- **It is the frame, reordered.** Every B, G and R byte equals the byte `resolve` and
  `to_clamped_u8` would have produced for that pixel, so the frame table's row 0, transfer
  and quantisation hold here unchanged. It is proven bit-identical in every blend mode, at
  both precisions and for every built method by `tests/handoff_pack.rs`.
- **Opaque, so premultiplied trivially.** Alpha is 255 because the frame is composited over
  its background before it is quantised. A translucent target, colour premultiplied and
  coverage in A, is not built: nothing asks for it.
- **The caller's stride, and the padding is not touched.** A row starts at `y x stride`.
  A stride under `4 x width`, or a slice too short for `stride x (height - 1) + 4 x width`
  bytes, is refused as `LightsError::ByteTarget`, and a refused slice is not written.
- **Nothing is allocated, and nothing is aligned by lights.** The buffer belongs to the host
  and is kept across frames. L-190 first proposed that lights own a page-aligned
  allocation for the backend to wrap with a no-copy buffer. WITHDRAWN 2026-09-29: the
  host's own shared buffer is already written in place, so a wrapped allocation would save
  no copy. It would only add an aligned allocator to lights and a deallocator to the host.
- **Named for its bytes, not for a platform.** Whether Windows or Linux want these bytes is
  found by writing their implementations (the deferral below). The name says what lands in
  memory, and no trait or format enum is built around it.

**Windows. Not built, and not built until someone asks.** Win32 owns the window and the
event stream: a window class and its message loop, pointer and key messages, and per-monitor
DPI awareness for the scale factor. A CPU frame reaches the screen through a DXGI flip-model
swap chain under Direct3D 11 or 12, or through GDI's device-independent bitmaps for the
smallest path.

**Linux. Not built, and not built until someone asks.** There are two display systems.
Under Wayland, the compositor owns placement: xdg-shell gives the window, the seat gives the
pointer and keyboard, a shared-memory buffer carries a CPU frame, and the fractional-scale
protocol gives the scale factor. Under X11, the window comes from Xlib or XCB, and MIT-SHM
carries a shared-memory image. On either one, Vulkan or EGL is the GPU present path.

## The deferral

    deferred.crate_split: whether lights is one crate or three is not open, and is not decided until a second implementation exists (PM, 2026-09-29)
    deferred.porting: demand-driven; a port is built when someone asks for one, and not before (PM, 2026-09-29)
    deferred.shared_part: isolated from two working implementations, never guessed from one (PM, 2026-09-29)

Until one of these reopens, a context does **not** do any of the following: design a
host trait, name a platform backend, add a `cfg` arm for Windows or Linux, split lights
into crates, or write a dependency list for a platform nobody asked for. Two things reopen
the question: a second implementation exists, or someone asks for a port. When one does,
the shared part is read off the two working implementations. It is not recovered from this
document.

## What this contract does not settle

- **Which transfer the host presents**, linear bytes as the demo page shows them or
  sRGB-encoded, and what an encode pass costs at 1080p. Unmeasured.
- ~~**Keys or a drawn control for the selector, and who draws the caption.** Both are
  L-197's.~~ SETTLED by L-197 (2026-09-29): both keys and a control, the keys 1 to 3 and
  an AppKit segmented control that the gallery reads back after every pump. The host's
  AppKit draws the caption and the frame time, and lights draws only the scene.
- ~~**AppKit directly or winit.**~~ SETTLED by L-186 (2026-09-29): AppKit directly, in
  `crates/lights-mac`. Both arms were written and driven by the same synthesized input.
  winit 0.30 drops any window event raised outside its own callback, and it would add 18
  crates on top of the 7 the Metal present path needs anyway. It saves about 100 lines of
  event code. The numbers are `l186-event-path-arms` in `docs/lights/RESEARCH.json`.
- ~~**The macOS byte format and the zero-copy target.** Both are L-190's.~~ SETTLED by
  L-190 (2026-09-29): the `bgra8.` table above, written into the host's shared buffer.
- **The upload and present cost.** Never measured in this crate (L-69, L-150's
  not-settled list).
- **Anything about Windows or Linux beyond naming their stacks.**

## Re-checking

    cargo test -p lights --all-features --test platform_contract
    sysctl -n hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.pagesize hw.cachelinesize
    grep -n "pub struct Image" -A4 crates/lights/src/image.rs
    grep -n "pub struct Vec3" -B2 crates/lights/src/numerics/vector.rs
    grep -n "fn lights_takes_no_window_gpu_input_or_frame_clock_dependency" crates/lights/tests/layering.rs
    cargo test -p lights --all-features --test layering platform_code_lives_in_the_host_crate_and_never_in_lights
