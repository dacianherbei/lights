#![cfg(feature = "method-splat")]
use lights::camera::CameraSpec;
use lights::dispatch::{Collect, Precision, RenderOutcome, RenderRequest, render};
use lights::image::FloatImage;
use lights::kernel::KernelFamily;
use lights::method::RenderMethod;
use lights::method::splat::{SortKind, SplatConfig};
use lights::numerics::quaternion::Quat;
use lights::numerics::vector::{Rgb, Vec3};
use lights::scene::{Atom, Scene};
use lights::settings::{MIN_SCALE, PixelConvention, SharedSettings};

const FLOOR: f64 = MIN_SCALE;
const WIDTH: u32 = 96;
const HEIGHT: u32 = 72;

fn camera() -> CameraSpec {
    CameraSpec::look_at(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        45.0,
        WIDTH,
        HEIGHT,
    )
    .with_intrinsics([600.0, 600.0], [48.0, 36.0])
    .with_convention(PixelConvention::CornerSample)
}

fn atoms() -> Vec<Atom<f64>> {
    let mut out = Vec::new();
    for i in 0..24u32 {
        let t = i as f64;
        let angle = 0.21 * t;
        let half = angle * 0.5;
        out.push(Atom::new(
            Vec3::new(
                0.35 * (0.7 * t).sin(),
                0.28 * (0.9 * t).cos(),
                3.0 + 0.17 * t,
            ),
            Quat::new_wxyz(half.cos(), 0.0, half.sin(), 0.0),
            Vec3::new(0.02 + 0.001 * t, 0.03, 0.045),
            0.35,
            Rgb::new(
                (0.13 * t).fract(),
                (0.29 * t).fract(),
                (0.47 * t).fract(),
            ),
        ));
    }
    out
}

fn scene_from(list: &[Atom<f64>]) -> Scene<f64> {
    let mut scene = Scene::<f64>::new();
    for atom in list {
        match scene.push(*atom, FLOOR) {
            Ok(()) => (),
            Err(e) => panic!("scene rejected an atom: {e:?}"),
        }
    }
    scene
}

fn render_with(scene: Scene<f64>, threads: usize, sort: SortKind) -> RenderOutcome {
    render_at(scene, threads, sort, Precision::F64)
}

fn render_at(
    scene: Scene<f64>,
    threads: usize,
    sort: SortKind,
    precision: Precision,
) -> RenderOutcome {
    let mut settings = SharedSettings::<f64>::contract();
    settings.threads = threads;
    settings.background = Rgb::new(0.05, 0.07, 0.11);
    let request = RenderRequest {
        precision,
        kernel: KernelFamily::Gaussian,
        method: RenderMethod::SplatRaster(SplatConfig {
            tile: 16,
            sort,
        }),
        settings,
        camera: camera(),
        scene,
        collect: Collect::Pixels,
        shapes: None,
    };
    match render(&request) {
        Ok(o) => o,
        Err(e) => panic!("render failed: {e:?}"),
    }
}

fn assert_bit_identical(a: &FloatImage<f64>, b: &FloatImage<f64>, what: &str) {
    assert_eq!((a.width, a.height), (b.width, b.height), "{what}: size");
    for (i, (left, right)) in a.pixels.iter().zip(b.pixels.iter()).enumerate() {
        assert_eq!(
            left.x.to_bits(),
            right.x.to_bits(),
            "{what}: pixel {i} red {} vs {}",
            left.x,
            right.x
        );
        assert_eq!(left.y.to_bits(), right.y.to_bits(), "{what}: pixel {i} green");
        assert_eq!(left.z.to_bits(), right.z.to_bits(), "{what}: pixel {i} blue");
    }
}

fn image_is_not_blank(image: &FloatImage<f64>) {
    let (peak, _, _) = image.max_abs();
    assert!(peak > 0.2, "the determinism scene must actually render");
    let (lo, hi) = image.min_max_channel();
    assert!(hi - lo > 0.1, "the image must have structure, not one flat colour");
}

#[test]
fn thread_count_bit_identical() {
    let list = atoms();
    let mut swept = 0usize;
    let mut first_of_each: Vec<FloatImage<f64>> = Vec::new();
    for precision in Precision::ALL {
        swept += 1;
        let name = precision.name();
        let single = render_at(scene_from(&list), 1, SortKind::GlobalSort, precision);
        first_of_each.push(single.image.clone());
        image_is_not_blank(&single.image);
        let again = render_at(scene_from(&list), 1, SortKind::GlobalSort, precision);
        assert_bit_identical(&single.image, &again.image, &format!("{name}: repeat at one thread"));

        for threads in [2usize, 4, 18] {
            let parallel = render_at(scene_from(&list), threads, SortKind::GlobalSort, precision);
            assert_bit_identical(
                &single.image,
                &parallel.image,
                &format!("{name}: {threads} threads against one"),
            );
            let repeat = render_at(scene_from(&list), threads, SortKind::GlobalSort, precision);
            assert_bit_identical(
                &parallel.image,
                &repeat.image,
                &format!("{name}: {threads} threads repeated"),
            );
        }
    }
    assert_eq!(
        swept,
        Precision::ALL.len(),
        "the precision axis must be swept, not assumed: f32 sums reorder far more \
         readily than f64, so a bit-identity claim made at f64 alone is the weaker \
         half of the axis"
    );
    assert!(swept > 1, "a sweep of one is not a sweep");
    assert_eq!(first_of_each.len(), Precision::ALL.len());
    assert!(
        differ(&first_of_each[0], &first_of_each[1]),
        "the two precisions produced BIT-IDENTICAL images, which means the \
         precision argument reached nothing and this sweep tested f64 twice. A \
         guard that passes by measuring the same thing under two names is the \
         failure this assertion exists to catch"
    );
}

fn differ(a: &FloatImage<f64>, b: &FloatImage<f64>) -> bool {
    if (a.width, a.height) != (b.width, b.height) {
        return true;
    }
    a.pixels
        .iter()
        .zip(b.pixels.iter())
        .any(|(l, r)| {
            l.x.to_bits() != r.x.to_bits()
                || l.y.to_bits() != r.y.to_bits()
                || l.z.to_bits() != r.z.to_bits()
        })
}

#[test]
fn input_permutation_bit_identical() {
    for precision in Precision::ALL {
        permutation_holds_at(precision);
    }
}

fn permutation_holds_at(precision: Precision) {
    let name = precision.name();
    let list = atoms();
    let reference = render_at(scene_from(&list), 4, SortKind::GlobalSort, precision);
    image_is_not_blank(&reference.image);

    let mut permutations = Vec::new();
    let mut reversed = list.clone();
    reversed.reverse();
    permutations.push(reversed);

    let mut strided = Vec::new();
    for start in 0..5usize {
        let mut i = start;
        while i < list.len() {
            strided.push(list[i]);
            i += 5;
        }
    }
    assert_eq!(strided.len(), list.len());
    permutations.push(strided);

    let mut rotated = list.clone();
    rotated.rotate_left(7);
    permutations.push(rotated);

    for (n, permuted) in permutations.iter().enumerate() {
        let outcome = render_at(scene_from(permuted), 4, SortKind::GlobalSort, precision);
        assert_bit_identical(
            &reference.image,
            &outcome.image,
            &format!("{name}: permutation {n}"),
        );
    }
}

#[test]
fn the_two_sort_kinds_produce_the_same_image() {
    let list = atoms();
    let global = render_with(scene_from(&list), 4, SortKind::GlobalSort);
    let per_tile = render_with(scene_from(&list), 4, SortKind::PerTileSort);
    assert_bit_identical(&global.image, &per_tile.image, "GlobalSort vs PerTileSort");
}

#[test]
fn equal_depth_order_follows_the_scene_index() {
    let depth = 5.0;
    let red = Atom::isotropic(Vec3::new(0.0, 0.0, depth), 0.05, 0.9, Rgb::new(1.0, 0.0, 0.0));
    let blue = Atom::isotropic(Vec3::new(0.0, 0.0, depth), 0.05, 0.9, Rgb::new(0.0, 0.0, 1.0));

    let red_first = render_with(scene_from(&[red, blue]), 1, SortKind::GlobalSort);
    let blue_first = render_with(scene_from(&[blue, red]), 1, SortKind::GlobalSort);

    let a = red_first.image.get(48, 36);
    let b = blue_first.image.get(48, 36);
    assert!(
        a.x > a.z,
        "the atom pushed first wins the tie: {a:?}"
    );
    assert!(b.z > b.x, "and the tie breaks the other way when the order flips: {b:?}");

    for threads in [1usize, 4, 18] {
        let repeat = render_with(scene_from(&[red, blue]), threads, SortKind::GlobalSort);
        assert_bit_identical(
            &red_first.image,
            &repeat.image,
            "an exact depth tie must still be deterministic",
        );
    }
}
