use crate::blend::{BlendMode, EarlyOut, Over, OverAccum, Step};
use crate::camera::Camera;
use crate::error::{LightsError, Result};
use crate::fit::atom2d::{Atom2d, to_scene};
use crate::fit::descent::P;
use crate::fit::greedy::{Context, residual_energy};
use crate::guard::Budget;
use crate::image::FloatImage;
use crate::kernel::gaussian::Gaussian;
use crate::kernel::{Kernel, KernelFamily};
use crate::method::splat::bin::depth_bits;
use crate::method::splat::project::{ScreenAtom, project_scene};
use crate::numerics::symmetric::{LowerTri2, Sym2, Sym3};
use crate::numerics::vector::{Rgb, Vec2, Vec3};
use crate::scene::prepare::PreparedScene;

pub const RENDERS_PER_ANALYTIC_GRADIENT: u64 = 1;

#[derive(Clone, Debug)]
pub struct Backward {
    pub loss: f64,
    pub gradient: [f64; P],
    pub image: FloatImage<f64>,
    pub visible: bool,
}

#[derive(Copy, Clone, Debug)]
struct ScreenTangent {
    centre: [Vec2<f64>; P],
    cov: [Sym2<f64>; P],
    peak: [f64; P],
}

fn sym3_outer(a: Vec3<f64>, b: Vec3<f64>, s: f64) -> Sym3<f64> {
    Sym3::new(
        s * 2.0 * a.x * b.x,
        s * (a.x * b.y + a.y * b.x),
        s * (a.x * b.z + a.z * b.x),
        s * 2.0 * a.y * b.y,
        s * (a.y * b.z + a.z * b.y),
        s * 2.0 * a.z * b.z,
    )
}

fn sym3_add(a: Sym3<f64>, b: Sym3<f64>) -> Sym3<f64> {
    Sym3::new(
        a.xx + b.xx,
        a.xy + b.xy,
        a.xz + b.xz,
        a.yy + b.yy,
        a.yz + b.yz,
        a.zz + b.zz,
    )
}

fn trace_inverse_times(a: Sym2<f64>, d: Sym2<f64>) -> f64 {
    let det = a.determinant();
    (a.yy * d.xx + a.xx * d.yy - 2.0 * a.xy * d.xy) / det
}

fn whitened_tangent(l_inv: LowerTri2<f64>, d: Sym2<f64>) -> LowerTri2<f64> {
    let m00 = l_inv.a;
    let m10 = l_inv.b;
    let m11 = l_inv.c;
    let x00 = m00 * m00 * d.xx;
    let x10 = m10 * m00 * d.xx + m11 * m00 * d.xy;
    let x11 = m10 * m10 * d.xx + 2.0 * m10 * m11 * d.xy + m11 * m11 * d.yy;
    LowerTri2 {
        a: 0.5 * x00,
        b: x10,
        c: 0.5 * x11,
    }
}

fn screen_tangent(
    atom: &Atom2d,
    index: usize,
    camera: &Camera<f64>,
    ctx: &Context<'_>,
) -> ScreenTangent {
    let zero2 = Vec2::new(0.0, 0.0);
    let zero_sym = Sym2::new(0.0, 0.0, 0.0);
    let mut t = ScreenTangent {
        centre: [zero2; P],
        cov: [zero_sym; P],
        peak: [0.0; P],
    };
    let floor = ctx.settings.scale_floor;
    let world_z = crate::fit::atom2d::depth_of(index);
    let world = crate::fit::atom2d::image_to_world(atom.centre, ctx.viewport);
    let cam = camera.to_camera(Vec3::new(world.x, world.y, world_z));
    let (clamped, _) = camera.clamp_for_jacobian(cam);
    let j = camera.jacobian(clamped);
    let j_world = j.mul_mat3(&camera.rotation);
    t.centre[0] = j_world.mul_vec(Vec3::new(1.0, 0.0, 0.0));
    t.centre[1] = j_world.mul_vec(Vec3::new(0.0, 1.0, 0.0));

    let sigma = atom.sigma();
    let sx = sigma.x.max(floor);
    let sy = sigma.y.max(floor);
    let sz_raw = sigma.x.min(sigma.y);
    let sz = sz_raw.max(floor);
    let (s, c) = atom.angle.sin_cos();
    let rx = Vec3::new(c, s, 0.0);
    let ry = Vec3::new(-s, c, 0.0);
    let rz = Vec3::new(0.0, 0.0, 1.0);
    let drx = Vec3::new(-s, c, 0.0);
    let dry = Vec3::new(-c, -s, 0.0);

    let z_live = sz_raw >= floor;
    let d_log_x = {
        let own = if sigma.x >= floor {
            sym3_outer(rx, rx, sx * sx)
        } else {
            Sym3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        };
        let depth = if z_live && sigma.x < sigma.y {
            sym3_outer(rz, rz, sz * sz)
        } else {
            Sym3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        };
        sym3_add(own, depth)
    };
    let d_log_y = {
        let own = if sigma.y >= floor {
            sym3_outer(ry, ry, sy * sy)
        } else {
            Sym3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        };
        let depth = if z_live && !(sigma.x < sigma.y) {
            sym3_outer(rz, rz, sz * sz)
        } else {
            Sym3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        };
        sym3_add(own, depth)
    };
    let d_angle = sym3_add(
        sym3_outer(drx, rx, sx * sx),
        sym3_outer(dry, ry, sy * sy),
    );
    for (slot, d3) in [(2usize, d_log_x), (3, d_log_y), (4, d_angle)] {
        let d_cam = camera.covariance_to_camera(&d3);
        t.cov[slot] = d_cam.project_2x3(&j);
    }

    let raw = {
        let mut floored = crate::fit::atom2d::to_atom(atom, index, ctx.viewport);
        floored.scale = Vec3::new(sx, sy, sz);
        camera.covariance_to_camera(&floored.covariance()).project_2x3(&j)
    };
    let v = ctx.settings.screen_dilation;
    let dilated = raw.add_diagonal(v);
    let compensated = ctx.settings.energy_compensation
        && v > 0.0
        && raw.determinant() > 0.0
        && dilated.determinant() > 0.0;
    let gain = if compensated {
        (raw.determinant() / dilated.determinant()).sqrt()
    } else {
        1.0
    };
    t.peak[5] = gain;
    if compensated {
        for slot in 2..5 {
            let d = t.cov[slot];
            let d_log_gain = 0.5 * (trace_inverse_times(raw, d) - trace_inverse_times(dilated, d));
            t.peak[slot] = atom.opacity * gain * d_log_gain;
        }
    }
    t
}

pub fn analytic_gradient(
    atoms: &[Atom2d],
    index: usize,
    ctx: &Context<'_>,
    budget: &mut Budget,
) -> Result<Backward> {
    match ctx.kernel {
        KernelFamily::Gaussian => (),
        other => {
            return Err(LightsError::GradientUnsupported {
                kernel: other.name(),
            });
        }
    }
    if index >= atoms.len() {
        return Err(LightsError::AtomOutOfRange {
            index,
            len: atoms.len(),
        });
    }
    budget.tick()?;
    let settings = ctx.settings;
    let blend = settings.blend;
    let camera = Camera::<f64>::from_spec(&ctx.spec, settings.near_plane)?;
    let scene = to_scene(atoms, ctx.viewport, settings.scale_floor)?;
    let params = vec![<Gaussian as Kernel<f64>>::Params::default(); scene.len()];
    let prepared = PreparedScene::<f64, Gaussian>::build(&scene, &params, &settings)?;
    let mut screen: Vec<ScreenAtom<f64, Gaussian>> = Vec::new();
    project_scene::<f64, Gaussian>(&prepared, &camera, &settings, &mut screen);
    screen.sort_by_key(|a| (depth_bits(a.depth), a.index));
    let visible = screen.iter().any(|a| a.index as usize == index);
    let tangent = screen_tangent(&atoms[index], index, &camera, ctx);

    let (width, height) = ctx.viewport;
    let mut image = FloatImage::filled(width, height, settings.background);
    let mut gradient = [0.0f64; P];
    for py in 0..height {
        for px in 0..width {
            let sample = camera.sample_point(px, py);
            let mut acc: OverAccum<f64> = <Over as BlendMode<f64>>::init();
            let mut hit: Option<(f64, f64, Rgb<f64>, Rgb<f64>, [f64; P])> = None;
            for entry in screen.iter() {
                let offset = entry.centre - sample;
                let whitened = entry.frame.l_inv.apply(offset);
                let r2 = whitened.length_squared();
                if !(r2 <= entry.support_sq) {
                    continue;
                }
                let weight = <Gaussian as Kernel<f64>>::eval_screen(&entry.prepared, whitened);
                let before = acc.transmittance;
                let product = entry.opacity * weight;
                let alpha = product.min(blend.alpha_cap);
                let step = <Over as BlendMode<f64>>::step(
                    &mut acc,
                    weight,
                    entry.colour,
                    entry.opacity,
                    &blend,
                );
                let composited = alpha >= blend.alpha_cutoff
                    && match (blend.early_out, step) {
                        (EarlyOut::Reference, Step::Stop) => false,
                        (EarlyOut::Reference, Step::Continue) => true,
                        (EarlyOut::Exact, Step::Stop) => true,
                        (EarlyOut::Exact, Step::Continue) => true,
                    };
                if entry.index as usize == index && composited {
                    let mut d_alpha = [0.0f64; P];
                    if product < blend.alpha_cap {
                        let g = match <Gaussian as Kernel<f64>>::d_eval_screen(
                            &entry.prepared,
                            whitened,
                        ) {
                            Some(g) => g,
                            None => {
                                return Err(LightsError::GradientUnsupported {
                                    kernel: <Gaussian as Kernel<f64>>::NAME,
                                });
                            }
                        };
                        for slot in 0..6 {
                            let moved = entry.frame.l_inv.apply(tangent.centre[slot])
                                - whitened_tangent(entry.frame.l_inv, tangent.cov[slot])
                                    .apply(whitened);
                            let d_weight = g.dot(moved);
                            d_alpha[slot] = weight * tangent.peak[slot] + entry.opacity * d_weight;
                        }
                    }
                    hit = Some((alpha, before, entry.colour, acc.rgb, d_alpha));
                }
                if let Step::Stop = step {
                    break;
                }
            }
            let colour = <Over as BlendMode<f64>>::resolve(&acc, settings.background);
            let slot = (py * width + px) as usize;
            image.pixels[slot] = colour;
            if let Some((alpha, before, own, rgb_after, d_alpha)) = hit {
                let truth = ctx.truth.pixels[slot];
                let upstream = (colour - truth) * 2.0;
                let rest = colour - rgb_after;
                let d_colour_d_alpha = own * before - rest * (1.0 / (1.0 - alpha));
                let along = upstream.dot(d_colour_d_alpha);
                for slot in 0..6 {
                    gradient[slot] += along * d_alpha[slot];
                }
                gradient[6] += upstream.x * alpha * before;
                gradient[7] += upstream.y * alpha * before;
                gradient[8] += upstream.z * alpha * before;
            }
        }
    }
    let loss = residual_energy(ctx.truth, &image)?;
    Ok(Backward {
        loss,
        gradient,
        image,
        visible,
    })
}
