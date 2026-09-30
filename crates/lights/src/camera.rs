use crate::error::{LightsError, Result};
use crate::numerics::float::FloatingPoint;
use crate::numerics::matrix::{Mat2x3, Mat3};
use crate::numerics::symmetric::Sym3;
use crate::numerics::vector::{Vec2, Vec3};
use crate::settings::{JACOBIAN_CLAMP_SLACK, PixelConvention};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Projection {
    Pinhole,
    Orthographic,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CameraSpec {
    pub eye: [f64; 3],
    pub target: [f64; 3],
    pub up: [f64; 3],
    pub fov_y_degrees: f64,
    pub width: u32,
    pub height: u32,
    pub projection: Projection,
    pub convention: PixelConvention,
    pub orthographic_extent_y: f64,
    pub intrinsics_override: Option<([f64; 2], [f64; 2])>,
}

impl CameraSpec {
    pub fn look_at(
        eye: [f64; 3],
        target: [f64; 3],
        up: [f64; 3],
        fov_y_degrees: f64,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            eye,
            target,
            up,
            fov_y_degrees,
            width,
            height,
            projection: Projection::Pinhole,
            convention: PixelConvention::CornerSample,
            orthographic_extent_y: 1.0,
            intrinsics_override: None,
        }
    }

    pub fn orthographic(mut self, extent_y: f64) -> Self {
        self.projection = Projection::Orthographic;
        self.orthographic_extent_y = extent_y;
        self
    }

    pub fn with_convention(mut self, convention: PixelConvention) -> Self {
        self.convention = convention;
        self
    }

    pub fn with_intrinsics(mut self, focal: [f64; 2], principal: [f64; 2]) -> Self {
        self.intrinsics_override = Some((focal, principal));
        self
    }

    pub fn sub_rect(&self, x0: u32, y0: u32, width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(LightsError::ViewportZero { width, height });
        }
        if x0 + width > self.width || y0 + height > self.height {
            return Err(LightsError::SubRectOutsideFrame {
                rect: (x0, y0, width, height),
                frame: (self.width, self.height),
            });
        }
        let full = Camera::<f64>::from_spec(self, 0.0)?;
        let mut out = self.clone();
        out.width = width;
        out.height = height;
        out.intrinsics_override = Some((
            [full.focal.x, full.focal.y],
            [full.principal.x - x0 as f64, full.principal.y - y0 as f64],
        ));
        Ok(out)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Camera<T> {
    pub rotation: Mat3<T>,
    pub translation: Vec3<T>,
    pub eye: Vec3<T>,
    pub focal: Vec2<T>,
    pub principal: Vec2<T>,
    pub width: u32,
    pub height: u32,
    pub near_plane: T,
    pub projection: Projection,
    pub convention: PixelConvention,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ray<T> {
    pub origin: Vec3<T>,
    pub direction: Vec3<T>,
    pub t_min: T,
    pub t_max: T,
}

impl<T: FloatingPoint> Camera<T> {
    pub fn from_spec(spec: &CameraSpec, near_plane: T) -> Result<Self> {
        if spec.width == 0 || spec.height == 0 {
            return Err(LightsError::ViewportZero {
                width: spec.width,
                height: spec.height,
            });
        }

        let eye = Vec3::<f64>::new(spec.eye[0], spec.eye[1], spec.eye[2]);
        let target = Vec3::<f64>::new(spec.target[0], spec.target[1], spec.target[2]);
        let up = Vec3::<f64>::new(spec.up[0], spec.up[1], spec.up[2]);

        let forward = match (target - eye).normalize() {
            Some(v) => v,
            None => return Err(LightsError::InvalidCamera("eye equals target")),
        };
        let down_hint = -up;
        let right = match down_hint.cross(forward).normalize() {
            Some(v) => v,
            None => return Err(LightsError::InvalidCamera("up is parallel to view direction")),
        };
        let down = forward.cross(right);

        let rotation64 = Mat3::from_rows(right, down, forward);
        let translation64 = -rotation64.mul_vec(eye);

        let width_f = spec.width as f64;
        let height_f = spec.height as f64;

        let (focal64, principal64) = match spec.intrinsics_override {
            Some((f, c)) => (Vec2::new(f[0], f[1]), Vec2::new(c[0], c[1])),
            None => {
                let f = match spec.projection {
                    Projection::Pinhole => {
                        let half = spec.fov_y_degrees.to_radians() * 0.5;
                        let tangent = half.tan();
                        if !(tangent > 0.0) || !tangent.is_finite() {
                            return Err(LightsError::InvalidCamera("fov_y out of range"));
                        }
                        (height_f * 0.5) / tangent
                    }
                    Projection::Orthographic => {
                        if !(spec.orthographic_extent_y > 0.0) {
                            return Err(LightsError::InvalidCamera(
                                "orthographic_extent_y must be positive",
                            ));
                        }
                        (height_f * 0.5) / spec.orthographic_extent_y
                    }
                };
                let c = match spec.convention {
                    PixelConvention::CornerSample => {
                        Vec2::new(width_f * 0.5 - 0.5, height_f * 0.5 - 0.5)
                    }
                    PixelConvention::CentreSample => Vec2::new(width_f * 0.5, height_f * 0.5),
                };
                (Vec2::new(f, f), c)
            }
        };

        if !(focal64.x > 0.0 && focal64.y > 0.0) || !focal64.is_finite() {
            return Err(LightsError::InvalidCamera("focal length must be positive"));
        }

        Ok(Self {
            rotation: Mat3::from_f64(&rotation64),
            translation: Vec3::from_f64(translation64),
            eye: Vec3::from_f64(eye),
            focal: Vec2::from_f64(focal64),
            principal: Vec2::from_f64(principal64),
            width: spec.width,
            height: spec.height,
            near_plane,
            projection: spec.projection,
            convention: spec.convention,
        })
    }

    #[inline(always)]
    pub fn to_camera(&self, world: Vec3<T>) -> Vec3<T> {
        self.rotation.mul_vec(world) + self.translation
    }

    #[inline(always)]
    pub fn covariance_to_camera(&self, sigma: &Sym3<T>) -> Sym3<T> {
        sigma.congruence(&self.rotation)
    }

    pub fn project(&self, cam: Vec3<T>) -> Option<Vec2<T>> {
        match self.projection {
            Projection::Pinhole => {
                if !(cam.z > T::ZERO) {
                    return None;
                }
                let inv = cam.z.recip();
                Some(Vec2::new(
                    self.focal.x * cam.x * inv + self.principal.x,
                    self.focal.y * cam.y * inv + self.principal.y,
                ))
            }
            Projection::Orthographic => Some(Vec2::new(
                self.focal.x * cam.x + self.principal.x,
                self.focal.y * cam.y + self.principal.y,
            )),
        }
    }

    pub fn jacobian_limits(&self) -> Vec2<T> {
        match self.projection {
            Projection::Pinhole => {
                let slack = T::from_f64(JACOBIAN_CLAMP_SLACK);
                Vec2::new(
                    slack * (T::from_f64(self.width as f64) * T::HALF) / self.focal.x,
                    slack * (T::from_f64(self.height as f64) * T::HALF) / self.focal.y,
                )
            }
            Projection::Orthographic => Vec2::new(T::INFINITY, T::INFINITY),
        }
    }

    pub fn clamp_for_jacobian(&self, cam: Vec3<T>) -> (Vec3<T>, bool) {
        match self.projection {
            Projection::Pinhole => {
                let lim = self.jacobian_limits();
                let inv = cam.z.recip();
                let tx = cam.x * inv;
                let ty = cam.y * inv;
                let cx = tx.clamp(-lim.x, lim.x);
                let cy = ty.clamp(-lim.y, lim.y);
                let fired = cx != tx || cy != ty;
                (Vec3::new(cx * cam.z, cy * cam.z, cam.z), fired)
            }
            Projection::Orthographic => (cam, false),
        }
    }

    pub fn jacobian(&self, cam_clamped: Vec3<T>) -> Mat2x3<T> {
        match self.projection {
            Projection::Pinhole => {
                let inv = cam_clamped.z.recip();
                let inv2 = inv * inv;
                Mat2x3::new([
                    [self.focal.x * inv, T::ZERO, -self.focal.x * cam_clamped.x * inv2],
                    [T::ZERO, self.focal.y * inv, -self.focal.y * cam_clamped.y * inv2],
                ])
            }
            Projection::Orthographic => Mat2x3::new([
                [self.focal.x, T::ZERO, T::ZERO],
                [T::ZERO, self.focal.y, T::ZERO],
            ]),
        }
    }

    #[inline(always)]
    pub fn sample_point(&self, px: u32, py: u32) -> Vec2<T> {
        let x = T::from_f64(px as f64);
        let y = T::from_f64(py as f64);
        match self.convention {
            PixelConvention::CornerSample => Vec2::new(x, y),
            PixelConvention::CentreSample => Vec2::new(x + T::HALF, y + T::HALF),
        }
    }

    pub fn ray(&self, px: u32, py: u32) -> Ray<T> {
        let s = self.sample_point(px, py);
        let ax = (s.x - self.principal.x) / self.focal.x;
        let ay = (s.y - self.principal.y) / self.focal.y;
        match self.projection {
            Projection::Pinhole => {
                let d = Vec3::new(ax, ay, T::ONE);
                let inv = d.length().recip();
                Ray {
                    origin: Vec3::zero(),
                    direction: d * inv,
                    t_min: self.near_plane,
                    t_max: T::INFINITY,
                }
            }
            Projection::Orthographic => Ray {
                origin: Vec3::new(ax, ay, T::ZERO),
                direction: Vec3::new(T::ZERO, T::ZERO, T::ONE),
                t_min: self.near_plane,
                t_max: T::INFINITY,
            },
        }
    }

    pub fn screen_disc_bound(&self, cam: Vec3<T>, world_radius: T) -> Option<(Vec2<T>, T)> {
        if !world_radius.is_finite() || world_radius < T::ZERO || !cam.is_finite() {
            return None;
        }
        match self.projection {
            Projection::Pinhole => {
                let z_min = cam.z - world_radius;
                if !(z_min > T::ZERO) {
                    return None;
                }
                let centre = self.project(cam)?;
                let denom = z_min * cam.z;
                let hx = self.focal.x * world_radius
                    * (cam.z * cam.z + cam.x * cam.x).sqrt()
                    / denom;
                let hy = self.focal.y * world_radius
                    * (cam.z * cam.z + cam.y * cam.y).sqrt()
                    / denom;
                let r = (hx * hx + hy * hy).sqrt();
                if !r.is_finite() {
                    return None;
                }
                Some((centre, r))
            }
            Projection::Orthographic => {
                let centre = self.project(cam)?;
                let hx = self.focal.x * world_radius;
                let hy = self.focal.y * world_radius;
                Some((centre, (hx * hx + hy * hy).sqrt()))
            }
        }
    }

    pub fn to_f64(&self) -> Camera<f64> {
        Camera {
            rotation: self.rotation.to_f64(),
            translation: self.translation.to_f64(),
            eye: self.eye.to_f64(),
            focal: self.focal.to_f64(),
            principal: self.principal.to_f64(),
            width: self.width,
            height: self.height,
            near_plane: self.near_plane.to_f64(),
            projection: self.projection,
            convention: self.convention,
        }
    }

    pub fn from_f64(c: &Camera<f64>) -> Self {
        Self {
            rotation: Mat3::from_f64(&c.rotation),
            translation: Vec3::from_f64(c.translation),
            eye: Vec3::from_f64(c.eye),
            focal: Vec2::from_f64(c.focal),
            principal: Vec2::from_f64(c.principal),
            width: c.width,
            height: c.height,
            near_plane: T::from_f64(c.near_plane),
            projection: c.projection,
            convention: c.convention,
        }
    }
}
