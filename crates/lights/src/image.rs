use crate::blend::BlendMode;
use crate::error::{LightsError, Result};
use crate::numerics::float::FloatingPoint;
use crate::numerics::vector::Rgb;
use crate::tiling::{TileGrid, TileRect};

pub struct Framebuffer<T: FloatingPoint, B: BlendMode<T>> {
    width: u32,
    height: u32,
    grid: TileGrid,
    accum: Vec<B::Accum>,
}

pub struct TileAccum<'a, T: FloatingPoint, B: BlendMode<T>> {
    pub rect: TileRect,
    pub stride: i32,
    pub cells: &'a mut [B::Accum],
}

impl<T: FloatingPoint, B: BlendMode<T>> Framebuffer<T, B> {
    pub fn new(width: u32, height: u32, tile: u32) -> Result<Self> {
        let grid = TileGrid::new(width, height, tile)?;
        let per_tile = (tile as usize) * (tile as usize);
        Ok(Self {
            width,
            height,
            grid,
            accum: vec![B::init(); grid.tile_count() * per_tile],
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn grid(&self) -> TileGrid {
        self.grid
    }

    pub fn clear(&mut self) {
        for c in self.accum.iter_mut() {
            *c = B::init();
        }
    }

    pub fn split_tiles_mut(&mut self, grid: &TileGrid) -> Result<Vec<TileAccum<'_, T, B>>> {
        if *grid != self.grid {
            return Err(LightsError::SizeMismatch {
                a: (self.grid.width, self.grid.height),
                b: (grid.width, grid.height),
            });
        }
        let tile = self.grid.tile as i32;
        let per_tile = (tile as usize) * (tile as usize);
        let rects: Vec<TileRect> = self
            .grid
            .tiles()
            .map(|(tx, ty)| self.grid.rect(tx, ty))
            .collect();
        Ok(self
            .accum
            .chunks_exact_mut(per_tile)
            .zip(rects)
            .map(|(cells, rect)| TileAccum {
                rect,
                stride: tile,
                cells,
            })
            .collect())
    }

    fn index(&self, px: u32, py: u32) -> usize {
        let tile = self.grid.tile;
        let tx = (px / tile) as i32;
        let ty = (py / tile) as i32;
        let id = self.grid.tile_id(tx, ty) as usize;
        let local = ((py % tile) * tile + (px % tile)) as usize;
        id * (tile as usize) * (tile as usize) + local
    }

    pub fn cell(&self, px: u32, py: u32) -> B::Accum {
        if px >= self.width || py >= self.height {
            crate::guard::unreachable(
                "Framebuffer::cell",
                &format!("pixel ({px},{py}) outside {}x{}", self.width, self.height),
            );
        }
        self.accum[self.index(px, py)]
    }

    pub fn resolve_float(&self, background: Rgb<T>) -> FloatImage<T> {
        let mut out = FloatImage::new(self.width, self.height);
        for py in 0..self.height {
            for px in 0..self.width {
                out.set(px, py, B::resolve(&self.accum[self.index(px, py)], background));
            }
        }
        out
    }

    pub fn resolve(&self, background: Rgb<T>) -> Image {
        self.resolve_float(background).to_clamped_u8()
    }

    pub fn resolve_bgra8(&self, background: Rgb<T>, bytes: &mut [u8], stride: usize) -> Result<()> {
        let row = 4 * self.width as usize;
        let needed = stride
            .checked_mul((self.height as usize).saturating_sub(1))
            .and_then(|n| n.checked_add(row));
        if stride < row || needed.is_none_or(|n| bytes.len() < n) {
            return Err(LightsError::ByteTarget {
                width: self.width,
                height: self.height,
                stride,
                len: bytes.len(),
            });
        }
        let tile = self.grid.tile as usize;
        let per_tile = tile * tile;
        let bands = bytes
            .chunks_mut(stride.saturating_mul(tile))
            .take(self.grid.grid_y as usize);
        for (ty, band) in bands.enumerate() {
            for tx in 0..self.grid.grid_x {
                let rect = self.grid.rect(tx, ty as i32);
                let id = self.grid.tile_id(tx, ty as i32) as usize;
                let cells = &self.accum[id * per_tile..(id + 1) * per_tile];
                let (x0, w) = (4 * rect.x0 as usize, rect.w as usize);
                for (ly, from) in cells.chunks_exact(tile).take(rect.h as usize).enumerate() {
                    let at = ly * stride + x0;
                    for (px, cell) in band[at..at + 4 * w].chunks_exact_mut(4).zip(from) {
                        let c = B::resolve(cell, background);
                        px.copy_from_slice(&[quantise(c.z), quantise(c.y), quantise(c.x), u8::MAX]);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn resident_bytes(&self) -> usize {
        self.accum.len() * core::mem::size_of::<B::Accum>()
    }

    pub fn resolve_premultiplied(&self) -> (FloatImage<T>, FloatImage<T>) {
        let mut colour = FloatImage::new(self.width, self.height);
        let mut alpha = FloatImage::new(self.width, self.height);
        for py in 0..self.height {
            for px in 0..self.width {
                let cell = &self.accum[self.index(px, py)];
                colour.set(px, py, B::resolve(cell, Rgb::zero()));
                alpha.set(px, py, Rgb::splat(B::coverage(cell)));
            }
        }
        (colour, alpha)
    }

    pub fn coverage(&self) -> FloatImage<T> {
        let mut out = FloatImage::new(self.width, self.height);
        for py in 0..self.height {
            for px in 0..self.width {
                let c = B::coverage(&self.accum[self.index(px, py)]);
                out.set(px, py, Rgb::splat(c));
            }
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FloatImage<T> {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Rgb<T>>,
}

impl<T: FloatingPoint> FloatImage<T> {
    pub fn new(width: u32, height: u32) -> Self {
        Self::filled(width, height, Rgb::zero())
    }

    pub fn filled(width: u32, height: u32, value: Rgb<T>) -> Self {
        Self {
            width,
            height,
            pixels: vec![value; (width as usize) * (height as usize)],
        }
    }

    fn offset(&self, x: u32, y: u32) -> usize {
        if x >= self.width || y >= self.height {
            crate::guard::unreachable(
                "FloatImage::offset",
                &format!("pixel ({x},{y}) outside {}x{}", self.width, self.height),
            );
        }
        (y as usize) * (self.width as usize) + (x as usize)
    }

    pub fn get(&self, x: u32, y: u32) -> Rgb<T> {
        self.pixels[self.offset(x, y)]
    }

    pub fn set(&mut self, x: u32, y: u32, v: Rgb<T>) {
        let o = self.offset(x, y);
        self.pixels[o] = v;
    }

    pub fn difference(&self, other: &Self) -> Result<FloatImage<T>> {
        if self.width != other.width || self.height != other.height {
            return Err(LightsError::SizeMismatch {
                a: (self.width, self.height),
                b: (other.width, other.height),
            });
        }
        Ok(FloatImage {
            width: self.width,
            height: self.height,
            pixels: self
                .pixels
                .iter()
                .zip(other.pixels.iter())
                .map(|(a, b)| *a - *b)
                .collect(),
        })
    }

    pub fn composite_over(&self, coverage: &Self, background: &Self) -> Result<FloatImage<T>> {
        if self.width != coverage.width || self.height != coverage.height {
            return Err(LightsError::SizeMismatch {
                a: (self.width, self.height),
                b: (coverage.width, coverage.height),
            });
        }
        if self.width != background.width || self.height != background.height {
            return Err(LightsError::SizeMismatch {
                a: (self.width, self.height),
                b: (background.width, background.height),
            });
        }
        Ok(FloatImage {
            width: self.width,
            height: self.height,
            pixels: self
                .pixels
                .iter()
                .zip(coverage.pixels.iter())
                .zip(background.pixels.iter())
                .map(|((c, a), b)| *c + *b * (T::ONE - a.x))
                .collect(),
        })
    }

    pub fn blit_from(
        &mut self,
        patch: &Self,
        coverage: Option<&Self>,
        x0: u32,
        y0: u32,
    ) -> Result<()> {
        if x0 + patch.width > self.width || y0 + patch.height > self.height {
            return Err(LightsError::SubRectOutsideFrame {
                rect: (x0, y0, patch.width, patch.height),
                frame: (self.width, self.height),
            });
        }
        if let Some(alpha) = coverage {
            if alpha.width != patch.width || alpha.height != patch.height {
                return Err(LightsError::SizeMismatch {
                    a: (patch.width, patch.height),
                    b: (alpha.width, alpha.height),
                });
            }
        }
        for py in 0..patch.height {
            for px in 0..patch.width {
                let source = patch.get(px, py);
                let value = match coverage {
                    Some(alpha) => {
                        let under = self.get(x0 + px, y0 + py);
                        source + under * (T::ONE - alpha.get(px, py).x)
                    }
                    None => source,
                };
                self.set(x0 + px, y0 + py, value);
            }
        }
        Ok(())
    }

    pub fn max_abs(&self) -> (T, u32, u32) {
        let mut best = T::ZERO;
        let mut bx = 0;
        let mut by = 0;
        for y in 0..self.height {
            for x in 0..self.width {
                let p = self.get(x, y);
                let m = p.x.abs().max(p.y.abs()).max(p.z.abs());
                if m > best {
                    best = m;
                    bx = x;
                    by = y;
                }
            }
        }
        (best, bx, by)
    }

    pub fn min_max_channel(&self) -> (T, T) {
        let mut lo = T::INFINITY;
        let mut hi = T::NEG_INFINITY;
        for p in self.pixels.iter() {
            lo = lo.min(p.x).min(p.y).min(p.z);
            hi = hi.max(p.x).max(p.y).max(p.z);
        }
        (lo, hi)
    }

    pub fn rms(&self) -> T {
        if self.pixels.is_empty() {
            return T::ZERO;
        }
        let mut acc = T::ZERO;
        for p in self.pixels.iter() {
            acc += p.x * p.x + p.y * p.y + p.z * p.z;
        }
        let n = T::from_f64((self.pixels.len() * 3) as f64);
        (acc / n).sqrt()
    }

    pub fn into_f64(self) -> FloatImage<f64> {
        FloatImage {
            width: self.width,
            height: self.height,
            pixels: T::widen_rgb(self.pixels),
        }
    }

    pub fn to_f64(&self) -> FloatImage<f64> {
        FloatImage {
            width: self.width,
            height: self.height,
            pixels: self.pixels.iter().map(|p| p.to_f64()).collect(),
        }
    }

    pub fn from_f64(v: &FloatImage<f64>) -> Self {
        FloatImage {
            width: v.width,
            height: v.height,
            pixels: v.pixels.iter().map(|p| Rgb::from_f64(*p)).collect(),
        }
    }

    pub fn encode_srgb(&self) -> Self {
        self.map_channels(srgb_encode)
    }

    pub fn decode_srgb(&self) -> Self {
        self.map_channels(srgb_decode)
    }

    fn map_channels(&self, f: fn(T) -> T) -> Self {
        FloatImage {
            width: self.width,
            height: self.height,
            pixels: self.pixels.iter().map(|p| Rgb::new(f(p.x), f(p.y), f(p.z))).collect(),
        }
    }

    pub fn to_clamped_u8(&self) -> Image {
        Image {
            width: self.width,
            height: self.height,
            pixels: self
                .pixels
                .iter()
                .map(|p| {
                    [
                        quantise(p.x),
                        quantise(p.y),
                        quantise(p.z),
                    ]
                })
                .collect(),
        }
    }
}

const BYTE_TOP: f64 = 255.0;

#[derive(Clone, Debug, PartialEq)]
pub struct DepthImage<T> {
    width: u32,
    height: u32,
    depth: Vec<T>,
}

impl<T: FloatingPoint> DepthImage<T> {
    pub fn far(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth: vec![T::INFINITY; (width as usize) * (height as usize)],
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn get(&self, x: u32, y: u32) -> T {
        self.depth[(y as usize) * (self.width as usize) + (x as usize)]
    }

    pub fn set(&mut self, x: u32, y: u32, depth: T) {
        let o = (y as usize) * (self.width as usize) + (x as usize);
        self.depth[o] = depth;
    }

    pub fn finite_count(&self) -> usize {
        self.depth.iter().filter(|d| d.is_finite()).count()
    }

    pub fn convert<U: FloatingPoint>(&self) -> DepthImage<U> {
        DepthImage {
            width: self.width,
            height: self.height,
            depth: self.depth.iter().map(|d| U::from_f64(d.to_f64())).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RasterLayer<T> {
    pub premultiplied: FloatImage<T>,
    pub coverage: FloatImage<T>,
}

fn check_raster(width: u32, height: u32, channels: usize, got: usize) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(LightsError::ViewportZero { width, height });
    }
    let expected = (width as usize) * (height as usize) * channels;
    if got != expected {
        return Err(LightsError::RasterLength {
            width,
            height,
            channels,
            got,
        });
    }
    Ok(())
}

impl<T: FloatingPoint> RasterLayer<T> {
    pub fn opaque(image: &Image) -> Result<Self> {
        check_raster(image.width, image.height, 1, image.pixels.len())?;
        let decode = |c: u8| T::from_f64(f64::from(c) / BYTE_TOP);
        let premultiplied = FloatImage {
            width: image.width,
            height: image.height,
            pixels: image
                .pixels
                .iter()
                .map(|p| Rgb::new(decode(p[0]), decode(p[1]), decode(p[2])))
                .collect(),
        };
        Ok(Self {
            coverage: FloatImage::filled(image.width, image.height, Rgb::splat(T::ONE)),
            premultiplied,
        })
    }

    pub fn from_rgba8(width: u32, height: u32, bytes: &[u8]) -> Result<Self> {
        check_raster(width, height, 4, bytes.len())?;
        let decode = |c: u8| T::from_f64(f64::from(c) / BYTE_TOP);
        let mut premultiplied = FloatImage::new(width, height);
        let mut coverage = FloatImage::new(width, height);
        for (i, px) in bytes.chunks_exact(4).enumerate() {
            let a = decode(px[3]);
            premultiplied.pixels[i] = Rgb::new(decode(px[0]), decode(px[1]), decode(px[2])) * a;
            coverage.pixels[i] = Rgb::splat(a);
        }
        Ok(Self {
            premultiplied,
            coverage,
        })
    }

    pub fn from_coverage8(width: u32, height: u32, bytes: &[u8], ink: Rgb<T>) -> Result<Self> {
        check_raster(width, height, 1, bytes.len())?;
        let decode = |c: u8| T::from_f64(f64::from(c) / BYTE_TOP);
        Ok(Self {
            premultiplied: FloatImage {
                width,
                height,
                pixels: bytes.iter().map(|c| ink * decode(*c)).collect(),
            },
            coverage: FloatImage {
                width,
                height,
                pixels: bytes.iter().map(|c| Rgb::splat(decode(*c))).collect(),
            },
        })
    }

    pub fn width(&self) -> u32 {
        self.premultiplied.width
    }

    pub fn height(&self) -> u32 {
        self.premultiplied.height
    }

    pub fn placed(&self, frame_width: u32, frame_height: u32, x0: i32, y0: i32) -> Result<Self> {
        if frame_width == 0 || frame_height == 0 {
            return Err(LightsError::ViewportZero {
                width: frame_width,
                height: frame_height,
            });
        }
        let mut premultiplied = FloatImage::new(frame_width, frame_height);
        let mut coverage = FloatImage::new(frame_width, frame_height);
        for sy in 0..self.height() {
            let fy = i64::from(y0) + i64::from(sy);
            if fy < 0 || fy >= i64::from(frame_height) {
                continue;
            }
            for sx in 0..self.width() {
                let fx = i64::from(x0) + i64::from(sx);
                if fx < 0 || fx >= i64::from(frame_width) {
                    continue;
                }
                premultiplied.set(fx as u32, fy as u32, self.premultiplied.get(sx, sy));
                coverage.set(fx as u32, fy as u32, self.coverage.get(sx, sy));
            }
        }
        Ok(Self {
            premultiplied,
            coverage,
        })
    }

    pub fn over(&self, background: &FloatImage<T>) -> Result<FloatImage<T>> {
        self.premultiplied.composite_over(&self.coverage, background)
    }
}

const SRGB_LINEAR_SLOPE: f64 = 12.92;
const SRGB_ENCODE_KNEE: f64 = 0.003_130_8;
const SRGB_DECODE_KNEE: f64 = 0.040_45;
const SRGB_OFFSET: f64 = 0.055;
const SRGB_EXPONENT: f64 = 2.4;

pub fn srgb_encode<T: FloatingPoint>(linear: T) -> T {
    let v = linear.to_f64();
    let encoded = if v <= SRGB_ENCODE_KNEE {
        v * SRGB_LINEAR_SLOPE
    } else {
        (1.0 + SRGB_OFFSET) * v.powf(1.0 / SRGB_EXPONENT) - SRGB_OFFSET
    };
    T::from_f64(encoded)
}

pub fn srgb_decode<T: FloatingPoint>(encoded: T) -> T {
    let v = encoded.to_f64();
    let linear = if v <= SRGB_DECODE_KNEE {
        v / SRGB_LINEAR_SLOPE
    } else {
        ((v + SRGB_OFFSET) / (1.0 + SRGB_OFFSET)).powf(SRGB_EXPONENT)
    };
    T::from_f64(linear)
}

fn quantise<T: FloatingPoint>(v: T) -> u8 {
    let c = v.clamp(T::ZERO, T::ONE).to_f64();
    (c * 255.0) as u8
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[u8; 3]>,
}

impl Image {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![[0u8; 3]; (width as usize) * (height as usize)],
        }
    }

    pub fn as_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixels.len() * 3);
        for p in self.pixels.iter() {
            out.extend_from_slice(p);
        }
        out
    }
}

pub fn write_png(path: &std::path::Path, image: &Image) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let buffered = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(buffered, image.width, image.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| LightsError::PngEncode(format!("{e}")))?;
    writer
        .write_image_data(&image.as_bytes())
        .map_err(|e| LightsError::PngEncode(format!("{e}")))?;
    Ok(())
}

fn decode_png(path: &std::path::Path) -> Result<(png::OutputInfo, Vec<u8>)> {
    let file = std::fs::File::open(path)?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder
        .read_info()
        .map_err(|e| LightsError::PngEncode(format!("{e}")))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| LightsError::PngEncode(format!("{e}")))?;
    buf.truncate(info.buffer_size());
    Ok((info, buf))
}

fn rgb_image(info: &png::OutputInfo, buf: &[u8]) -> Image {
    Image {
        width: info.width,
        height: info.height,
        pixels: buf.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
    }
}

pub fn read_png(path: &std::path::Path) -> Result<Image> {
    let (info, buf) = decode_png(path)?;
    if info.color_type != png::ColorType::Rgb || info.bit_depth != png::BitDepth::Eight {
        return Err(LightsError::PngEncode(
            "expected 8-bit RGB png".to_string(),
        ));
    }
    Ok(rgb_image(&info, &buf))
}

impl<T: FloatingPoint> RasterLayer<T> {
    pub fn read_png(path: &std::path::Path) -> Result<Self> {
        let (info, buf) = decode_png(path)?;
        if info.bit_depth != png::BitDepth::Eight {
            return Err(LightsError::PngEncode(
                "expected an 8-bit png".to_string(),
            ));
        }
        match info.color_type {
            png::ColorType::Rgb => Self::opaque(&rgb_image(&info, &buf)),
            png::ColorType::Rgba => Self::from_rgba8(info.width, info.height, &buf),
            png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha | png::ColorType::Indexed => {
                Err(LightsError::PngEncode(
                    "expected an 8-bit RGB or RGBA png".to_string(),
                ))
            }
        }
    }
}

pub fn write_png_normalized(path: &std::path::Path, image: &FloatImage<f64>) -> Result<f64> {
    let (peak, _, _) = image.max_abs();
    let scale = if peak > 0.0 { 1.0 / peak } else { 1.0 };
    let scaled = FloatImage {
        width: image.width,
        height: image.height,
        pixels: image.pixels.iter().map(|p| *p * scale).collect(),
    };
    write_png(path, &scaled.to_clamped_u8())?;
    Ok(scale)
}
