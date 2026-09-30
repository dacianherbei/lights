#![allow(dead_code)]
use lights::fit::{Criterion, Mapping, Primitive, Target, TargetMetric, measure_target_mapped};
use lights::{DrawList, DrawOp, DrawShape, FloatImage, Ink, RasterLayer, Rgb, Similarity2, Vec2};
use rayon::prelude::*;
use std::num::NonZeroU32;
use tiny_skia::{
    Color, FillRule, IntSize, LineCap, LineJoin, Mask, Paint, Path, PathBuilder, Pixmap, PixmapMut, Rect, Stroke,
    Transform,
};

pub const KAPPA: f32 = 0.552_284_8;
pub const ARC_POINTS: usize = 97;
pub const HALF_PLANE_MARGIN: f64 = 64.0;

pub enum Figure {
    Rect(Rect),
    Fill(Path, FillRule),
    Stroke(Path, Stroke),
}

pub struct TinyOp {
    pub figure: Figure,
    pub paint: Paint<'static>,
    pub reach: [f32; 4],
    pub clip: Option<[f32; 4]>,
    pub masked: bool,
    pub cut: bool,
}

#[derive(Default)]
pub struct TinyList {
    pub ops: Vec<TinyOp>,
    pub skipped: usize,
    pub masked: usize,
    pub clipped_rects: usize,
}

fn paint(ink: &Ink<f32>) -> Option<Paint<'static>> {
    let a = ink.alpha.clamp(0.0, 1.0);
    if a <= 0.0 {
        return None;
    }
    let straight = |c: f32| (c / a).clamp(0.0, 1.0);
    let colour = Color::from_rgba(
        straight(ink.premultiplied.x),
        straight(ink.premultiplied.y),
        straight(ink.premultiplied.z),
        a,
    )?;
    let mut p = Paint::default();
    p.set_color(colour);
    p.anti_alias = true;
    Some(p)
}

pub fn rounded(l: f32, t: f32, r: f32, b: f32, radius: f32) -> Option<Path> {
    let rad = radius.min((r - l) * 0.5).min((b - t) * 0.5).max(0.0);
    if rad <= 0.0 {
        return Some(PathBuilder::from_rect(Rect::from_ltrb(l, t, r, b)?));
    }
    let k = rad * (1.0 - KAPPA);
    let mut pb = PathBuilder::new();
    pb.move_to(l + rad, t);
    pb.line_to(r - rad, t);
    pb.cubic_to(r - k, t, r, t + k, r, t + rad);
    pb.line_to(r, b - rad);
    pb.cubic_to(r, b - k, r - k, b, r - rad, b);
    pb.line_to(l + rad, b);
    pb.cubic_to(l + k, b, l, b - k, l, b - rad);
    pb.line_to(l, t + rad);
    pb.cubic_to(l, t + k, l + k, t, l + rad, t);
    pb.close();
    pb.finish()
}

fn polyline(points: &[(f32, f32)], closed: bool) -> Option<Path> {
    let mut pb = PathBuilder::new();
    for (i, (x, y)) in points.iter().enumerate() {
        if i == 0 {
            pb.move_to(*x, *y);
        } else {
            pb.line_to(*x, *y);
        }
    }
    if closed {
        pb.close();
    }
    pb.finish()
}

fn stroke(half_width: f32, scale: f32, cap: LineCap) -> Stroke {
    Stroke {
        width: 2.0 * half_width * scale,
        miter_limit: 4.0,
        line_cap: cap,
        line_join: LineJoin::Round,
        dash: None,
    }
}

pub fn figure(shape: &DrawShape<f32>, view: Similarity2<f32>) -> Option<Figure> {
    let s = view.scale();
    let at = |p: Vec2<f32>| {
        let q = view.apply(p);
        (q.x, q.y)
    };
    match shape {
        DrawShape::Rect {
            min,
            max,
            corner_radius,
        } => {
            let ((l, t), (r, b)) = (at(*min), at(*max));
            if *corner_radius * s <= 0.0 {
                Some(Figure::Rect(Rect::from_ltrb(l, t, r, b)?))
            } else {
                Some(Figure::Fill(rounded(l, t, r, b, *corner_radius * s)?, FillRule::Winding))
            }
        }
        DrawShape::RectOutline {
            min,
            max,
            corner_radius,
            half_width,
        } => {
            let ((l, t), (r, b)) = (at(*min), at(*max));
            Some(Figure::Stroke(rounded(l, t, r, b, *corner_radius * s)?, stroke(*half_width, s, LineCap::Butt)))
        }
        DrawShape::Segment { a, b, half_width } => {
            let (a, b) = (at(*a), at(*b));
            if a == b {
                return Some(Figure::Fill(PathBuilder::from_circle(a.0, a.1, *half_width * s)?, FillRule::Winding));
            }
            Some(Figure::Stroke(polyline(&[a, b], false)?, stroke(*half_width, s, LineCap::Round)))
        }
        DrawShape::Disc { centre, radius } => {
            let (x, y) = at(*centre);
            Some(Figure::Fill(PathBuilder::from_circle(x, y, *radius * s)?, FillRule::Winding))
        }
        DrawShape::Ring {
            centre,
            radius,
            half_width,
        } => {
            let (x, y) = at(*centre);
            Some(Figure::Stroke(PathBuilder::from_circle(x, y, *radius * s)?, stroke(*half_width, s, LineCap::Butt)))
        }
        DrawShape::Polygon { points } => {
            let device: Vec<(f32, f32)> = points.iter().map(|p| at(*p)).collect();
            Some(Figure::Fill(polyline(&device, true)?, FillRule::EvenOdd))
        }
        DrawShape::Polyline {
            points,
            closed,
            half_width,
        } => {
            let device: Vec<(f32, f32)> = points.iter().map(|p| at(*p)).collect();
            Some(Figure::Stroke(polyline(&device, *closed)?, stroke(*half_width, s, LineCap::Round)))
        }
    }
}

pub const FRINGE: f32 = 1.0;
pub const HAIRLINE_HALF_WIDTH: f32 = 0.5;

pub fn reach(figure: &Figure) -> [f32; 4] {
    let (b, h) = match figure {
        Figure::Rect(r) => ([r.left(), r.top(), r.right(), r.bottom()], 0.0),
        Figure::Fill(p, _) => {
            let b = p.bounds();
            ([b.left(), b.top(), b.right(), b.bottom()], 0.0)
        }
        Figure::Stroke(p, s) => {
            let b = p.bounds();
            ([b.left(), b.top(), b.right(), b.bottom()], (s.width * 0.5).max(HAIRLINE_HALF_WIDTH))
        }
    };
    let o = h + FRINGE;
    [b[0] - o, b[1] - o, b[2] + o, b[3] + o]
}

fn inside(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] >= b[0] && a[1] >= b[1] && a[2] <= b[2] && a[3] <= b[3]
}

fn disjoint(a: [f32; 4], b: [f32; 4]) -> bool {
    a[2] <= b[0] || a[0] >= b[2] || a[3] <= b[1] || a[1] >= b[3]
}

pub fn tiny_op(op: &DrawOp<f32>, view: Similarity2<f32>) -> Option<TinyOp> {
    let paint = paint(&op.ink)?;
    let mut figure = figure(&op.shape, view)?;
    let clip = op.clip.map(|c| {
        let (lo, hi) = (view.apply(c.min), view.apply(c.max));
        [lo.x.min(hi.x), lo.y.min(hi.y), lo.x.max(hi.x), lo.y.max(hi.y)]
    });
    let mut masked = false;
    let mut cut = false;
    if let Some(c) = clip {
        let r = reach(&figure);
        if disjoint(r, c) {
            return None;
        }
        if !inside(r, c) {
            match &figure {
                Figure::Rect(rect) => {
                    let inner = Rect::from_ltrb(
                        rect.left().max(c[0]),
                        rect.top().max(c[1]),
                        rect.right().min(c[2]),
                        rect.bottom().min(c[3]),
                    )?;
                    figure = Figure::Rect(inner);
                    cut = true;
                }
                Figure::Fill(..) | Figure::Stroke(..) => masked = true,
            }
        }
    }
    Some(TinyOp {
        reach: reach(&figure),
        figure,
        paint,
        clip,
        masked,
        cut,
    })
}

impl TinyList {
    pub fn from_list(list: &DrawList<f32>, view: Similarity2<f32>) -> Option<Self> {
        if view.rotation() != 0.0 {
            return None;
        }
        let mut out = TinyList::default();
        for op in list.ops().iter() {
            match tiny_op(op, view) {
                Some(t) => {
                    out.masked += usize::from(t.masked);
                    out.clipped_rects += usize::from(t.cut);
                    out.ops.push(t);
                }
                None => out.skipped += 1,
            }
        }
        Some(out)
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }
}

fn overlap(p: f32, lo: f32, hi: f32) -> f32 {
    (hi.min(p + 1.0) - lo.max(p)).clamp(0.0, 1.0)
}

pub fn clip_mask(clip: [f32; 4], width: u32, top: u32, rows: u32) -> Option<Mask> {
    let size = IntSize::from_wh(width, rows)?;
    let mut data = vec![0u8; width as usize * rows as usize];
    let x0 = clip[0].floor().max(0.0) as u32;
    let x1 = (clip[2].ceil().max(0.0) as u32).min(width);
    let y0 = (clip[1].floor().max(0.0) as u32).max(top);
    let y1 = (clip[3].ceil().max(0.0) as u32).min(top + rows);
    for py in y0..y1 {
        let fy = overlap(py as f32, clip[1], clip[3]);
        let row = (py - top) as usize * width as usize;
        for px in x0..x1 {
            let fx = overlap(px as f32, clip[0], clip[2]);
            data[row + px as usize] = (fx * fy * 255.0).round() as u8;
        }
    }
    Mask::from_vec(data, size)
}

pub fn draw_band(list: &TinyList, bytes: &mut [u8], width: u32, top: u32, rows: u32) {
    let Some(mut pixmap) = PixmapMut::from_bytes(bytes, width, rows) else {
        return;
    };
    let (upper, lower) = (top as f32, (top + rows) as f32);
    let shift = Transform::from_translate(0.0, -upper);
    let mut masks: Vec<([f32; 4], Mask)> = Vec::new();
    for op in list.ops.iter() {
        if op.reach[3] <= upper || op.reach[1] >= lower {
            continue;
        }
        let mut slot = None;
        if let (true, Some(clip)) = (op.masked, op.clip) {
            slot = masks.iter().position(|(c, _)| *c == clip);
            if slot.is_none() {
                let Some(m) = clip_mask(clip, width, top, rows) else {
                    continue;
                };
                masks.push((clip, m));
                slot = Some(masks.len() - 1);
            }
        }
        let mask = slot.and_then(|i| masks.get(i)).map(|(_, m)| m);
        match &op.figure {
            Figure::Rect(r) => {
                if let Some(moved) = Rect::from_ltrb(r.left(), r.top() - upper, r.right(), r.bottom() - upper) {
                    pixmap.fill_rect(moved, &op.paint, Transform::identity(), mask);
                }
            }
            Figure::Fill(path, rule) => pixmap.fill_path(path, &op.paint, *rule, shift, mask),
            Figure::Stroke(path, s) => pixmap.stroke_path(path, &op.paint, s, shift, mask),
        }
    }
}

pub fn render(list: &TinyList, width: u32, height: u32, band_rows: Option<NonZeroU32>) -> Option<Pixmap> {
    let mut pixmap = Pixmap::new(width, height)?;
    let stride = width as usize * 4;
    match band_rows {
        None => draw_band(list, pixmap.data_mut(), width, 0, height),
        Some(rows) => {
            let band = rows.get() as usize * stride;
            pixmap
                .data_mut()
                .par_chunks_mut(band)
                .enumerate()
                .for_each(|(i, chunk)| draw_band(list, chunk, width, i as u32 * rows.get(), (chunk.len() / stride) as u32));
        }
    }
    Some(pixmap)
}

pub fn touch(width: u32, height: u32, band_rows: Option<NonZeroU32>) -> Option<Pixmap> {
    let mut pixmap = Pixmap::new(width, height)?;
    match band_rows {
        None => pixmap.data_mut().fill(255),
        Some(rows) => {
            let band = rows.get() as usize * width as usize * 4;
            pixmap.data_mut().par_chunks_mut(band).for_each(|c| c.fill(255));
        }
    }
    Some(pixmap)
}

pub fn pixmap_image(pixmap: &Pixmap) -> FloatImage<f64> {
    let mut image = FloatImage::<f64>::new(pixmap.width(), pixmap.height());
    for (i, px) in pixmap.data().chunks_exact(4).enumerate() {
        image.pixels[i] = Rgb::new(f64::from(px[0]), f64::from(px[1]), f64::from(px[2])) * (1.0 / 255.0);
    }
    image
}

pub fn pixmap_ink(pixmap: &Pixmap) -> f64 {
    pixmap.data().chunks_exact(4).map(|px| f64::from(px[3])).sum::<f64>() / 255.0
}

pub fn paint_sum(rgba: &[u8]) -> f64 {
    rgba.chunks_exact(4).map(|px| f64::from(px[0]) + f64::from(px[1]) + f64::from(px[2])).sum::<f64>() / (3.0 * 255.0)
}

pub const FAR_LEVELS: u8 = 64;

pub fn departure(a: &[u8], b: &[u8]) -> (f64, u64) {
    let mut total = 0u64;
    let mut far = 0u64;
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let mut worst = 0u8;
        for (p, q) in x.iter().zip(y.iter()) {
            let d = p.abs_diff(*q);
            total += u64::from(d);
            worst = worst.max(d);
        }
        far += u64::from(worst > FAR_LEVELS);
    }
    (total as f64 / a.len().max(1) as f64, far)
}

pub fn layer_ink(layer: &RasterLayer<f32>) -> f64 {
    layer.coverage.pixels.iter().map(|p| f64::from(p.x)).sum()
}

pub fn pack_rgba(layer: &RasterLayer<f32>) -> Vec<u8> {
    let rgb = layer.premultiplied.to_clamped_u8();
    let mut out = Vec::with_capacity(rgb.pixels.len() * 4);
    for (px, a) in rgb.pixels.iter().zip(layer.coverage.pixels.iter()) {
        out.extend_from_slice(&[px[0], px[1], px[2], quantise(a.x)]);
    }
    out
}

fn quantise(v: f32) -> u8 {
    (f64::from(v.clamp(0.0, 1.0)) * 255.0) as u8
}

pub fn pack_rgba_parallel(layer: &RasterLayer<f32>, band_rows: NonZeroU32) -> Vec<u8> {
    let width = layer.premultiplied.width as usize;
    let rows = band_rows.get() as usize;
    let mut out = vec![0u8; layer.premultiplied.pixels.len() * 4];
    out.par_chunks_mut(rows * width * 4)
        .zip(layer.premultiplied.pixels.par_chunks(rows * width))
        .zip(layer.coverage.pixels.par_chunks(rows * width))
        .for_each(|((dst, colour), cover)| {
            for ((d, c), a) in dst.chunks_exact_mut(4).zip(colour.iter()).zip(cover.iter()) {
                d[0] = quantise(c.x);
                d[1] = quantise(c.y);
                d[2] = quantise(c.z);
                d[3] = quantise(a.x);
            }
        });
    out
}

pub fn pixmap_difference(a: &Pixmap, b: &Pixmap) -> (u8, u64) {
    let mut worst = 0u8;
    let mut pixels = 0u64;
    for (x, y) in a.data().chunks_exact(4).zip(b.data().chunks_exact(4)) {
        let d = x.iter().zip(y.iter()).map(|(p, q)| p.abs_diff(*q)).max().unwrap_or(0);
        worst = worst.max(d);
        pixels += u64::from(d > 0);
    }
    (worst, pixels)
}

fn clip_half_plane(polygon: &[Vec2<f64>], point: Vec2<f64>, normal: Vec2<f64>, keep_inside: bool) -> Vec<Vec2<f64>> {
    let side = |p: Vec2<f64>| {
        let d = (p - point).dot(normal);
        if keep_inside { -d } else { d }
    };
    let mut out = Vec::new();
    for i in 0..polygon.len() {
        let (a, b) = (polygon[i], polygon[(i + 1) % polygon.len()]);
        let (sa, sb) = (side(a), side(b));
        if sa >= 0.0 {
            out.push(a);
        }
        if (sa >= 0.0) != (sb >= 0.0) {
            out.push(a + (b - a) * (sa / (sa - sb)));
        }
    }
    out
}

fn v32(p: Vec2<f64>) -> Vec2<f32> {
    Vec2::new(p.x as f32, p.y as f32)
}

pub fn target_list(target: &Target<f64>) -> Option<DrawList<f32>> {
    if !target.set.regions.is_empty() {
        return None;
    }
    let ink = Ink::opaque(Rgb::new(target.set.ink.x as f32, target.set.ink.y as f32, target.set.ink.z as f32));
    let paper = Ink::opaque(Rgb::new(target.paper.x as f32, target.paper.y as f32, target.paper.z as f32));
    let (w, h) = (f64::from(target.viewport.0), f64::from(target.viewport.1));
    let m = HALF_PLANE_MARGIN;
    let frame = vec![Vec2::new(-m, -m), Vec2::new(w + m, -m), Vec2::new(w + m, h + m), Vec2::new(-m, h + m)];
    let mut list = DrawList::new();
    for part in target.set.parts.iter() {
        let shape = match part.shape {
            Primitive::HalfPlane { point, normal } => {
                let mut polygon = clip_half_plane(&frame, point, normal, true);
                for hole in target.set.holes.iter() {
                    if let Primitive::HalfPlane { point, normal } = hole.shape {
                        polygon = clip_half_plane(&polygon, point, normal, false);
                    }
                }
                DrawShape::Polygon {
                    points: polygon.into_iter().map(v32).collect(),
                }
            }
            Primitive::Segment { a, b, half_width } => DrawShape::Segment {
                a: v32(a),
                b: v32(b),
                half_width: half_width as f32,
            },
            Primitive::Disc { centre, radius } => DrawShape::Disc {
                centre: v32(centre),
                radius: radius as f32,
            },
            Primitive::Arc {
                centre,
                radius,
                from,
                to,
                half_width,
            } => DrawShape::Polyline {
                points: (0..ARC_POINTS)
                    .map(|i| {
                        let t = from + (to - from) * i as f64 / (ARC_POINTS - 1) as f64;
                        v32(centre + Vec2::new(t.cos(), t.sin()) * radius)
                    })
                    .collect(),
                closed: false,
                half_width: half_width as f32,
            },
        };
        list.push(DrawOp::new(shape, ink)).ok()?;
    }
    for hole in target.set.holes.iter() {
        match hole.shape {
            Primitive::Disc { centre, radius } => list
                .push(DrawOp::new(
                    DrawShape::Disc {
                        centre: v32(centre),
                        radius: radius as f32,
                    },
                    paper,
                ))
                .ok()?,
            Primitive::HalfPlane { .. } => {
                if !target.set.parts.iter().all(|p| matches!(p.shape, Primitive::HalfPlane { .. })) {
                    return None;
                }
            }
            Primitive::Segment { .. } | Primitive::Arc { .. } => return None,
        }
    }
    Some(list)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QualityArm {
    Lights,
    Tiny,
}

impl QualityArm {
    pub fn name(self) -> &'static str {
        match self {
            QualityArm::Lights => "lights-drawlist",
            QualityArm::Tiny => "tiny-skia",
        }
    }
}

pub fn quality_image(arm: QualityArm, target: &Target<f64>, scale: f64) -> Option<FloatImage<f64>> {
    let list = target_list(target)?;
    let (w, h) = Mapping::at_scale(scale).device_viewport(target.viewport);
    let view = Similarity2::uniform(scale as f32);
    match arm {
        QualityArm::Lights => {
            let layer = list.render_through(view, w, h).ok()?;
            let paper = FloatImage::filled(w, h, Rgb::new(target.paper.x as f32, target.paper.y as f32, target.paper.z as f32));
            Some(layer.over(&paper).ok()?.to_f64())
        }
        QualityArm::Tiny => {
            let tiny = TinyList::from_list(&list, view)?;
            Some(pixmap_image(&render(&tiny, w, h, None)?))
        }
    }
}

pub fn quality(arm: QualityArm, target: &Target<f64>, scale: f64) -> Option<TargetMetric> {
    let image = quality_image(arm, target, scale)?;
    measure_target_mapped(&image, target, &Criterion::registered(), Mapping::at_scale(scale)).ok()
}
