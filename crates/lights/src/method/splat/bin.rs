use crate::extent::min_quadratic_over_rect;
use crate::kernel::Kernel;
use crate::method::splat::SortKind;
use crate::method::splat::project::ScreenAtom;
use crate::numerics::float::FloatingPoint;
use crate::numerics::symmetric::{LowerTri2, Sym2};
use crate::numerics::vector::Vec2;
use crate::tiling::TileGrid;

pub type SortKey = (u32, u64, u32);

pub const TILE_REJECTION_SLACK: f64 = 1.0e-4;

#[inline(always)]
pub fn whitened_quadratic_form<T: FloatingPoint>(l_inv: &LowerTri2<T>) -> Sym2<T> {
    Sym2::new(
        l_inv.a * l_inv.a + l_inv.b * l_inv.b,
        l_inv.b * l_inv.c,
        l_inv.c * l_inv.c,
    )
}

#[inline(always)]
pub fn depth_bits<T: FloatingPoint>(camera_z: T) -> u64 {
    camera_z.to_bits_ordered()
}

pub fn scan_tile_ranges(keys: &[SortKey], tile_count: usize, ranges: &mut Vec<(u32, u32)>) {
    ranges.clear();
    ranges.resize(tile_count, (0u32, 0u32));
    let mut start = 0usize;
    for i in 1..=keys.len() {
        if i == keys.len() || keys[i].0 != keys[start].0 {
            let tile = keys[start].0 as usize;
            if tile < tile_count {
                ranges[tile] = (start as u32, i as u32);
            }
            start = i;
        }
    }
}

pub fn bin_and_sort<T: FloatingPoint, K: Kernel<T>>(
    screen: &[ScreenAtom<T, K>],
    grid: &TileGrid,
    order_dependent: bool,
    sort: SortKind,
    keys: &mut Vec<SortKey>,
    ranges: &mut Vec<(u32, u32)>,
) -> u64 {
    keys.clear();
    for (slot, atom) in screen.iter().enumerate() {
        let min = Vec2::new(
            (atom.centre.x - atom.half_extent.x).to_f64(),
            (atom.centre.y - atom.half_extent.y).to_f64(),
        );
        let max = Vec2::new(
            (atom.centre.x + atom.half_extent.x).to_f64(),
            (atom.centre.y + atom.half_extent.y).to_f64(),
        );
        let span = match grid.span_of_box(min, max) {
            Some(s) => s,
            None => continue,
        };
        let bits = depth_bits(atom.depth);
        let form = whitened_quadratic_form(&atom.frame.l_inv);
        let reach = atom.support_sq * (T::ONE + T::from_f64(TILE_REJECTION_SLACK));
        for ty in span.ty0..span.ty1 {
            for tx in span.tx0..span.tx1 {
                let rect = grid.rect(tx, ty);
                let lo = Vec2::new(T::from_f64(rect.x0 as f64), T::from_f64(rect.y0 as f64));
                let hi = Vec2::new(
                    T::from_f64((rect.x0 + rect.w) as f64),
                    T::from_f64((rect.y0 + rect.h) as f64),
                );
                if min_quadratic_over_rect(form, atom.centre, lo, hi) > reach {
                    continue;
                }
                keys.push((grid.tile_id(tx, ty), bits, slot as u32));
            }
        }
    }

    if order_dependent {
        match sort {
            SortKind::GlobalSort => keys.sort_unstable(),
            SortKind::PerTileSort => {
                keys.sort_unstable_by_key(|k| k.0);
                let mut start = 0usize;
                for i in 1..=keys.len() {
                    if i == keys.len() || keys[i].0 != keys[start].0 {
                        keys[start..i].sort_unstable_by_key(|k| (k.1, k.2));
                        start = i;
                    }
                }
            }
        }
    } else {
        keys.sort_unstable_by_key(|k| (k.0, k.2));
    }

    scan_tile_ranges(keys, grid.tile_count(), ranges);
    keys.len() as u64
}
