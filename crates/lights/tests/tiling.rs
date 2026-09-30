use lights::numerics::vector::Vec2;
use lights::tiling::{DEFAULT_TILE, TileGrid, floor_div, for_each_tile};
use std::collections::BTreeSet;
use std::sync::Mutex;

#[test]
fn floor_div_is_a_mathematical_floor() {
    assert_eq!(floor_div(-5, 16), -1, "Rust's -5/16 truncates to 0, not -1");
    assert_eq!(-5 / 16, 0, "recording the language behaviour this replaces");
    assert_eq!(floor_div(-16, 16), -1);
    assert_eq!(floor_div(-17, 16), -2);
    assert_eq!(floor_div(0, 16), 0);
    assert_eq!(floor_div(15, 16), 0);
    assert_eq!(floor_div(16, 16), 1);
    assert_eq!(floor_div(31, 16), 1);
}

#[test]
fn tv_s18a_floor_div_on_negative_coordinates() {
    let grid = TileGrid::new(800, 600, DEFAULT_TILE).expect("grid");
    assert_eq!(grid.grid_x, 50);
    let span = grid
        .span_of_box(Vec2::new(-5.0, 4.0), Vec2::new(15.0, 12.0))
        .expect("overlaps the frame");
    assert_eq!(span.tx0, 0, "floor to -1 then CLAMP to 0");
    assert_eq!(span.tx1, 1);

    let mut differed = false;
    for x in -4000i32..4000 {
        let clamp_after_floor = floor_div(x, 16).clamp(0, grid.grid_x);
        let clamp_before_floor = floor_div(x.clamp(0, grid.width as i32), 16);
        if clamp_after_floor != clamp_before_floor {
            differed = true;
        }
    }
    assert!(
        !differed,
        "RECORDED, NOT ASSUMED: because span_of_box clamps TILE INDICES to \
         [0, grid] rather than clamping PIXEL coordinates, the floor/clamp \
         ordering is unobservable in this implementation. The math sheet's \
         hazard needs the truncating division as well, and that IS observable - \
         see the assertion below and floor_div_is_a_mathematical_floor."
    );

    let truncating_and_clamp_first = |x0: i32, x1: i32| {
        let t0 = i32::max(x0, 0) / 16;
        let t1 = (i32::max(x1, 0) - 1) / 16 + 1;
        t1 - t0
    };
    assert!(
        truncating_and_clamp_first(-40, -20) > 0,
        "the COMPOUND defect - truncating division AND clamping the pixel \
         coordinate before the floor - gives an off-screen-left atom a phantom \
         tile-0 column, where the shipped code gives none"
    );

    let straddling = grid
        .span_of_box(Vec2::new(-40.0, 4.0), Vec2::new(-20.0, 12.0));
    assert!(
        straddling.is_none(),
        "an atom entirely left of the frame yields no tiles, never a negative product"
    );
    let clamp_before_floor = floor_div(i32::max(-40, 0), 16);
    assert_eq!(
        clamp_before_floor, 0,
        "moving the clamp BEFORE the floor gives tile column 0 for an atom that \
         is entirely off-screen left: a phantom tile-0 column"
    );
}

#[test]
fn tv_s18b_rightmost_tile_column_included() {
    let grid = TileGrid::new(800, 600, DEFAULT_TILE).expect("grid");
    let span = grid
        .span_of_box(Vec2::new(797.0, 100.0), Vec2::new(801.0, 104.0))
        .expect("overlaps");
    assert_eq!(span.tx1, 50, "rect_max is an EXCLUSIVE ceil clamped to grid_x");
    assert_eq!(span.tx0, 49);

    let plain_floor = floor_div(801, 16);
    assert_eq!(
        plain_floor, 50,
        "a plain floor for the maximum drops the last tile column and row of \
         every atom, producing a 16-px-periodic grid of missing contributions"
    );
    let exclusive = floor_div(801 - 1, 16) + 1;
    assert_eq!(exclusive, 51);

    let single_pixel = grid
        .span_of_box(Vec2::new(0.2, 0.2), Vec2::new(0.4, 0.4))
        .expect("one pixel still covers one tile");
    assert_eq!((single_pixel.tx0, single_pixel.tx1), (0, 1));
}

#[test]
fn tv_s18c_offscreen_atom_yields_zero_tiles() {
    let grid = TileGrid::new(800, 600, DEFAULT_TILE).expect("grid");
    assert!(grid.span_of_box(Vec2::new(-100.0, 0.0), Vec2::new(-90.0, 10.0)).is_none());
    assert!(grid.span_of_box(Vec2::new(900.0, 0.0), Vec2::new(910.0, 10.0)).is_none());
    assert!(grid.span_of_box(Vec2::new(0.0, -50.0), Vec2::new(10.0, -40.0)).is_none());
    assert!(grid.span_of_box(Vec2::new(0.0, 700.0), Vec2::new(10.0, 710.0)).is_none());
    assert!(grid.span_of_box(Vec2::new(5.0, 5.0), Vec2::new(5.0, 5.0)).is_none());
    assert!(
        grid.span_of_box(Vec2::new(f64::NAN, 0.0), Vec2::new(10.0, 10.0))
            .is_none()
    );
    assert!(
        grid.span_of_box(Vec2::new(-1e30, -1e30), Vec2::new(1e30, 1e30))
            .is_some()
    );
}

#[test]
fn grid_geometry_and_edge_rectangles() {
    assert!(TileGrid::new(0, 10, 16).is_err());
    assert!(TileGrid::new(10, 0, 16).is_err());
    assert!(TileGrid::new(10, 10, 0).is_err());

    let grid = TileGrid::new(35, 20, 16).expect("grid");
    assert_eq!(grid.grid_x, 3);
    assert_eq!(grid.grid_y, 2);
    assert_eq!(grid.tile_count(), 6);
    assert_eq!(grid.tile_id(2, 1), 5);

    let full = grid.rect(0, 0);
    assert_eq!((full.w, full.h), (16, 16));
    let right_edge = grid.rect(2, 0);
    assert_eq!(right_edge.x0, 32);
    assert_eq!(right_edge.w, 3, "the edge tile is clipped to the image");
    let bottom_edge = grid.rect(0, 1);
    assert_eq!(bottom_edge.h, 4);

    let visited: Vec<(i32, i32)> = grid.tiles().collect();
    assert_eq!(visited.len(), 6);
    assert_eq!(visited[0], (0, 0));
    assert_eq!(visited[5], (2, 1));

    let mut total = 0;
    for (tx, ty) in grid.tiles() {
        let r = grid.rect(tx, ty);
        total += r.w * r.h;
    }
    assert_eq!(total, 35 * 20, "the tiles must exactly partition the image");
}

#[test]
fn for_each_tile_covers_every_tile_at_any_thread_count() {
    let grid = TileGrid::new(100, 70, 16).expect("grid");
    for threads in [1usize, 4, 18] {
        let seen = Mutex::new(BTreeSet::new());
        for_each_tile(&grid, threads, |rect| {
            let mut guard = match seen.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            assert!(
                guard.insert((rect.x0, rect.y0)),
                "a tile was visited twice at {threads} threads"
            );
        });
        let guard = match seen.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        assert_eq!(guard.len(), grid.tile_count());
    }
}
