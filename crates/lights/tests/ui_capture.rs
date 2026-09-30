use lights::{CAPTURE_COLUMNS, Capture, DrawShape, Ink, LightsError, Rgb, Stand, Vec2};

fn v(x: f64, y: f64) -> Vec2<f64> {
    Vec2::new(x, y)
}

fn capture(rows: &[&str]) -> String {
    let mut text = String::from("#scene\ttest\n");
    text.push_str(&CAPTURE_COLUMNS.join("\t"));
    text.push('\n');
    for r in rows {
        text.push_str(&r.split_whitespace().collect::<Vec<_>>().join("\t"));
        text.push('\n');
    }
    text
}

#[allow(clippy::too_many_arguments)]
fn row(shape: usize, kind: &str, geom: &str, radius: &str, width: &str, blur: &str, align: &str, closed: &str, rgba: &str, textured: &str, points: &str) -> String {
    format!("{shape} {kind} 0 0 100 50 {geom} {radius} {width} {blur} {align} {closed} {rgba} {textured} {points}")
}

fn rect_fill(shape: usize, geom: &str) -> String {
    row(shape, "rect_fill", geom, "0,0,0,0", "-", "0", "-", "-", "255,0,0,255", "0", "-")
}

#[test]
fn every_captured_kind_becomes_the_draw_op_it_names_and_says_how_it_stands() {
    let rows = [
        row(0, "rect_fill", "10 10 30 20", "2,4,4,2", "-", "0", "-", "-", "255,0,0,255", "0", "-"),
        row(0, "rect_stroke", "10 10 30 20", "3,3,3,3", "2", "-", "inside", "-", "0,0,255,255", "0", "-"),
        row(1, "rect_stroke", "10 10 30 20", "3,3,3,3", "2", "-", "outside", "-", "0,0,255,255", "0", "-"),
        row(2, "rect_stroke", "10 10 30 20", "0,0,0,0", "2", "-", "middle", "-", "0,0,255,255", "0", "-"),
        row(3, "circle_fill", "40 25 40 25", "5", "-", "-", "-", "-", "0,255,0,128", "0", "-"),
        row(4, "circle_stroke", "40 25 40 25", "5", "1.5", "-", "middle", "-", "0,255,0,255", "0", "-"),
        row(5, "line", "1 2 3 4", "-", "2", "-", "middle", "-", "10,20,30,255", "0", "-"),
        row(6, "path_fill", "0 0 10 10", "-", "-", "-", "-", "1", "255,255,255,255", "0", "0,0;10,0;10,10"),
        row(6, "path_stroke", "0 0 10 10", "-", "1", "-", "middle", "0", "uv", "0", "0,0;10,0;10,10"),
        row(7, "glyph", "12 30 18 41", "-", "-", "-", "-", "-", "255,255,255,255", "0", "-"),
        row(8, "mesh_triangle", "0 0 5 5", "-", "-", "-", "-", "-", "40,40,40,255", "1", "0,0;5,0;0,5"),
        row(9, "rect_fill", "10 10 30 20", "0,0,0,0", "-", "4", "-", "-", "0,0,0,64", "0", "-"),
        row(10, "ellipse_fill", "40 25 40 25", "5,3", "-", "-", "-", "-", "0,0,0,255", "0", "-"),
        row(11, "callback", "0 0 10 10", "-", "-", "-", "-", "-", "-", "0", "-"),
    ];
    let refs: Vec<&str> = rows.iter().map(String::as_str).collect();
    let c = Capture::<f64>::parse(&capture(&refs)).unwrap();
    assert_eq!(c.header.get("scene").map(String::as_str), Some("test"));
    assert_eq!(c.ops.iter().map(|o| o.shape).collect::<Vec<_>>(), [0, 0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11]);
    let stands: Vec<Stand> = c.ops.iter().map(|o| o.stand).collect();
    let (n, p, a) = (Stand::Native, Stand::Proxy, Stand::Absent);
    assert_eq!(stands, [n, n, n, n, n, n, n, n, n, p, p, p, a, a]);
    let shape = |i: usize| c.ops[i].op.as_ref().map(|o| o.shape.clone());
    let outline = |lo: f64, hi_x: f64, hi_y: f64, r: f64| DrawShape::RectOutline {
        min: v(lo, lo),
        max: v(hi_x, hi_y),
        corner_radius: r,
        half_width: 1.0,
    };
    assert_eq!(shape(0), Some(DrawShape::Rect { min: v(10.0, 10.0), max: v(30.0, 20.0), corner_radius: 4.0 }));
    assert_eq!(shape(1), Some(outline(11.0, 29.0, 19.0, 2.0)), "an inside stroke is centred half its width inside the rect");
    assert_eq!(shape(2), Some(outline(9.0, 31.0, 21.0, 4.0)), "an outside stroke is centred half its width outside it");
    assert_eq!(shape(3), Some(outline(10.0, 30.0, 20.0, 0.0)), "a middle stroke is centred on the edge");
    assert_eq!(shape(4), Some(DrawShape::Disc { centre: v(40.0, 25.0), radius: 5.0 }));
    assert_eq!(shape(5), Some(DrawShape::Ring { centre: v(40.0, 25.0), radius: 5.0, half_width: 0.75 }));
    assert_eq!(shape(6), Some(DrawShape::Segment { a: v(1.0, 2.0), b: v(3.0, 4.0), half_width: 1.0 }));
    let tri = vec![v(0.0, 0.0), v(10.0, 0.0), v(10.0, 10.0)];
    assert_eq!(shape(7), Some(DrawShape::Polygon { points: tri.clone() }));
    assert_eq!(shape(8), Some(DrawShape::Polyline { points: tri, closed: false, half_width: 0.5 }));
    assert_eq!(shape(9), Some(DrawShape::Rect { min: v(12.0, 30.0), max: v(18.0, 41.0), corner_radius: 0.0 }));
    assert_eq!(shape(10), Some(DrawShape::Polygon { points: vec![v(0.0, 0.0), v(5.0, 0.0), v(0.0, 5.0)] }));
    assert_eq!(
        shape(11),
        Some(DrawShape::Rect { min: v(8.0, 8.0), max: v(32.0, 22.0), corner_radius: 0.0 }),
        "a blurred rect stands in as a flat rect out to half its blur"
    );
    assert!(c.ops[12].op.is_none() && c.ops[13].op.is_none(), "an absent kind carries no op");
    let first = c.ops[0].op.as_ref().unwrap();
    assert_eq!(first.ink, Ink::premultiplied(Rgb::new(1.0, 0.0, 0.0), 1.0), "a full channel is exactly one");
    assert_eq!(first.clip.map(|k| (k.min, k.max)), Some((v(0.0, 0.0), v(100.0, 50.0))));
    assert_eq!(c.ops[4].op.as_ref().unwrap().ink.alpha, 128.0 / 255.0);
    assert_eq!(c.ops[8].op.as_ref().unwrap().ink, Ink::opaque(Rgb::splat(0.5)), "a uv colour stands in as mid grey");
}

#[test]
fn a_malformed_capture_is_refused_at_the_line_that_breaks_it() {
    let good = rect_fill(0, "10 10 30 20");
    let line_of = |text: &str| match Capture::<f64>::parse(text) {
        Err(LightsError::CaptureFormat { line, .. }) => line,
        other => panic!("expected a refusal, got {other:?}"),
    };
    assert!(Capture::<f64>::parse(&capture(&[&good])).is_ok());
    assert_eq!(line_of(&capture(&[&good]).replacen("shape\t", "index\t", 1)), 2, "a foreign column header");
    assert_eq!(line_of(&capture(&[&good, "0 rect_fill 0 0"])), 4, "a short row");
    assert_eq!(line_of(&capture(&[&good.replacen("10 10 30 20", "10 ten 30 20", 1)])), 3, "a word where a number goes");
    assert_eq!(line_of(&capture(&[&good.replacen("rect_fill", "sprite", 1)])), 3, "an unknown kind");
    let sideways = row(0, "rect_stroke", "10 10 30 20", "0,0,0,0", "2", "-", "left", "-", "0,0,0,255", "0", "-");
    assert_eq!(line_of(&capture(&[&sideways])), 3, "an unknown alignment");
    assert_eq!(line_of(&capture(&[&good.replacen("255,0,0,255", "255,0,0", 1)])), 3, "three channels");
    assert_eq!(line_of("#scene\tempty\n"), 1, "no column header at all");
}

#[test]
fn a_list_admits_the_stands_it_is_asked_for_in_capture_order_and_counts_what_it_refuses() {
    let rows = [
        rect_fill(0, "0 0 10 10"),
        row(1, "glyph", "1 1 4 6", "-", "-", "-", "-", "-", "255,255,255,255", "0", "-"),
        rect_fill(2, "30 10 20 20"),
        row(3, "circle_fill", "5 5 5 5", "2", "-", "-", "-", "-", "0,0,0,255", "0", "-"),
        row(4, "bezier", "0 0 10 10", "-", "-", "-", "-", "0", "0,0,0,255", "0", "0,0;5,10;10,0"),
    ];
    let refs: Vec<&str> = rows.iter().map(String::as_str).collect();
    let c = Capture::<f32>::parse(&capture(&refs)).unwrap();
    let (native, refused) = c.draw_list(&[Stand::Native]);
    assert_eq!(native.len(), 2, "the inverted rect is refused and the glyph is not native");
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert!(refused[0].starts_with("shape 2 rect_fill"), "{refused:?}");
    assert!(matches!(native.ops()[1].shape, DrawShape::Disc { .. }));
    let (framed, _) = c.draw_list(&[Stand::Native, Stand::Proxy]);
    assert_eq!(framed.len(), 3);
    assert!(matches!(framed.ops()[1].shape, DrawShape::Rect { .. }), "the glyph proxy keeps its place in paint order");
    let (absent, none) = c.draw_list(&[Stand::Absent]);
    assert!(absent.is_empty() && none.is_empty(), "an absent kind has nothing to draw and nothing to refuse");
}
