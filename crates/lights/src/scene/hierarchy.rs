use crate::camera::Camera;
use crate::numerics::float::FloatingPoint;
use crate::numerics::quaternion::Quat;
use crate::numerics::symmetric::Sym3;
use crate::numerics::vector::Vec3;
use crate::scene::Atom;

pub const MAX_TREE_DEPTH: usize = 24;

pub fn atom_mass<T: FloatingPoint>(atom: &Atom<T>) -> T {
    atom.opacity * atom.scale.x * atom.scale.y * atom.scale.z
}

fn weighted_spread<T: FloatingPoint>(sigma: &Sym3<T>, d: Vec3<T>, w: T) -> Sym3<T> {
    Sym3::new(
        (sigma.xx + d.x * d.x) * w,
        (sigma.xy + d.x * d.y) * w,
        (sigma.xz + d.x * d.z) * w,
        (sigma.yy + d.y * d.y) * w,
        (sigma.yz + d.y * d.z) * w,
        (sigma.zz + d.z * d.z) * w,
    )
}

fn sym_add<T: FloatingPoint>(a: &Sym3<T>, b: &Sym3<T>) -> Sym3<T> {
    Sym3::new(
        a.xx + b.xx,
        a.xy + b.xy,
        a.xz + b.xz,
        a.yy + b.yy,
        a.yz + b.yz,
        a.zz + b.zz,
    )
}

pub fn moment_merge<T: FloatingPoint>(atoms: &[Atom<T>]) -> Option<Atom<T>> {
    let mut total = T::ZERO;
    let mut first = Vec3::zero();
    let mut colour = Vec3::zero();
    for atom in atoms {
        let m = atom_mass(atom);
        total = total + m;
        first += atom.mean * m;
        colour += atom.colour * m;
    }
    if !(total > T::ZERO) || !total.is_finite() {
        return None;
    }
    let mean = first / total;
    let mut second = Sym3::zero();
    for atom in atoms {
        let w = atom_mass(atom) / total;
        second = sym_add(&second, &weighted_spread(&atom.covariance(), atom.mean - mean, w));
    }
    let (values, vectors) = second.eigen()?;
    if !(values.min_component() > T::ZERO) {
        return None;
    }
    let rotation = Quat::from_rotation_matrix(&vectors)?;
    let scale = Vec3::new(values.x.sqrt(), values.y.sqrt(), values.z.sqrt());
    Some(Atom {
        mean,
        rotation,
        scale,
        opacity: total / (scale.x * scale.y * scale.z),
        colour: colour / total,
    })
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Node<T> {
    summary: Atom<T>,
    start: usize,
    end: usize,
    first_child: usize,
    child_count: usize,
}

#[derive(Clone, Debug)]
pub struct MomentTree<T> {
    atoms: Vec<Atom<T>>,
    nodes: Vec<Node<T>>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CutStats {
    pub leaves: usize,
    pub summaries: usize,
    pub summarised_atoms: usize,
}

impl<T: FloatingPoint> MomentTree<T> {
    pub fn build(atoms: &[Atom<T>]) -> Option<Self> {
        let mut tree = Self {
            atoms: atoms.to_vec(),
            nodes: Vec::new(),
        };
        if tree.atoms.is_empty() {
            return None;
        }
        tree.nodes.push(Node {
            summary: atoms[0],
            start: 0,
            end: atoms.len(),
            first_child: 0,
            child_count: 0,
        });
        tree.split(0, 0)?;
        Some(tree)
    }

    fn split(&mut self, index: usize, depth: usize) -> Option<()> {
        let (start, end) = (self.nodes[index].start, self.nodes[index].end);
        self.nodes[index].summary = moment_merge(&self.atoms[start..end])?;
        if end - start <= 1 || depth >= MAX_TREE_DEPTH {
            return Some(());
        }
        let mut lo = self.atoms[start].mean;
        let mut hi = lo;
        for atom in &self.atoms[start..end] {
            lo = Vec3::new(lo.x.min(atom.mean.x), lo.y.min(atom.mean.y), lo.z.min(atom.mean.z));
            hi = Vec3::new(hi.x.max(atom.mean.x), hi.y.max(atom.mean.y), hi.z.max(atom.mean.z));
        }
        let centre = (lo + hi) * T::from_f64(0.5);
        let octant = |a: &Atom<T>| -> usize {
            usize::from(a.mean.x > centre.x)
                + 2 * usize::from(a.mean.y > centre.y)
                + 4 * usize::from(a.mean.z > centre.z)
        };
        self.atoms[start..end].sort_by_key(|a| octant(a));
        let mut counts = [0usize; 8];
        for atom in &self.atoms[start..end] {
            counts[octant(atom)] += 1;
        }
        let mut ranges = Vec::new();
        let mut cursor = start;
        for count in counts {
            if count > 0 {
                ranges.push((cursor, cursor + count));
                cursor += count;
            }
        }
        if ranges.len() <= 1 {
            return Some(());
        }
        let first_child = self.nodes.len();
        self.nodes[index].first_child = first_child;
        self.nodes[index].child_count = ranges.len();
        for &(begin, finish) in &ranges {
            self.nodes.push(Node {
                summary: self.atoms[begin],
                start: begin,
                end: finish,
                first_child: 0,
                child_count: 0,
            });
        }
        for child in first_child..first_child + ranges.len() {
            self.split(child, depth + 1)?;
        }
        Some(())
    }

    pub fn atoms(&self) -> &[Atom<T>] {
        &self.atoms
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn root_summary(&self) -> Atom<T> {
        self.nodes[0].summary
    }

    pub fn footprint_px(camera: &Camera<T>, atom: &Atom<T>) -> Option<T> {
        let cam = camera.to_camera(atom.mean);
        camera
            .screen_disc_bound(cam, atom.scale.max_component())
            .map(|(_, r)| r)
    }

    pub fn cut(&self, camera: &Camera<T>, threshold_px: T) -> (Vec<Atom<T>>, CutStats) {
        let mut out = Vec::new();
        let mut stats = CutStats::default();
        self.descend(0, camera, threshold_px, &mut out, &mut stats);
        (out, stats)
    }

    fn descend(
        &self,
        index: usize,
        camera: &Camera<T>,
        threshold_px: T,
        out: &mut Vec<Atom<T>>,
        stats: &mut CutStats,
    ) {
        let node = &self.nodes[index];
        let span = node.end - node.start;
        if node.child_count == 0 && span == 1 {
            out.push(self.atoms[node.start]);
            stats.leaves += 1;
            return;
        }
        let small = Self::footprint_px(camera, &node.summary)
            .is_some_and(|r| r < threshold_px);
        if small {
            out.push(node.summary);
            stats.summaries += 1;
            stats.summarised_atoms += span;
            return;
        }
        if node.child_count == 0 {
            out.extend_from_slice(&self.atoms[node.start..node.end]);
            stats.leaves += span;
            return;
        }
        for child in node.first_child..node.first_child + node.child_count {
            self.descend(child, camera, threshold_px, out, stats);
        }
    }
}
