# The corpus is cited here and not carried here

`docs/lights/CORPUS_READING_MAP.md` triages thirteen gaussian-splatting papers and the
research record cites them. **The PDFs themselves are not in this repository**: they are
third-party copyrighted works and redistributing them is not ours to do. Each is named in
the reading map with enough detail to find it.

This directory is not empty only because `crates/lights/tests/prose_pointers.rs` reads it:
its shelf check refuses to pass vacuously against an empty shelf, and it cross-checks the
three papers the map marks `— **not on disk**` against what is actually here. Those three
— Mip-Splatting, EWA Volume Splatting and GaussianImage — were never on disk in the
monorepo either, so the check reads the same in both places.
