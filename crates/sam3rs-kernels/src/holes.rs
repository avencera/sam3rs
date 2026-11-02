//! Hole filling of the point decoder's low-res masks: upstream's `postprocess_masks`
//! turns each 8-connected background region (logit <= 0, so NaN is foreground, as
//! upstream's `mask <= mask_threshold`) of at most `max_area` pixels into foreground (+10)
//!
//! Union-find over every mask of a batch at once, in four launches, after ECL-CC (Jaiganesh
//! and Burtscher): point each background pixel at its first earlier background neighbor,
//! union it with the rest, count each root's pixels, then fill the pixels of small roots.
//! A pixel's parent always has a smaller index, so every chain ends at a root and `find`
//! terminates. `find` also points each pixel it passes at its grandparent; without that,
//! chains across a large background region ran hundreds of pixels long and the merge took
//! 0.19 ms a mask. Which thread wins a race changes the trees but not the components, and
//! the counts are integers, so the output doesn't depend on scheduling. Pixel indices run
//! over the whole batch, `mask * side * side + y * side + x`, and neighbors never cross a
//! mask's edge

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

/// Label of a foreground pixel
const FOREGROUND: u32 = u32::MAX;

/// Whether a logit is background. NaN is foreground, as upstream reads it; the host fill
/// this replaced let a NaN pixel start a region but never join one, which depended on
/// scan order
#[inline(always)]
fn background(v: f32) -> bool {
    v <= 0.0
}

/// The root of `a`'s tree, pointing each pixel on the way at its grandparent (ECL-CC's
/// intermediate pointer jumping). A racing store can undo another thread's link, but only
/// on a pixel that thread found was no longer a root, so it goes on to join the trees
/// itself and no component splits
#[inline(always)]
fn find(labels: &[DeviceAtomicU32], mut a: u32) -> u32 {
    let mut parent = labels[a as usize].load(AtomicOrdering::Relaxed);
    loop {
        let next = labels[parent as usize].load(AtomicOrdering::Relaxed);
        if next >= parent {
            return parent;
        }
        labels[a as usize].store(next, AtomicOrdering::Relaxed);
        a = parent;
        parent = next;
    }
}

/// The root of `a`'s tree without shortening the path, for once the trees stop changing:
/// a shortcut store from a stale read could overwrite the root `holes_count` has just
/// written into a pixel with an ancestor that isn't one
#[inline(always)]
fn root(labels: &[DeviceAtomicU32], mut a: u32) -> u32 {
    loop {
        let parent = labels[a as usize].load(AtomicOrdering::Relaxed);
        if parent == a {
            return a;
        }
        a = parent;
    }
}

/// Joins the trees of `a` and `b`, hanging the larger root under the smaller (Playne and
/// Hawick's union-find). A lost race leaves the old parent detached, so the loop joins it
/// in turn
#[inline(always)]
fn union(labels: &[DeviceAtomicU32], mut a: u32, mut b: u32) {
    loop {
        a = find(labels, a);
        b = find(labels, b);
        if a == b {
            return;
        }
        let (hi, lo) = if a > b { (a, b) } else { (b, a) };
        let old = labels[hi as usize].fetch_min(lo, AtomicOrdering::Relaxed);
        if old == hi {
            return;
        }
        (a, b) = (old, lo);
    }
}

/// The earlier 8-connected neighbors of pixel `i` of `x [masks, side, side]` that are
/// background, smallest index first: north-west, north, north-east, west. Each pair of
/// neighbors is the earlier neighbor of exactly one of the two
#[inline(always)]
fn earlier_neighbors(x: &[f32], i: u32, side: u32) -> [u32; 4] {
    let (c, r) = (i % side, (i / side) % side);
    let mut out = [FOREGROUND; 4];
    let mut keep = |slot: usize, ok: bool, j: u32| {
        if ok && background(x[j as usize]) {
            out[slot] = j;
        }
    };
    if r > 0 {
        let up = i - side;
        keep(0, c > 0, up.wrapping_sub(1));
        keep(1, true, up);
        keep(2, c + 1 < side, up + 1);
    }
    keep(3, c > 0, i.wrapping_sub(1));
    out
}

/// A background pixel's label is its first earlier background neighbor, or itself, so
/// most of each region starts in one shallow tree; a foreground pixel's is [`FOREGROUND`]
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn holes_init(mut labels: DisjointSlice<u32>, x: &[f32], side: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut label) = labels.element_thread32(t) else { return };
    if !background(x[i as usize]) {
        label.write(FOREGROUND);
        return;
    }
    let near = earlier_neighbors(x, i, side);
    let mut first = i;
    for k in 0..near.len() {
        if near[k] < first {
            first = near[k];
        }
    }
    label.write(first);
}

/// Unions each background pixel with its other earlier background neighbors, which
/// covers every 8-connected pair once
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn holes_merge(labels: &[DeviceAtomicU32], x: &[f32], side: u32) {
    let i = thread::index_1d_u32(lc).get();
    if i as usize >= x.len() || !background(x[i as usize]) {
        return;
    }
    let near = earlier_neighbors(x, i, side);
    // the first background neighbor is already `i`'s parent from `holes_init`
    let mut linked = false;
    let mut join = |j: u32| {
        if j != FOREGROUND {
            if linked {
                union(labels, i, j);
            }
            linked = true;
        }
    };
    // constant indices keep `near` in registers; a loop over it went to local memory
    join(near[0]);
    join(near[1]);
    join(near[2]);
    join(near[3]);
}

/// Points each background pixel straight at its root and counts the root's pixels into
/// `area`, which starts zeroed
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn holes_count(labels: &[DeviceAtomicU32], area: &[DeviceAtomicU32]) {
    let i = thread::index_1d_u32(lc).get();
    if i as usize >= labels.len() || labels[i as usize].load(AtomicOrdering::Relaxed) == FOREGROUND {
        return;
    }
    let r = root(labels, i);
    // a concurrent `root` through `i` reads the old parent or the root, both on its path
    labels[i as usize].store(r, AtomicOrdering::Relaxed);
    area[r as usize].fetch_add(1, AtomicOrdering::Relaxed);
}

/// Sets each background pixel of a region of at most `max_area` pixels to +10, in place
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn holes_fill(mut x: DisjointSlice<f32>, labels: &[u32], area: &[u32], max_area: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get() as usize;
    let Some(mut x) = x.element_thread32(t) else { return };
    let label = labels[i];
    if label != FOREGROUND && area[label as usize] <= max_area {
        x.write(10.0);
    }
}
