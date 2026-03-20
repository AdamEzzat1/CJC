//! Clustering algorithms — K-Means, DBSCAN, Agglomerative.
//!
//! # Determinism Contract
//! - All floating-point reductions use `BinnedAccumulatorF64` (binned summation).
//! - No `HashMap`/`HashSet` — only `BTreeMap`/`BTreeSet` for deterministic iteration.
//! - RNG uses `cjc_repro::Rng` (SplitMix64) with explicit seeding.
//! - All algorithms produce bit-identical results for the same inputs and seed.

use crate::accumulator::{binned_sum_f64, BinnedAccumulatorF64};
use std::collections::BTreeSet;

// ---------------------------------------------------------------------------
// Helper: deterministic squared Euclidean distance
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two points (slices of length `n_features`).
/// Uses binned summation for deterministic accumulation.
#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(&ai, &bi)| {
        let d = ai - bi;
        d * d
    }).collect();
    binned_sum_f64(&diffs)
}

// ---------------------------------------------------------------------------
// 9A. K-Means with k-means++ initialization
// ---------------------------------------------------------------------------

/// K-Means clustering with k-means++ initialization.
///
/// Returns `(centroids, labels, inertia)` where:
/// - `centroids`: flat `Vec<f64>` of shape `[k, n_features]`
/// - `labels`: `Vec<usize>` of length `n_samples` (cluster assignment per sample)
/// - `inertia`: sum of squared distances from each sample to its nearest centroid
///
/// # Determinism
/// - Uses `cjc_repro::Rng` (SplitMix64) for k-means++ initialization.
/// - All distance reductions use `binned_sum_f64`.
/// - Assignment and update steps are fully deterministic.
pub fn kmeans(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<f64>, Vec<usize>, f64) {
    assert_eq!(data.len(), n_samples * n_features, "kmeans: data length mismatch");
    assert!(k > 0 && k <= n_samples, "kmeans: k must be in [1, n_samples]");

    let point = |i: usize| -> &[f64] {
        &data[i * n_features..(i + 1) * n_features]
    };

    // --- K-means++ initialization ---
    let mut rng = cjc_repro::Rng::seeded(seed);
    let mut centroids = Vec::with_capacity(k * n_features);

    // Pick first centroid uniformly at random
    let first = (rng.next_u64() as usize) % n_samples;
    centroids.extend_from_slice(point(first));

    // Pick remaining centroids with probability proportional to D(x)^2
    for _c in 1..k {
        // Compute distance from each point to nearest existing centroid
        let n_centroids = centroids.len() / n_features;
        let mut dist_sq = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let p = point(i);
            let mut min_d = f64::INFINITY;
            for j in 0..n_centroids {
                let c = &centroids[j * n_features..(j + 1) * n_features];
                let d = sq_dist(p, c);
                if d < min_d {
                    min_d = d;
                }
            }
            dist_sq.push(min_d);
        }

        // Cumulative sum for weighted sampling
        let total = binned_sum_f64(&dist_sq);
        if total == 0.0 {
            // All points are at existing centroids; pick randomly
            let idx = (rng.next_u64() as usize) % n_samples;
            centroids.extend_from_slice(point(idx));
            continue;
        }

        let threshold = rng.next_f64() * total;
        let mut cumulative = 0.0;
        let mut chosen = n_samples - 1;
        for i in 0..n_samples {
            cumulative += dist_sq[i];
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.extend_from_slice(point(chosen));
    }

    // --- Lloyd's algorithm ---
    let mut labels = vec![0usize; n_samples];

    for _iter in 0..max_iter {
        let old_labels = labels.clone();

        // Assignment step: assign each point to nearest centroid
        for i in 0..n_samples {
            let p = point(i);
            let mut best_k = 0;
            let mut best_d = f64::INFINITY;
            for j in 0..k {
                let c = &centroids[j * n_features..(j + 1) * n_features];
                let d = sq_dist(p, c);
                if d < best_d {
                    best_d = d;
                    best_k = j;
                }
            }
            labels[i] = best_k;
        }

        // Update step: recompute centroids as mean of assigned points
        let mut new_centroids = vec![0.0f64; k * n_features];
        let mut counts = vec![0usize; k];

        for i in 0..n_samples {
            let cluster = labels[i];
            counts[cluster] += 1;
            let p = point(i);
            for f in 0..n_features {
                // Collect per-feature values for binned summation at finalization
                new_centroids[cluster * n_features + f] += p[f];
            }
        }

        // Use binned summation for the centroid computation
        // We accumulated naively above; for full determinism we recompute per cluster
        for j in 0..k {
            if counts[j] == 0 {
                // Empty cluster: keep old centroid
                continue;
            }
            for f in 0..n_features {
                // Recompute with binned sum for determinism
                let mut acc = BinnedAccumulatorF64::new();
                for i in 0..n_samples {
                    if labels[i] == j {
                        acc.add(data[i * n_features + f]);
                    }
                }
                new_centroids[j * n_features + f] = acc.finalize() / counts[j] as f64;
            }
        }

        centroids = new_centroids;

        // Check convergence
        if labels == old_labels {
            break;
        }
    }

    // Compute inertia: sum of squared distances to assigned centroid
    let mut inertia_values = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let p = point(i);
        let c = &centroids[labels[i] * n_features..(labels[i] + 1) * n_features];
        inertia_values.push(sq_dist(p, c));
    }
    let inertia = binned_sum_f64(&inertia_values);

    (centroids, labels, inertia)
}

// ---------------------------------------------------------------------------
// 9B. DBSCAN
// ---------------------------------------------------------------------------

/// DBSCAN density-based clustering.
///
/// Returns `labels: Vec<i64>` where `-1` indicates a noise point.
///
/// # Determinism
/// - Neighbor lists use deterministic ordering (sequential index scan).
/// - No `HashMap` — only `BTreeSet` for visited/cluster tracking.
/// - Distance computations use `binned_sum_f64`.
pub fn dbscan(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    eps: f64,
    min_samples: usize,
) -> Vec<i64> {
    assert_eq!(data.len(), n_samples * n_features, "dbscan: data length mismatch");

    let eps_sq = eps * eps;

    let point = |i: usize| -> &[f64] {
        &data[i * n_features..(i + 1) * n_features]
    };

    // Precompute neighbors for each point (deterministic: sorted by index)
    let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let pi = point(i);
        let mut nbrs = Vec::new();
        for j in 0..n_samples {
            if sq_dist(pi, point(j)) <= eps_sq {
                nbrs.push(j);
            }
        }
        neighbors.push(nbrs);
    }

    let mut labels = vec![-1i64; n_samples];
    let mut visited = BTreeSet::new();
    let mut cluster_id: i64 = 0;

    for i in 0..n_samples {
        if visited.contains(&i) {
            continue;
        }
        visited.insert(i);

        if neighbors[i].len() < min_samples {
            // Noise point (may be claimed by a cluster later)
            continue;
        }

        // Start a new cluster
        labels[i] = cluster_id;

        // Expand cluster using a queue (deterministic: FIFO with sorted neighbors)
        let mut queue: Vec<usize> = neighbors[i].clone();
        let mut qi = 0;
        while qi < queue.len() {
            let q = queue[qi];
            qi += 1;

            if !visited.contains(&q) {
                visited.insert(q);
                if neighbors[q].len() >= min_samples {
                    // Add q's neighbors to the queue
                    for &nb in &neighbors[q] {
                        if !visited.contains(&nb) || labels[nb] == -1 {
                            if !queue.contains(&nb) {
                                queue.push(nb);
                            }
                        }
                    }
                }
            }

            if labels[q] == -1 {
                labels[q] = cluster_id;
            }
        }

        cluster_id += 1;
    }

    labels
}

// ---------------------------------------------------------------------------
// 9C. Agglomerative (Hierarchical) Clustering
// ---------------------------------------------------------------------------

/// Hierarchical agglomerative clustering.
///
/// Supported linkage methods: `"single"`, `"complete"`, `"average"`, `"ward"`.
///
/// Returns `labels: Vec<usize>` with cluster assignments `[0, n_clusters)`.
///
/// # Determinism
/// - Distance matrix uses `binned_sum_f64` for all computations.
/// - Merge order is deterministic: always picks the lexicographically smallest
///   `(distance, i, j)` pair.
/// - No `HashMap` — uses `BTreeSet` for active cluster tracking.
pub fn agglomerative(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
    linkage: &str,
) -> Vec<usize> {
    assert_eq!(data.len(), n_samples * n_features, "agglomerative: data length mismatch");
    assert!(n_clusters > 0 && n_clusters <= n_samples, "agglomerative: invalid n_clusters");

    let point = |i: usize| -> &[f64] {
        &data[i * n_features..(i + 1) * n_features]
    };

    // Build initial distance matrix (upper triangle, indexed as dist[i][j] for i < j)
    // Using a flat vector for the condensed distance matrix.
    let n = n_samples;

    // Index into condensed distance matrix: for i < j, index = i*n - i*(i+1)/2 + j - i - 1
    let condensed_idx = |i: usize, j: usize| -> usize {
        debug_assert!(i < j);
        i * n - i * (i + 1) / 2 + j - i - 1
    };
    let condensed_len = n * (n - 1) / 2;

    let mut dist = vec![0.0f64; condensed_len];
    for i in 0..n {
        for j in (i + 1)..n {
            dist[condensed_idx(i, j)] = sq_dist(point(i), point(j));
        }
    }

    // For Ward linkage, we need sqrt of distances for proper Ward update
    // but we can work with squared distances and use the Lance-Williams formula.

    // Track which clusters are active and their sizes
    let mut active = BTreeSet::new();
    for i in 0..n {
        active.insert(i);
    }
    let mut sizes = vec![1usize; n];

    // Each sample starts as its own cluster; we'll track merges via a union-find
    let mut parent = vec![0usize; 2 * n]; // up to n-1 merges create new nodes
    for i in 0..n {
        parent[i] = i;
    }
    let mut next_cluster = n;

    // Map from original cluster IDs to the merged cluster ID
    let mut cluster_map: Vec<usize> = (0..n).collect();

    // Perform n - n_clusters merges
    for _ in 0..(n - n_clusters) {
        // Find the pair with minimum distance
        let mut best_dist = f64::INFINITY;
        let mut best_i = 0;
        let mut best_j = 0;

        let active_vec: Vec<usize> = active.iter().copied().collect();
        for ai in 0..active_vec.len() {
            for aj in (ai + 1)..active_vec.len() {
                let ci = active_vec[ai];
                let cj = active_vec[aj];
                let (lo, hi) = if ci < cj { (ci, cj) } else { (cj, ci) };
                let d = dist[condensed_idx(lo, hi)];
                if d < best_dist || (d == best_dist && (lo, hi) < (best_i, best_j)) {
                    best_dist = d;
                    best_i = lo;
                    best_j = hi;
                }
            }
        }

        // Merge best_i and best_j into a new cluster
        let new_id = next_cluster;
        next_cluster += 1;
        let size_i = sizes[best_i];
        let size_j = sizes[best_j];
        let size_new = size_i + size_j;

        // Update cluster_map: all samples in best_i and best_j now map to new_id
        for s in 0..n {
            if cluster_map[s] == best_i || cluster_map[s] == best_j {
                cluster_map[s] = new_id;
            }
        }

        // Remove old clusters, add new one
        active.remove(&best_i);
        active.remove(&best_j);

        // Compute distances from the new cluster to all remaining active clusters
        // using the Lance-Williams formula
        // We need to expand the dist vector to accommodate the new cluster index.
        // Since new_id >= n, we need a different approach.
        // Use a secondary distance store for new clusters.
        // Actually, let's use a simpler approach: maintain a full n_samples x n_samples
        // approach won't scale, but for correctness we recompute from the merged points.

        // For each remaining active cluster, compute distance to new cluster
        // We temporarily store new distances by reusing the condensed matrix.
        // We'll extend it to support new indices.

        // Simpler approach: extend dist to handle new cluster IDs
        // New index pairs: (min(active_c, new_id), max(active_c, new_id))
        // Since new_id > all original IDs, we always have (active_c, new_id)
        // But condensed_idx assumes max index < n. So we use a separate map.

        // Let's use a Vec to store distances for new clusters.
        // Actually the simplest correct approach: for each remaining cluster,
        // compute new distance using Lance-Williams and store back.
        // We'll reassign best_i's slot to the new cluster.

        // Reassign: new cluster takes best_i's position in the condensed matrix
        // (best_j is removed)
        // This avoids growing the distance matrix.
        let new_slot = best_i;
        sizes.push(0); // pad if needed
        while sizes.len() <= new_id {
            sizes.push(0);
        }
        sizes[new_id] = size_new;
        // Also keep sizes[new_slot] updated for the condensed matrix lookups
        sizes[new_slot] = size_new;

        for &c in &active {
            let (lo_i, hi_i) = if c < best_i { (c, best_i) } else { (best_i, c) };
            let (lo_j, hi_j) = if c < best_j { (c, best_j) } else { (best_j, c) };
            let d_ci = dist[condensed_idx(lo_i, hi_i)];
            let d_cj = dist[condensed_idx(lo_j, hi_j)];

            let new_dist = match linkage {
                "single" => d_ci.min(d_cj),
                "complete" => d_ci.max(d_cj),
                "average" => {
                    // Weighted average by cluster size
                    (size_i as f64 * d_ci + size_j as f64 * d_cj) / size_new as f64
                }
                "ward" => {
                    // Lance-Williams formula for Ward's method (using squared distances)
                    let size_c = sizes[c] as f64;
                    let sn = size_new as f64;
                    let si = size_i as f64;
                    let sj = size_j as f64;
                    ((size_c + si) * d_ci + (size_c + sj) * d_cj - size_c * best_dist)
                        / (size_c + sn)
                }
                _ => panic!("agglomerative: unsupported linkage '{}'. Use single, complete, average, or ward.", linkage),
            };

            // Store in the condensed matrix at new_slot's position
            let (lo, hi) = if c < new_slot { (c, new_slot) } else { (new_slot, c) };
            dist[condensed_idx(lo, hi)] = new_dist;
        }

        // Map cluster_map entries from new_id to new_slot
        for s in 0..n {
            if cluster_map[s] == new_id {
                cluster_map[s] = new_slot;
            }
        }

        active.insert(new_slot);
    }

    // Remap cluster_map to contiguous labels [0, n_clusters)
    let unique_clusters: BTreeSet<usize> = cluster_map.iter().copied().collect();
    let label_map: Vec<(usize, usize)> = unique_clusters.into_iter().enumerate().map(|(i, c)| (c, i)).collect();

    let mut labels = vec![0usize; n_samples];
    for i in 0..n_samples {
        for &(c, l) in &label_map {
            if cluster_map[i] == c {
                labels[i] = l;
                break;
            }
        }
    }

    labels
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- K-Means tests ---

    #[test]
    fn test_kmeans_two_clusters() {
        // Two well-separated clusters in 2D
        // Cluster 0: points around (0, 0)
        // Cluster 1: points around (10, 10)
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            -0.1, 0.1,
            0.1, -0.1,
            -0.1, -0.1,
            10.0, 10.0,
            10.1, 10.1,
            9.9, 10.1,
            10.1, 9.9,
            9.9, 9.9,
        ];
        let n_samples = 5 + 5;
        let n_features = 2;
        let k = 2;

        let (centroids, labels, inertia) = kmeans(&data, n_samples, n_features, k, 100, 42);

        // Verify we get exactly 2 distinct labels
        let unique: BTreeSet<usize> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 2);

        // Verify the first 5 points have the same label
        assert!(labels[0..5].iter().all(|&l| l == labels[0]));
        // Verify the last 5 points have the same label
        assert!(labels[5..10].iter().all(|&l| l == labels[5]));
        // And they're different clusters
        assert_ne!(labels[0], labels[5]);

        // Centroids should be 2 * n_features = 4 values
        assert_eq!(centroids.len(), k * n_features);

        // Inertia should be small (points are tight around centroids)
        assert!(inertia < 1.0, "inertia too large: {}", inertia);
    }

    #[test]
    fn test_kmeans_deterministic() {
        let data = vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ];
        let (c1, l1, i1) = kmeans(&data, 6, 2, 2, 50, 123);
        let (c2, l2, i2) = kmeans(&data, 6, 2, 2, 50, 123);
        assert_eq!(l1, l2);
        assert_eq!(c1, c2);
        assert_eq!(i1.to_bits(), i2.to_bits());
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (_centroids, labels, _inertia) = kmeans(&data, 3, 2, 1, 10, 0);
        assert!(labels.iter().all(|&l| l == 0));
    }

    // --- DBSCAN tests ---

    #[test]
    fn test_dbscan_two_clusters_with_noise() {
        // Cluster A: tight group around (0, 0)
        // Cluster B: tight group around (10, 10)
        // Noise point at (50, 50)
        let data = vec![
            0.0, 0.0,
            0.1, 0.0,
            0.0, 0.1,
            0.1, 0.1,
            10.0, 10.0,
            10.1, 10.0,
            10.0, 10.1,
            10.1, 10.1,
            50.0, 50.0, // noise
        ];
        let n_samples = 9;
        let n_features = 2;

        let labels = dbscan(&data, n_samples, n_features, 0.5, 2);

        // First 4 points should be in one cluster
        assert!(labels[0] >= 0);
        assert!(labels[0..4].iter().all(|&l| l == labels[0]));

        // Next 4 points should be in another cluster
        assert!(labels[4] >= 0);
        assert!(labels[4..8].iter().all(|&l| l == labels[4]));

        // The two clusters should be different
        assert_ne!(labels[0], labels[4]);

        // Last point should be noise
        assert_eq!(labels[8], -1);
    }

    #[test]
    fn test_dbscan_all_noise() {
        // Points too far apart for any cluster
        let data = vec![0.0, 0.0, 100.0, 100.0, 200.0, 200.0];
        let labels = dbscan(&data, 3, 2, 0.5, 2);
        assert!(labels.iter().all(|&l| l == -1));
    }

    #[test]
    fn test_dbscan_single_cluster() {
        // All points within eps of each other
        let data = vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1];
        let labels = dbscan(&data, 4, 2, 0.5, 2);
        assert!(labels.iter().all(|&l| l == labels[0] && l >= 0));
    }

    // --- Agglomerative tests ---

    #[test]
    fn test_agglomerative_two_clusters_single() {
        // Two well-separated groups
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            0.2, 0.0,
            10.0, 10.0,
            10.1, 10.1,
            10.2, 10.0,
        ];
        let labels = agglomerative(&data, 6, 2, 2, "single");

        // First 3 and last 3 should be in different clusters
        assert!(labels[0..3].iter().all(|&l| l == labels[0]));
        assert!(labels[3..6].iter().all(|&l| l == labels[3]));
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_agglomerative_complete_linkage() {
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            10.0, 10.0,
            10.1, 10.1,
        ];
        let labels = agglomerative(&data, 4, 2, 2, "complete");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_agglomerative_average_linkage() {
        let data = vec![
            0.0, 0.0,
            1.0, 0.0,
            20.0, 0.0,
            21.0, 0.0,
        ];
        let labels = agglomerative(&data, 4, 2, 2, "average");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_agglomerative_ward_linkage() {
        let data = vec![
            0.0, 0.0,
            0.5, 0.0,
            10.0, 10.0,
            10.5, 10.0,
        ];
        let labels = agglomerative(&data, 4, 2, 2, "ward");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_agglomerative_n_clusters_equals_n() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let labels = agglomerative(&data, 4, 1, 4, "single");
        // Each point is its own cluster
        let unique: BTreeSet<usize> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn test_agglomerative_single_cluster() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let labels = agglomerative(&data, 4, 1, 1, "single");
        assert!(labels.iter().all(|&l| l == 0));
    }

    // --- Determinism tests ---

    #[test]
    fn test_kmeans_different_seeds_may_differ() {
        let data = vec![
            0.0, 0.0, 1.0, 1.0, 2.0, 2.0,
            10.0, 10.0, 11.0, 11.0, 12.0, 12.0,
        ];
        let (_, l1, _) = kmeans(&data, 6, 2, 2, 50, 1);
        let (_, l2, _) = kmeans(&data, 6, 2, 2, 50, 2);
        // Both should produce valid 2-cluster partitions
        let u1: BTreeSet<usize> = l1.iter().copied().collect();
        let u2: BTreeSet<usize> = l2.iter().copied().collect();
        assert_eq!(u1.len(), 2);
        assert_eq!(u2.len(), 2);
    }

    #[test]
    fn test_dbscan_deterministic() {
        let data = vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1,
            10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
        ];
        let l1 = dbscan(&data, 6, 2, 0.5, 2);
        let l2 = dbscan(&data, 6, 2, 0.5, 2);
        assert_eq!(l1, l2);
    }

    #[test]
    fn test_agglomerative_deterministic() {
        let data = vec![
            0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1,
        ];
        let l1 = agglomerative(&data, 4, 2, 2, "ward");
        let l2 = agglomerative(&data, 4, 2, 2, "ward");
        assert_eq!(l1, l2);
    }
}
