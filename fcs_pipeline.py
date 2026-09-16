"""
False Contour Suppression via Active Gradient Restoration
Based on the paper's theoretical foundations and proposed method.

Target platform: Raspberry Pi 5 (ARM Cortex-A76, 4 cores, 8 GB RAM)
No GPU/NPU acceleration - pure NumPy implementation.
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Set, Optional
import math


# =============================================================================
# Utility Functions
# =============================================================================

def quantize_image(f: np.ndarray, n: int) -> np.ndarray:
    """
    Direct quantization: discard n least significant bits.
    Eq. (1): F(i,j) = floor(f(i,j) / 2^n) * 2^n
    """
    step = 1 << n
    return (f // step) * step


def get_8_neighbors(i: int, j: int, H: int, W: int) -> List[Tuple[int, int]]:
    """Return 8-connected neighbors within image bounds."""
    neighbors = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                neighbors.append((ni, nj))
    return neighbors


def get_4_neighbors(i: int, j: int, H: int, W: int) -> List[Tuple[int, int]]:
    """Return 4-connected neighbors within image bounds."""
    neighbors = []
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < H and 0 <= nj < W:
            neighbors.append((ni, nj))
    return neighbors


def local_mean(F: np.ndarray, e: int) -> np.ndarray:
    """
    Compute local mean within e x e window using integral image (fast, cache-friendly).
    """
    H, W = F.shape
    r = e // 2
    # Pad F for boundary handling (replicate)
    Fp = np.pad(F.astype(np.float64), r, mode='edge')
    # Integral image of padded array
    Ip = np.zeros((H + 2 * r + 1, W + 2 * r + 1), dtype=np.float64)
    Ip[1:, 1:] = np.cumsum(np.cumsum(Fp, axis=0), axis=1)

    # Window sums via integral image
    Ii, Jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    S = (Ip[Ii + 2 * r + 1, Jj + 2 * r + 1]
         - Ip[Ii, Jj + 2 * r + 1]
         - Ip[Ii + 2 * r + 1, Jj]
         + Ip[Ii, Jj])
    count = (2 * r + 1) ** 2
    return S / count


# =============================================================================
# Stage 1: False Contour Region Detection
# =============================================================================

def variable_thresholding(F: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Variable thresholding.
    Eq. (2)-(3): g(i,j) = 1 if F(i,j) == local mean m_{i,j}, else 0.
    Adaptive window size s = 2*floor(sqrt(W*H)/150) + 3.
    """
    H, W = F.shape
    s = 2 * int(math.floor(math.sqrt(W * H) / 150.0)) + 3
    m = local_mean(F, s)
    g = (F == m).astype(np.uint8)
    return g, s


def exclude_extremes(g: np.ndarray, F: np.ndarray, x: int) -> np.ndarray:
    """Exclude pixels with values 0 or 2^x - 1 from candidate set."""
    g = g.copy()
    max_val = (1 << x) - 1
    g[(F == 0) | (F == max_val)] = 0
    return g


def morphological_dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Binary dilation with a square structuring element of given radius.
    Separable max-filter using sliding window maximum via cumulative approach.
    """
    if radius <= 0:
        return mask.copy()
    H, W = mask.shape
    # Horizontal dilation
    out = np.zeros_like(mask)
    for j in range(W):
        j0 = max(0, j - radius)
        j1 = min(W, j + radius + 1)
        out[:, j] = np.max(mask[:, j0:j1], axis=1)
    # Vertical dilation
    res = np.zeros_like(out)
    for i in range(H):
        i0 = max(0, i - radius)
        i1 = min(H, i + radius + 1)
        res[i, :] = np.max(out[i0:i1, :], axis=0)
    return res


def compute_n_sp(W: int, H: int, G_active: int, N1: int, N0: int) -> int:
    """
    Eq. (4): N_sp = 2*sqrt(G_active) * (N1/(N1+N0))^2 * sqrt(W*H)
    """
    ratio = N1 / max(N1 + N0, 1)
    n_sp = 2.0 * math.sqrt(max(G_active, 1)) * (ratio ** 2) * math.sqrt(W * H)
    return max(1, int(round(n_sp)))


def simple_linear_iterative_clustering_single_iter(
    F: np.ndarray,
    g: np.ndarray,
    G_active: int,
    n_sp: int
) -> Tuple[np.ndarray, List[Tuple[int, int]], int]:
    """
    Single-iteration variant of SLIC (vectorized, batched).
    Since pre-quantized input has large uniform bands with deterministic
    boundaries, a single assignment step is sufficient.
    """
    H, W = F.shape
    grid_step = max(1, int(math.sqrt(H * W / max(n_sp, 1))))

    centers = []
    for i in range(grid_step // 2, H, grid_step):
        for j in range(grid_step // 2, W, grid_step):
            centers.append((i, j, int(F[i, j])))
    n_sp_actual = len(centers)
    if n_sp_actual == 0:
        return np.zeros((H, W), dtype=np.int32), [], 0

    centers_arr = np.array(centers, dtype=np.float32)  # (K, 3)
    m = 10.0
    m2 = m * m
    S2 = float(grid_step * grid_step)

    labels = np.empty((H, W), dtype=np.int32)
    batch = 64  # process 64 rows at a time to limit memory
    for i0 in range(0, H, batch):
        i1 = min(i0 + batch, H)
        yy, xx = np.mgrid[i0:i1, 0:W]
        vv = F[i0:i1, :].astype(np.float32)
        # dc: (b, W, K)
        dc = (yy[..., None] - centers_arr[:, 0]) ** 2 \
             + (xx[..., None] - centers_arr[:, 1]) ** 2
        dv = (vv[..., None] - centers_arr[:, 2]) ** 2
        d = dc + (dv / m2) * S2
        labels[i0:i1, :] = np.argmin(d, axis=-1)

    return labels, [(int(c[0]), int(c[1])) for c in centers], n_sp_actual


def adjacency_driven_region_merging(
    F: np.ndarray,
    labels: np.ndarray,
    g: np.ndarray
) -> Tuple[List[Set[Tuple[int, int]]], List[int]]:
    """
    ARM: merge adjacent superpixels sharing identical pixel value and
    8-connected boundaries.
    """
    H, W = F.shape
    label_value_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for i in range(H):
        for j in range(W):
            if g[i, j] == 0:
                continue
            key = (int(labels[i, j]), int(F[i, j]))
            label_value_map.setdefault(key, []).append((i, j))

    group_keys = list(label_value_map.keys())
    n_groups = len(group_keys)

    parent = list(range(n_groups))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pixel_group: Dict[Tuple[int, int], int] = {}
    for idx, key in enumerate(group_keys):
        for (i, j) in label_value_map[key]:
            pixel_group[(i, j)] = idx

    for (i, j), gi in pixel_group.items():
        for ni, nj in get_8_neighbors(i, j, H, W):
            if (ni, nj) in pixel_group:
                gj = pixel_group[(ni, nj)]
                if gi != gj and group_keys[gi][1] == group_keys[gj][1]:
                    union(gi, gj)

    root_to_pixels: Dict[int, Set[Tuple[int, int]]] = {}
    root_to_value: Dict[int, int] = {}
    for idx, key in enumerate(group_keys):
        r = find(idx)
        root_to_pixels.setdefault(r, set()).update(label_value_map[key])
        root_to_value[r] = key[1]

    clusters = list(root_to_pixels.values())
    cluster_values = [root_to_value[r] for r in root_to_pixels.keys()]
    return clusters, cluster_values


def region_qualification_criteria(
    clusters: List[Set[Tuple[int, int]]],
    cluster_values: List[int],
    f_original: np.ndarray,
    n: int,
    F: np.ndarray
) -> List[Set[Tuple[int, int]]]:
    """
    RQC: filter clusters by:
      1. Neighborhood criterion: at least one neighboring cluster
      2. Equidistant step criterion: neighbor diff == 2^n
      3. Grayscale richness criterion: original region has >= n distinct levels
    """
    H, W = f_original.shape
    n_clusters = len(clusters)

    pixel_cluster: Dict[Tuple[int, int], int] = {}
    for idx, c in enumerate(clusters):
        for p in c:
            pixel_cluster[p] = idx

    neighbors: List[Set[int]] = [set() for _ in range(n_clusters)]
    for idx, c in enumerate(clusters):
        for (i, j) in c:
            for ni, nj in get_8_neighbors(i, j, H, W):
                if (ni, nj) in pixel_cluster:
                    nidx = pixel_cluster[(ni, nj)]
                    if nidx != idx:
                        neighbors[idx].add(nidx)

    step = 1 << n
    qualified = []
    for idx, c in enumerate(clusters):
        if len(neighbors[idx]) == 0:
            continue
        ok = True
        for nidx in neighbors[idx]:
            if abs(cluster_values[idx] - cluster_values[nidx]) != step:
                ok = False
                break
        if not ok:
            continue
        vals = set()
        for (i, j) in c:
            vals.add(int(f_original[i, j]))
            if len(vals) >= n:
                break
        if len(vals) < n:
            continue
        qualified.append(c)

    return qualified


def detect_false_contour_regions(
    f: np.ndarray,
    n: int,
    x: int
) -> Tuple[List[Set[Tuple[int, int]]], np.ndarray, np.ndarray]:
    """
    Full detection pipeline:
      pre-quantize -> variable thresholding -> exclude extremes -> dilate
      -> SLIC (single iter) -> ARM -> RQC
    """
    F = quantize_image(f, n)
    H, W = f.shape

    g, s = variable_thresholding(F)
    g = exclude_extremes(g, F, x)
    radius = (s - 1) // 2
    g = morphological_dilate_binary(g, radius)

    N1 = int(np.sum(g == 1))
    N0 = int(np.sum(g == 0))

    candidate_vals = f[g == 1]
    G_active = len(np.unique(candidate_vals)) if candidate_vals.size > 0 else 1

    n_sp = compute_n_sp(W, H, G_active, N1, N0)

    labels, centers, n_sp_actual = simple_linear_iterative_clustering_single_iter(
        F, g, G_active, n_sp
    )

    clusters, cluster_values = adjacency_driven_region_merging(F, labels, g)

    qualified = region_qualification_criteria(clusters, cluster_values, f, n, F)

    return qualified, F, g


# =============================================================================
# Stage 2: Pre-compression Processing
# =============================================================================

def grayscale_level_grouping(
    A: List[Set[Tuple[int, int]]],
    f_original: np.ndarray,
    F_quantized: np.ndarray,
    n: int
) -> Tuple[List[Dict], int, int, int]:
    """
    For each false contour cluster A_i:
      - Extract original pixel values -> T_i
      - Sort distinct levels ascending -> E_i
      - Partition into n subgroups -> R_i
      - Compute H_i via Eq. (5)

    Returns:
        clusters_info, r_min, h_max, h_min
    """
    clusters_info = []
    r_min = float('inf')

    for A_i in A:
        coords = list(A_i)
        orig_vals = np.array([int(f_original[i, j]) for (i, j) in coords])
        q_i = int(F_quantized[coords[0][0], coords[0][1]])

        E_i = sorted(set(orig_vals.tolist()))
        o = len(E_i)

        R_i = []
        subgroup_size = math.ceil(o / n) if o > 0 else 1
        for j in range(n):
            start = j * subgroup_size
            end = min(start + subgroup_size, o)
            R_i.append(E_i[start:end] if start < o else [])

        # Count pixels per subgroup using searchsorted (faster than np.isin)
        E_arr = np.array(E_i, dtype=np.int64) if E_i else np.array([], dtype=np.int64)
        counts = []
        for R_j in R_i:
            if len(R_j) == 0:
                counts.append(0)
                continue
            lo = np.searchsorted(E_arr, R_j[0], side='left')
            hi = np.searchsorted(E_arr, R_j[-1], side='right')
            cnt = int(np.sum((orig_vals >= E_arr[lo]) & (orig_vals <= E_arr[hi - 1]))) if hi > lo else 0
            counts.append(cnt)
            if 0 < cnt < r_min:
                r_min = cnt

        step = 1 << n
        H_i = [q_i + j * step for j in range(n)]

        clusters_info.append({
            'coords': coords,
            'q_i': q_i,
            'E_i': E_i,
            'R_i': R_i,
            'H_i': H_i,
            'counts': counts,
            'orig_vals': orig_vals,
        })

    if r_min == float('inf'):
        r_min = 1
    r_min = min(int(r_min), (1 << 24) - 1)

    all_H = []
    for info in clusters_info:
        all_H.extend(info['H_i'])
    h_max = max(all_H) if all_H else 0
    h_min = min(all_H) if all_H else 0

    return clusters_info, r_min, h_max, h_min


def apply_pre_compression_mapping(
    clusters_info: List[Dict],
    f_original: np.ndarray
) -> np.ndarray:
    """
    Apply the pre-compression mapping to original pixel values.
    Each pixel in subgroup R^j_i is mapped to H^j_i.
    """
    mapped = f_original.copy()
    for info in clusters_info:
        coords = info['coords']
        R_i = info['R_i']
        H_i = info['H_i']
        orig_vals = info['orig_vals']

        # Build value -> subgroup index lookup
        val_to_j: Dict[int, int] = {}
        for jdx, R_j in enumerate(R_i):
            for v in R_j:
                val_to_j[v] = jdx

        for idx, (i, j) in enumerate(coords):
            v = int(orig_vals[idx])
            jdx = val_to_j.get(v, 0)
            mapped[i, j] = H_i[jdx]

    return mapped


# =============================================================================
# Stage 3: Decompression and Image Reconstruction
# =============================================================================

def connected_components_8(F_q: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """
    Find 8-connected components of pixels sharing the same grayscale value.
    """
    H, W = F_q.shape
    comp_map = -np.ones((H, W), dtype=np.int32)
    components = []
    current_label = 0

    for i in range(H):
        for j in range(W):
            if comp_map[i, j] != -1:
                continue
            val = F_q[i, j]
            q = deque([(i, j)])
            comp_map[i, j] = current_label
            pixels = []
            while q:
                ci, cj = q.popleft()
                pixels.append((ci, cj))
                for ni, nj in get_8_neighbors(ci, cj, H, W):
                    if comp_map[ni, nj] == -1 and F_q[ni, nj] == val:
                        comp_map[ni, nj] = current_label
                        q.append((ni, nj))
            components.append({
                'value': int(val),
                'pixels': pixels,
                'label': current_label
            })
            current_label += 1

    return comp_map, components


def component_filtering_criteria(
    components: List[Dict],
    r_min: int,
    h_min: int,
    h_max: int
) -> List[Dict]:
    """
    CFC: cardinality >= r_min and value in [h_min, h_max].
    """
    qualified = []
    for comp in components:
        if len(comp['pixels']) < r_min:
            continue
        if not (h_min <= comp['value'] <= h_max):
            continue
        qualified.append(comp)
    return qualified


def secondary_mapping_positioning(
    components: List[Dict],
    n: int
) -> List[Dict]:
    """
    Path-by-path feature determination:
      For each seed (component with value Z_1), find n-1 successive merges
      where each next component's value = current + 1 and is 8-adjacent.

    Returns: list of composite components.
    """
    if not components:
        return []

    # Index components by value
    value_to_indices: Dict[int, List[int]] = {}
    for idx, comp in enumerate(components):
        value_to_indices.setdefault(comp['value'], []).append(idx)

    # Build adjacency between components
    pixel_to_comp: Dict[Tuple[int, int], int] = {}
    for idx, comp in enumerate(components):
        for p in comp['pixels']:
            pixel_to_comp[p] = idx

    n_comps = len(components)
    adj: List[Set[int]] = [set() for _ in range(n_comps)]
    for idx, comp in enumerate(components):
        for (i, j) in comp['pixels']:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    p = (i + di, j + dj)
                    if p in pixel_to_comp:
                        nidx = pixel_to_comp[p]
                        if nidx != idx:
                            adj[idx].add(nidx)

    used = [False] * n_comps
    composites = []
    sorted_values = sorted(value_to_indices.keys())

    for seed_val in sorted_values:
        if seed_val + n - 1 not in value_to_indices:
            continue
        for seed_idx in value_to_indices[seed_val]:
            if used[seed_idx]:
                continue
            # DFS to collect all valid paths of length n-1
            paths: List[List[int]] = []

            def dfs(cur_idx: int, cur_val: int, path: List[int], visited: Set[int]):
                if len(path) == n:
                    paths.append(path.copy())
                    return
                next_val = cur_val + 1
                if next_val not in value_to_indices:
                    return
                for nidx in value_to_indices[next_val]:
                    if used[nidx] or nidx in visited:
                        continue
                    if nidx in adj[cur_idx]:
                        visited.add(nidx)
                        path.append(nidx)
                        dfs(nidx, next_val, path, visited)
                        path.pop()
                        visited.remove(nidx)

            visited = {seed_idx}
            dfs(seed_idx, seed_val, [seed_idx], visited)

            for path in paths:
                # Mark all as used (prevents reuse by later seeds)
                for idx in path:
                    used[idx] = True
                values = [components[idx]['value'] for idx in path]
                pixels_by_value: Dict[int, List[Tuple[int, int]]] = {}
                all_pixels = []
                for idx in path:
                    v = components[idx]['value']
                    pixels_by_value.setdefault(v, []).extend(components[idx]['pixels'])
                    all_pixels.extend(components[idx]['pixels'])
                composites.append({
                    'values': sorted(values),
                    'pixels_by_value': pixels_by_value,
                    'all_pixels': all_pixels,
                })

    return composites


def secondary_mapping(
    composites: List[Dict],
    F_q: np.ndarray,
    n: int
) -> np.ndarray:
    """
    Apply secondary mapping Eqs. (6)-(7):
      H2^1 = 2^n * R2^1
      H2^n = H2^1 + 2^n - 1
      H2^j = H2^1 + floor(2^n/(n-1)) * (j-1), j=2..n-1
    """
    restored = F_q.copy()
    step = 1 << n

    for comp in composites:
        values = comp['values']
        if len(values) < 2:
            continue
        R2_1 = values[0]
        H2_1 = step * R2_1
        H2_n = H2_1 + step - 1

        H2 = [0] * len(values)
        H2[0] = H2_1
        H2[-1] = H2_n
        if n >= 3 and len(values) >= 3:
            base = step // (n - 1)
            for j in range(1, len(values) - 1):
                H2[j] = H2_1 + base * j

        for jdx, v in enumerate(values):
            for (i, j) in comp['pixels_by_value'].get(v, []):
                restored[i, j] = H2[jdx]

    return restored


# =============================================================================
# Full Pipeline
# =============================================================================

class FalseContourSuppression:
    """
    Complete pipeline:
      1. Detection: pre-quantize -> variable thresholding -> SLIC -> ARM -> RQC
      2. Pre-compression: grayscale level grouping + mapping
      3. Decompression: CFC -> secondary mapping positioning -> secondary mapping
    """

    def __init__(self, n: int, x: int = 8):
        assert 2 <= n <= x - 3, f"Operating range: 2 <= n <= {x-3}"
        self.n = n
        self.x = x

    def compress(self, f: np.ndarray) -> Dict:
        n, x = self.n, self.x
        f = f.astype(np.int32)

        A, F_q, g = detect_false_contour_regions(f, n, x)

        clusters_info, r_min, h_max, h_min = grayscale_level_grouping(
            A, f, F_q, n
        )
        mapped = apply_pre_compression_mapping(clusters_info, f)

        step = 1 << n
        compressed = mapped // step

        return {
            'compressed': compressed.astype(np.uint16),
            'F_q': F_q,
            'A': A,
            'clusters_info': clusters_info,
            'r_min': r_min,
            'h_max': h_max,
            'h_min': h_min,
            'shape': f.shape,
        }

    def decompress(self, payload: Dict) -> np.ndarray:
        n = self.n
        r_min = payload['r_min']
        h_min = payload['h_min']
        h_max = payload['h_max']

        step = 1 << n
        F_q_recon = payload['compressed'].astype(np.int32) * step

        comp_map, components = connected_components_8(F_q_recon)
        qualified = component_filtering_criteria(components, r_min, h_min, h_max)
        composites = secondary_mapping_positioning(qualified, n)
        restored = secondary_mapping(composites, F_q_recon, n)

        max_val = (1 << self.x) - 1
        restored = np.clip(restored, 0, max_val)
        return restored.astype(np.uint8 if self.x == 8 else np.uint16)


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    H, W = 256, 256
    x = 8
    n = 4

    f = np.zeros((H, W), dtype=np.int32)
    for i in range(H):
        for j in range(W):
            if 64 <= i < 192 and 64 <= j < 192:
                t = (i - 64) / 128.0
                f[i, j] = int(144 + t * 15)
            else:
                f[i, j] = np.random.randint(0, 256)
    f = f.astype(np.uint8)

    F_direct = quantize_image(f.astype(np.int32), n).astype(np.uint8)

    pipeline = FalseContourSuppression(n=n, x=x)
    payload = pipeline.compress(f)
    restored = pipeline.decompress(payload)

    print("=" * 60)
    print("False Contour Suppression Pipeline - Demo")
    print("=" * 60)
    print(f"Image shape: {f.shape}")
    print(f"Bit depth x = {x}, quantization level n = {n}")
    print(f"Original range: [{f.min()}, {f.max()}]")
    print(f"Direct quantized range: [{F_direct.min()}, {F_direct.max()}]")
    print(f"Restored range: [{restored.min()}, {restored.max()}]")
    print(f"Detected false contour regions: {len(payload['A'])}")
    print(f"r_min = {payload['r_min']}")
    print(f"h_min = {payload['h_min']}, h_max = {payload['h_max']}")

    region = (slice(64, 192), slice(64, 192))
    orig_region = f[region].astype(np.float64)
    direct_region = F_direct[region].astype(np.float64)
    restored_region = restored[region].astype(np.float64)

    mse_direct = np.mean((orig_region - direct_region) ** 2)
    mse_restored = np.mean((orig_region - restored_region) ** 2)

    print(f"\nIn smooth region:")
    print(f"  MSE (direct quantization): {mse_direct:.2f}")
    print(f"  MSE (proposed method):     {mse_restored:.2f}")
    print(f"  Improvement: {mse_direct / max(mse_restored, 1e-6):.2f}x")

    print(f"\nDistinct levels in smooth region:")
    print(f"  Original: {len(np.unique(orig_region))}")
    print(f"  Direct:   {len(np.unique(direct_region))}")
    print(f"  Restored: {len(np.unique(restored_region))}")
