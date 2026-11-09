import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import CubicSpline

# ---------------- Ramer–Douglas–Peucker (RDP) 简化 ----------------
def rdp(points, eps):
    """
    points: (N, D) array-like
    eps: tolerance
    returns simplified points as ndarray
    """
    P = np.asarray(points, dtype=float)
    if P.shape[0] < 3:
        return P.copy()

    def point_line_distance(pt, a, b):
        # distance from pt to segment ab
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        pt = np.asarray(pt, dtype=float)
        ab = b - a
        ab2 = (ab * ab).sum()
        if ab2 == 0:
            return np.linalg.norm(pt - a)
        t = np.dot(pt - a, ab) / ab2
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return np.linalg.norm(pt - proj)

    def rdp_rec(pts):
        if pts.shape[0] < 3:
            return pts
        a = pts[0]
        b = pts[-1]
        dmax = -1.0
        idx = -1
        for i in range(1, pts.shape[0] - 1):
            d = point_line_distance(pts[i], a, b)
            if d > dmax:
                dmax = d
                idx = i
        if dmax > eps:
            left = rdp_rec(pts[:idx + 1])
            right = rdp_rec(pts[idx:])
            return np.vstack([left[:-1], right])
        else:
            return np.vstack([a, b])

    return rdp_rec(P)

# ---------- Catmull–Rom 基础 (保留原函数以兼容) ----------
def _catmull_rom_t(P0, P1, alpha=0.5):
    d = np.linalg.norm(P1 - P0) + 1e-9
    return d ** alpha

def _catmull_rom_segment(P0, P1, P2, P3, n_sample=10, alpha=0.5):
    P0, P1, P2, P3 = map(lambda p: np.asarray(p, dtype=np.float64), (P0, P1, P2, P3))
    t0 = 0.0
    t1 = t0 + _catmull_rom_t(P0, P1, alpha)
    t2 = t1 + _catmull_rom_t(P1, P2, alpha)
    t3 = t2 + _catmull_rom_t(P2, P3, alpha)
    m1 = ((P2 - P1) / (t2 - t1) - (P1 - P0) / (t1 - t0)) * (t2 - t1) / (t2 - t0)
    m2 = ((P3 - P2) / (t3 - t2) - (P2 - P1) / (t2 - t1)) * (t2 - t1) / (t3 - t1)
    ts = np.linspace(0.0, 1.0, n_sample, endpoint=False)
    pts = []
    for u in ts:
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1
        h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2
        h11 = u ** 3 - u ** 2
        p = h00 * P1 + h10 * (t2 - t1) * m1 + h01 * P2 + h11 * (t2 - t1) * m2
        pts.append(p)
    return np.asarray(pts)

def smooth_path_catmull_rom(path_xyz, samples_per_seg=12, alpha=0.5, clamp_z_min=0):
    P = np.asarray(path_xyz, dtype=np.float64)
    if len(P) < 4:
        if len(P) <= 1:
            return P.copy()
        out = []
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            ts = np.linspace(0, 1, samples_per_seg, endpoint=False)
            seg = (1 - ts)[:, None] * a + ts[:, None] * b
            out.append(seg)
        out.append(P[-1][None, :])
        S = np.vstack(out)
        if clamp_z_min is not None:
            S[:, 2] = np.maximum(S[:, 2], clamp_z_min)
        return S

    pts = []
    for i in range(-1, len(P) - 2):
        P0 = P[max(i, 0)]
        P1 = P[i + 1]
        P2 = P[i + 2]
        P3 = P[min(i + 3, len(P) - 1)]
        seg = _catmull_rom_segment(P0, P1, P2, P3, n_sample=samples_per_seg, alpha=alpha)
        pts.append(seg)
    pts.append(P[-1][None, :])
    S = np.vstack(pts)
    if clamp_z_min is not None:
        S[:, 2] = np.maximum(S[:, 2], clamp_z_min)
    return S

# ---------- 距离场/碰撞 ----------
def make_distance_field(env):
    occ = (env.grid.astype(np.uint8) > 0)
    free = ~occ
    df = distance_transform_edt(free)
    return df

def sample_segment(p, q, step=0.5):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    L = np.linalg.norm(q - p)
    if L < 1e-9:
        return np.array([p])
    n = max(2, int(np.ceil(L / step)) + 1)
    ts = np.linspace(0.0, 1.0, n)
    return (1 - ts)[:, None] * p + ts[:, None] * q

def in_bounds(env, pt):
    x, y, z = pt
    return (0 <= x < env.size_x) and (0 <= y < env.size_y) and (0 <= z < env.size_z)

def is_segment_free(env, df, p, q, clearance=1.0, step=0.5):
    for s in sample_segment(p, q, step=step):
        if not in_bounds(env, s):
            return False
        zz, yy, xx = int(round(s[2])), int(round(s[1])), int(round(s[0]))
        if env.grid[zz, yy, xx] == 1:
            return False
        if df[zz, yy, xx] < clearance:
            return False
    return True

# ---------- 捷径（分段 + 锚点保护）----------
def shortcut_smooth(env, df, path_xyz, iters=400, clearance=1.0, step=0.5, seed=0):
    rng = np.random.RandomState(seed)
    P = np.array(path_xyz, dtype=float).tolist()
    if len(P) <= 2:
        return np.array(P)
    for _ in range(iters):
        if len(P) <= 2:
            break
        i = rng.randint(0, len(P) - 2)
        j = rng.randint(i + 2, len(P))
        if is_segment_free(env, df, P[i], P[j - 1], clearance=clearance, step=step):
            P = P[:i + 1] + P[j - 1:]
    return np.asarray(P)

# ---------- 倒角（曲率更平滑 & 更密采样）----------
def fillet_corner(A, B, C, r):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)
    v1 = A - B
    v2 = C - B
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return []
    v1u = v1 / n1
    v2u = v2 / n2
    dot = np.clip(np.dot(v1u, v2u), -1.0, 1.0)
    theta = np.arccos(-dot)
    if theta < 1e-3:
        return []
    t = r / np.tan(theta / 2)
    t = min(t, 0.45 * n1, 0.45 * n2)
    B1 = B + v1u * t
    B2 = B + v2u * t
    e1 = (B2 - B1)
    n_e1 = np.linalg.norm(e1)
    if n_e1 < 1e-9:
        return []
    e1 = e1 / n_e1
    nrm = np.cross(v1u, v2u)
    n_nrm = np.linalg.norm(nrm)
    if n_nrm < 1e-9:
        return [(1 - t) * B + t * C]
    nrm = nrm / n_nrm
    e2 = np.cross(nrm, e1)
    mid = 0.5 * (B1 + B2)
    chord = np.linalg.norm(B2 - B1)
    Rc = chord / (2 * np.sin(theta / 2))
    offset = np.sqrt(max(Rc * Rc - (chord * 0.5) ** 2, 0.0))
    sgn = np.sign(np.dot(B - mid, e2))
    Cc = mid + sgn * offset * e2
    a0 = np.arctan2(np.dot(B1 - Cc, e2), np.dot(B1 - Cc, e1))
    a1 = np.arctan2(np.dot(B2 - Cc, e2), np.dot(B2 - Cc, e1))
    da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    # 更密的圆弧采样：最少6段，与角度、半径成正比
    nseg = max(6, int(np.ceil(max(theta, 0.35) * max(r, 1.0) * 1.2)))
    ts = np.linspace(0, 1, nseg + 2)[1:-1]
    arc = []
    for t_ in ts:
        a = a0 + da * t_
        p = Cc + Rc * np.cos(a) * e1 + Rc * np.sin(a) * e2
        arc.append(p)
    return [B1] + arc + [B2]

# ---------- 限幅 Catmull–Rom（centripetal，防过冲）----------
def catmull_rom_safe(P, samples_per_seg=12, alpha=0.5, tangent_cap=0.8):
    P = np.asarray(P, float)
    if len(P) < 4:
        return smooth_path_catmull_rom(P, samples_per_seg=samples_per_seg, alpha=alpha, clamp_z_min=0)
    out = []
    for i in range(-1, len(P) - 2):
        P0 = P[max(i, 0)]
        P1 = P[i + 1]
        P2 = P[i + 2]
        P3 = P[min(i + 3, len(P) - 1)]

        def _t(A, B):
            return (np.linalg.norm(B - A) + 1e-9) ** alpha

        t0 = 0.0
        t1 = t0 + _t(P0, P1)
        t2 = t1 + _t(P1, P2)
        t3 = t2 + _t(P2, P3)
        m1 = ((P2 - P1) / (t2 - t1) - (P1 - P0) / (t1 - t0)) * (t2 - t1) / (t2 - t0)
        m2 = ((P3 - P2) / (t3 - t2) - (P2 - P1) / (t2 - t1)) * (t2 - t1) / (t3 - t1)
        cap1 = tangent_cap * min(np.linalg.norm(P2 - P1), np.linalg.norm(P1 - P0))
        cap2 = tangent_cap * min(np.linalg.norm(P3 - P2), np.linalg.norm(P2 - P1))
        n1 = np.linalg.norm(m1)
        n2 = np.linalg.norm(m2)
        if n1 > cap1 and n1 > 1e-9:
            m1 = m1 * (cap1 / n1)
        if n2 > cap2 and n2 > 1e-9:
            m2 = m2 * (cap2 / n2)
        ts = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)
        for u in ts:
            h00 = 2 * u ** 3 - 3 * u ** 2 + 1
            h10 = u ** 3 - 2 * u ** 2 + u
            h01 = -2 * u ** 3 + 3 * u ** 2
            h11 = u ** 3 - u ** 2
            p = h00 * P1 + h10 * (t2 - t1) * m1 + h01 * P2 + h11 * (t2 - t1) * m2
            out.append(p)
    out.append(P[-1])
    return np.asarray(out)

# ---------- 碰撞投影（支持锁定锚点不动）----------
def project_out_of_collision(env, df, S, clearance=1.0, iters=5, step=0.5, locked_idx=None):
    S = np.asarray(S, float).copy()
    locked = set(locked_idx or [])
    for _ in range(iters):
        updated = False
        for k in range(1, len(S) - 1):
            if k in locked:
                continue
            p = S[k]
            if not in_bounds(env, p):
                continue
            zz, yy, xx = int(round(p[2])), int(round(p[1])), int(round(p[0]))
            if env.grid[zz, yy, xx] == 1 or df[zz, yy, xx] < clearance:
                def g(axis):
                    dpos = np.array([0, 0, 0], float)
                    dneg = np.array([0, 0, 0], float)
                    dpos[axis] = 1
                    dneg[axis] = -1
                    ppos = np.clip(
                        p + dpos,
                        [0, 0, 0],
                        [env.size_x - 1, env.size_y - 1, env.size_z - 1],
                    ).astype(int)
                    pneg = np.clip(
                        p + dneg,
                        [0, 0, 0],
                        [env.size_x - 1, env.size_y - 1, env.size_z - 1],
                    ).astype(int)
                    return df[ppos[2], ppos[1], ppos[0]] - df[pneg[2], pneg[1], pneg[0]]
                grad = np.array([g(0), g(1), g(2)], float)
                n = np.linalg.norm(grad)
                if n < 1e-6:
                    best = p
                    bestd = df[zz, yy, xx]
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                q = np.array([p[0] + dx, p[1] + dy, p[2] + dz])
                                if in_bounds(env, q):
                                    qz, qy, qx = int(round(q[2])), int(round(q[1])), int(round(q[0]))
                                    if df[qz, qy, qx] > bestd and env.grid[qz, qy, qx] == 0:
                                        bestd = df[qz, qy, qx]
                                        best = q
                    S[k] = best
                else:
                    S[k] = S[k] + (grad / n) * step
                updated = True
        if not updated:
            break
    return S

# ---------- 弧长重采样（保持锚点）----------
def _arclength(P):
    d = np.sqrt(((P[1:] - P[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s, s[-1]

def resample_by_arclength(P, n_samples, keep_idx=None):
    P = np.asarray(P, float)
    if len(P) <= 2 or n_samples <= len(P):
        # 简单线性下采样到 n_samples
        idx = np.linspace(0, len(P) - 1, num=n_samples, dtype=int)
        return P[idx]
    s, L = _arclength(P)
    if L < 1e-9:
        return np.repeat(P[:1], n_samples, axis=0)
    # 目标等距弧长
    tgt = np.linspace(0, L, n_samples)
    out = []
    j = 0
    for t in tgt:
        while j + 1 < len(s) and s[j + 1] < t:
            j += 1
        if j + 1 == len(s):
            out.append(P[-1])
            break
        w = (t - s[j]) / max(s[j + 1] - s[j], 1e-9)
        out.append(P[j] * (1 - w) + P[j + 1] * w)
    out = np.asarray(out)

    # 将必须保留的索引位置，强制覆盖到最近采样点
    if keep_idx:
        for k in keep_idx:
            if 0 <= k < len(P):
                # 找到离原路径点 P[k] 最近的采样索引
                d = np.linalg.norm(out - P[k], axis=1)
                out[np.argmin(d)] = P[k]
    return out

# ---------- 工具：在路径中插入锚点（用最近顶点；简单稳妥）----------
def insert_anchor_vertex(P, anchor):
    P = np.asarray(P, float)
    a = np.asarray(anchor, float)
    d = np.linalg.norm(P - a, axis=1)
    i = int(np.argmin(d))
    if np.linalg.norm(P[i] - a) < 1e-6:
        return P, i  # 已存在
    P2 = np.vstack([P[:i + 1], a[None, :], P[i + 1 :]])
    return P2, i + 1  # 返回新路径及锚点索引

# ---------- 安全平滑主入口（锚点：起点/阶段点/切换点/风扰点/终点）----------
def smooth_path_safe(
    env,
    path_xyz,
    clearance=1.5,
    shortcut_iters=400,
    fillet=True,
    fillet_scale=1.1,
    cr_alpha=0.5,
    cr_samples=12,
    cr_tangent_cap=0.8,
    enforce_waypoint=True,
    waypoint=None,
    max_points=100,
    rdp_eps=1.0,
):
    """
    返回经过安全光顺后的路径（≈max_points个点，包含关键锚点）：
    - 起点
    - 阶段目标点（waypoint，若 enforce_waypoint=True）
    - 模态切换点（z: 0<->>0）
    - 风扰前/后的两个点（通过最大步长检测）
    - 终点

    其它接口保持不变。
    """
    # 原始路径
    P_orig = np.asarray(path_xyz, float)
    if len(P_orig) <= 2:
        return P_orig.copy()

    # === 0. 构建锚点集合（起点 / 终点 / 阶段点 / 切换点 / 风扰点前后） ===
    anchor_coords = []

    # 起点 & 终点
    anchor_coords.append(P_orig[0])
    anchor_coords.append(P_orig[-1])

    # 阶段目标点（仅在 enforce_waypoint=True 时强制经过）
    if waypoint is None:
        waypoint_env = getattr(env, "waypoint", None)
    else:
        waypoint_env = waypoint
    if enforce_waypoint and waypoint_env is not None:
        waypoint_arr = np.asarray(waypoint_env, float)
        anchor_coords.append(waypoint_arr)
        waypoint = waypoint_arr  # 后面统一用 array
    else:
        waypoint = None

    # 切换点（z=0 <-> z>0 的两个点都当锚点）
    zs = P_orig[:, 2]
    for i in range(len(P_orig) - 1):
        z0, z1 = zs[i], zs[i + 1]
        if (z0 <= 0 and z1 > 0) or (z0 > 0 and z1 <= 0):
            anchor_coords.append(P_orig[i])
            anchor_coords.append(P_orig[i + 1])

    # 风扰点前后两个点：用“最大一步长”近似检测
    if len(P_orig) >= 3:
        diffs = np.linalg.norm(P_orig[1:] - P_orig[:-1], axis=1)
        k = int(np.argmax(diffs))
        # 阈值略大于正常一步（正常格子步长为1）
        if diffs[k] > 1.5:
            # k / k+1 分别视为风扰前/后两个锚点
            anchor_coords.append(P_orig[k])
            anchor_coords.append(P_orig[k + 1])

    # 去重（按坐标约等于），使用四舍五入到 3 位小数
    uniq = {}
    for a in anchor_coords:
        key = tuple(np.round(a, 3))
        uniq[key] = np.asarray(a, float)
    anchor_coords = list(uniq.values())

    # === 1) RDP 简化（在任何其它处理前）===
    P_rdp = rdp(P_orig, rdp_eps)
    if P_rdp.shape[0] < 2:
        return P_rdp

    # === 2) 距离场 ===
    df = make_distance_field(env)

    # === 3) 将所有锚点重新插入到简化后路径中 ===
    P0 = P_rdp
    anchor_indices = []
    for ac in anchor_coords:
        P0, idx = insert_anchor_vertex(P0, ac)
        anchor_indices.append(idx)
    anchor_present = len(anchor_indices) > 0

    # === 4) 按锚点分段做捷径，避免跨越锚点直接抹掉 ===
    seg_bounds = [0, len(P0) - 1]
    if anchor_present:
        seg_bounds.extend(anchor_indices)
    seg_bounds = sorted(set(seg_bounds))

    parts = []
    for a, b in zip(seg_bounds[:-1], seg_bounds[1:]):
        sub = P0[a : b + 1]
        sub2 = shortcut_smooth(
            env,
            df,
            sub,
            iters=shortcut_iters,
            clearance=clearance,
            step=0.5,
            seed=0,
        )
        # 避免连接处重复
        if parts:
            parts.append(sub2[1:])
        else:
            parts.append(sub2)
    P = np.vstack(parts)

    # === 5) 倒角（所有锚点保护：不过角）===
    if fillet and len(P) >= 3:
        Q = [P[0]]
        for i in range(1, len(P) - 1):
            B = P[i]
            # 如果该点是某个锚点，则不倒角，直接保留
            is_anchor = False
            for ac in anchor_coords:
                if np.allclose(B, ac, atol=1e-6):
                    is_anchor = True
                    break
            if is_anchor:
                Q.append(B)
                continue

            A, C = P[i - 1], P[i + 1]
            B_idx = (int(round(B[2])), int(round(B[1])), int(round(B[0])))
            local_clear = float(df[B_idx])
            # 更积极的半径（受清空距离限制）
            r = max(0.6, min(local_clear * fillet_scale, 5.0))
            arc_pts = fillet_corner(A, B, C, r)
            if arc_pts:
                Q.append(arc_pts[0])
                Q.extend(arc_pts[1:-1])
                Q.append(arc_pts[-1])
            else:
                Q.append(B)
        Q.append(P[-1])
        P = np.asarray(Q)
    else:
        P = np.asarray(P)

    # === 6) 用 CubicSpline（三次样条）按弧长参数化进行平滑（替代 Catmull-Rom）===
    if len(P) == 1:
        S = P.copy()
    else:
        s, L = _arclength(P)
        if L < 1e-9:
            S = np.repeat(P[:1], max(2, cr_samples), axis=0)
        else:
            n_seg = max(1, len(P) - 1)
            total_samples = max(2, int(cr_samples * n_seg))
            s_new = np.linspace(0.0, L, total_samples, endpoint=False)
            # 为确保包含终点，稍后 append 最后点
            cs_x = CubicSpline(s, P[:, 0], bc_type="natural")
            cs_y = CubicSpline(s, P[:, 1], bc_type="natural")
            cs_z = CubicSpline(s, P[:, 2], bc_type="natural")
            xs = cs_x(s_new)
            ys = cs_y(s_new)
            zs = cs_z(s_new)
            S = np.vstack([xs, ys, zs]).T
            S = np.vstack([S, P[-1][None, :]])  # 添加终点，保证闭合

    # === 7) 再次把锚点精确插入到样条结果中，后续投影时可锁定 ===
    for ac in anchor_coords:
        S, _ = insert_anchor_vertex(S, ac)

    # === 8) 碰撞投影（锁定所有锚点不动）===
    locked = []
    for ac in anchor_coords:
        idx = np.linalg.norm(S - ac, axis=1).argmin()
        locked.append(idx)
    locked = sorted(set(locked))
    S = project_out_of_collision(
        env,
        df,
        S,
        clearance=clearance,
        iters=6,
        step=0.6,
        locked_idx=locked,
    )

    # === 9) 简单可视化前的线段合并（保持可达性）===
    S2 = [S[0]]
    for i in range(1, len(S)):
        if is_segment_free(env, df, S2[-1], S[i], clearance=clearance, step=0.5):
            S2.append(S[i])
        else:
            mid = (S2[-1] + S[i]) * 0.5
            if is_segment_free(env, df, S2[-1], mid, clearance=clearance, step=0.5):
                S2.append(mid)
            if is_segment_free(env, df, mid, S[i], clearance=clearance, step=0.5):
                S2.append(S[i])
    S2 = np.asarray(S2)

    # === 10) 目标点数控制（≈max_points），并保持所有锚点 ===
    if len(S2) > max_points:
        if anchor_coords:
            # 找到每个锚点在 S2 中最近的索引，作为必须保留的索引
            keep_idx = []
            for ac in anchor_coords:
                keep_idx.append(np.linalg.norm(S2 - ac, axis=1).argmin())
            keep_idx = sorted(set(keep_idx))
            S3 = resample_by_arclength(S2, max_points, keep_idx=keep_idx)
        else:
            S3 = resample_by_arclength(S2, max_points)
    else:
        S3 = S2

    return S3
