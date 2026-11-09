# path_planner/viz.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
# --- 放在 viz.py 顶部 import 之后 ---

def _ensure_gui_backend():
    """
    若当前后端为 Agg/inline，则尝试切换到可交互后端。
    优先 QtAgg/Qt5Agg（需要 PyQt/PySide），失败再退到 TkAgg（需要 tkinter）。
    切换失败则静默返回，让上层决定是否跳过弹窗。
    """
    import matplotlib
    import matplotlib.pyplot as _plt

    b = matplotlib.get_backend().lower()
    if 'agg' in b or 'inline' in b:
        for cand in ('QtAgg', 'Qt5Agg', 'TkAgg', 'MacOSX'):
            try:
                _plt.switch_backend(cand)   # 注意：必须在 import pyplot 之后调用
                # 小自检：开关一个空 figure 看是否报错
                fig = _plt.figure(); _plt.close(fig)
                print(f"[matplotlib] switched backend -> {cand}")
                return True
            except Exception:
                continue
        print(f"[matplotlib] ⚠️ no GUI backend available (current: {matplotlib.get_backend()})")
        return False
    return True

def _set_world_ticks_and_labels(ax, env):
    xticks = np.linspace(0, env.size_x, 6)
    yticks = np.linspace(0, env.size_y, 6)
    zticks = np.linspace(0, env.size_z, 6)
    ax.set_xticks(xticks); ax.set_yticks(yticks); ax.set_zticks(zticks)
    ax.set_xticklabels([f"{x - env.size_x/2:.0f}" for x in xticks])
    ax.set_yticklabels([f"{y - env.size_y/2:.0f}" for y in yticks])
    ax.set_zticklabels([f"{z:.0f}" for z in zticks])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

def prepare_voxels(env):
    """
    返回:
      filled: bool[x,y,z]
      colors: RGBA[x,y,z,4]
    颜色映射：
      1=灰(基线障碍)、2=红(A叶)、3=绿(B叶)、4=蓝(C叶)
      新增障碍（相较 baseline）= 亮黄 (1.0, 0.9, 0.2, 0.95)
    说明：
      - 基线障碍：直接按 grid 值映射
      - 新增障碍：覆盖显示为亮黄
    """
    g = env.grid  # z,y,x
    filled = (g > 0).transpose(2,1,0)  # x,y,z
    colors = np.zeros((*filled.shape, 4), dtype=float)  # RGBA

    # 统一的透明灰：塔/机舱/柜子/墙等（grid==1）
    mask1 = (g == 1).transpose(2,1,0)
    colors[mask1] = (0.60, 0.60, 0.60, 0.35)

    # 三片叶：更高不透明度，颜色区分
    mask2 = (g == 2).transpose(2,1,0)  # A
    mask3 = (g == 3).transpose(2,1,0)  # B
    mask4 = (g == 4).transpose(2,1,0)  # C
    colors[mask2] = (0.90, 0.10, 0.10, 0.88)  # 红
    colors[mask3] = (0.10, 0.70, 0.20, 0.88)  # 绿
    colors[mask4] = (0.10, 0.40, 0.95, 0.88)  # 蓝

    # 新增：新增障碍亮黄覆盖
    # env.added_obstacles_mask 是 z,y,x 的布尔掩码，仅在 reset 后（扰动发生时）有效
    added = getattr(env, "added_obstacles_mask", None)
    if added is not None:
        add_mask_xyz = added.transpose(2,1,0)  # x,y,z
        # 只在 filled 的位置覆盖（理论上 added 都是 filled）
        colors[add_mask_xyz] = (1.00, 0.90, 0.20, 0.95)  # 亮黄

    # 未填充位置 alpha=0（防止默认为黑）
    colors[~filled] = (0,0,0,0)
    return filled, colors
def draw_stage_points(ax, env, color="orange"):
    """
    仅绘制阶段点的中心（实心点）并标注 W1..Wk，不画范围圈。
    返回一个用于图例的代理句柄；若无阶段点则返回 None。
    """
    if not hasattr(env, "stage_points") or len(env.stage_points) == 0:
        return None

    handle = None
    for idx, sp in enumerate(env.stage_points, start=1):
        s = ax.scatter(sp[0], sp[1], sp[2], s=60, c=color, depthshade=False)
        ax.text(sp[0], sp[1], sp[2] + 0.5, f"W{idx}", color=color,
                fontsize=9, ha="center", va="bottom")
        if handle is None:
            handle = s  # 用第一个散点作为图例句柄
    return handle


def draw_static(ax, env, filled, colors):
    ax.voxels(filled, facecolors=colors, edgecolor=(0,0,0,0.18), linewidth=0.2, shade=False)

    # Start / Goal
    sx, sy, sz = env.start; gx, gy, gz = env.goal
    start_scatter = ax.scatter(sx, sy, sz, s=60, c='b', depthshade=False, label='Start')
    goal_scatter  = ax.scatter(gx, gy, gz, s=60, c='g', depthshade=False, label='Goal')

    ax.set_xlim(0, env.size_x); ax.set_ylim(0, env.size_y); ax.set_zlim(0, env.size_z)
    ax.set_box_aspect((env.size_x, env.size_y, env.size_z))
    _set_world_ticks_and_labels(ax, env)

    # 图例：起点/终点 + 障碍 + 叶片
    obstacle_proxy = plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.35, ec='k')
    bladeA_proxy   = plt.Rectangle((0, 0), 1, 1, fc=(0.90, 0.10, 0.10, 0.88))
    bladeB_proxy   = plt.Rectangle((0, 0), 1, 1, fc=(0.10, 0.70, 0.20, 0.88))
    bladeC_proxy   = plt.Rectangle((0, 0), 1, 1, fc=(0.10, 0.40, 0.95, 0.88))
    handles = [start_scatter, goal_scatter, obstacle_proxy, bladeA_proxy, bladeB_proxy, bladeC_proxy]
    labels  = ["Start", "Goal", "Obstacle", "Blade A", "Blade B", "Blade C"]

    # 阶段点
    wp_proxy = draw_stage_points(ax, env, color="orange")
    if wp_proxy is not None:
        handles.append(wp_proxy); labels.append("Waypoint")

    # 新增：如果有新增障碍，加入 legend 项
    added = getattr(env, "added_obstacles_mask", None)
    if added is not None and np.any(added):
        new_obs_proxy = plt.Rectangle((0, 0), 1, 1, fc=(1.00, 0.90, 0.20, 0.95), ec='k')
        handles.append(new_obs_proxy); labels.append("New Obstacles")

    ax.legend(handles, labels, loc="upper right")
def save_training_curves(rewards, png_path="train_rewards.png", gif_path="train_rewards.gif"):
    plt.figure(figsize=(7,4))
    plt.plot(rewards)
    plt.title("Training Rewards"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(png_path, dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, len(rewards))
    ax.set_ylim(min(rewards) - 1, max(rewards) + 1)
    line, = ax.plot([], [], 'r-')
    k = 5
    frames_idx = list(range(0, len(rewards) + 1, k))
    if frames_idx[-1] != len(rewards):
        frames_idx.append(len(rewards))
    def _upd(i):
        f = frames_idx[i]
        if f == 0: 
            return line,
        line.set_data(range(f), rewards[:f])
        return line,
    ani = FuncAnimation(fig, _upd, frames=len(frames_idx), blit=True, interval=1)
    ani.save(gif_path, writer=PillowWriter(fps=40))
    plt.close(fig)

def animate_path(
    env,
    *,
    path_raw=None,
    path_smooth=None,
    out_gif="test.gif",
    out_mp4=None,
    anim_stride=2,
    interval_ms=300
):
    """
    生成动态 3D 路径动图（GIF/MP4）。支持两条路径同步推进：
      - Raw Path   : 蓝色
      - Smoothed   : 红色
    注意：path_raw / path_smooth 必须用关键字参数传入；不再做“单路径当 Raw”的自动替换。
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def _as_np(path):
        if path is None:
            return None
        arr = np.asarray(path)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Path must have shape (N, 3)")
        return arr

    path_raw    = _as_np(path_raw)
    path_smooth = _as_np(path_smooth)

    if path_raw is None and path_smooth is None:
        raise ValueError("Please pass at least one of path_raw or path_smooth (by keyword).")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    filled, colors = prepare_voxels(env)
    draw_static(ax, env, filled, colors)

    line_raw = line_smooth = None
    if path_raw is not None:
        line_raw, = ax.plot([], [], [], '-', color='blue', linewidth=1.8, label='Raw Path', zorder=60)
    if path_smooth is not None:
        line_smooth, = ax.plot([], [], [], '-', color='red',  linewidth=2.2, label='Smoothed Path', zorder=70)

    def _mk_frames(n):
        stride = max(1, int(anim_stride))
        idx = list(range(0, n, stride))
        if n > 0 and idx[-1] != n - 1:
            idx.append(n - 1)
        return idx

    n_raw = len(path_raw) if path_raw is not None else 0
    n_smt = len(path_smooth) if path_smooth is not None else 0
    n_max = max(n_raw, n_smt)
    frames_idx = _mk_frames(n_max if n_max > 0 else 1)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    if line_raw is not None and 'Raw Path' not in labels:
        handles.append(line_raw); labels.append('Raw Path')
    if line_smooth is not None and 'Smoothed Path' not in labels:
        handles.append(line_smooth); labels.append('Smoothed Path')
    ax.legend(handles, labels, loc='upper right')

    def _upd(frame_i):
        k = frames_idx[frame_i]
        artists = []
        if line_raw is not None and n_raw > 0:
            kr = min(k, n_raw - 1)
            line_raw.set_data(path_raw[:kr+1, 0], path_raw[:kr+1, 1])
            line_raw.set_3d_properties(path_raw[:kr+1, 2])
            artists.append(line_raw)
        if line_smooth is not None and n_smt > 0:
            ks = min(k, n_smt - 1)
            line_smooth.set_data(path_smooth[:ks+1, 0], path_smooth[:ks+1, 1])
            line_smooth.set_3d_properties(path_smooth[:ks+1, 2])
            artists.append(line_smooth)
        return tuple(artists)

    ani = FuncAnimation(fig, _upd, frames=len(frames_idx), interval=interval_ms, blit=False)

    fps = max(1, int(1000 / max(1, interval_ms)))
    if out_mp4:
        try:
            ani.save(out_mp4, writer=FFMpegWriter(fps=fps, bitrate=1500))
        except Exception:
            ani.save(out_gif, writer=PillowWriter(fps=fps))
    else:
        ani.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_static_path(
    env,
    *,
    path_raw=None,
    path_smooth=None,
    png_path="ruins_path.png",
    dpi=140
):
    """
    静态渲染两条路径到同一张图：
      - Raw Path   : 蓝色
      - Smoothed   : 红色
    注意：path_raw / path_smooth 必须用关键字参数传入；不再做自动替换。
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def _as_np(path):
        if path is None:
            return None
        arr = np.asarray(path)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Path must have shape (N, 3)")
        return arr

    path_raw    = _as_np(path_raw)
    path_smooth = _as_np(path_smooth)

    if path_raw is None and path_smooth is None:
        raise ValueError("Please pass at least one of path_raw or path_smooth (by keyword).")

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    filled, colors = prepare_voxels(env)
    draw_static(ax, env, filled, colors)

    handles, labels = ax.get_legend_handles_labels()

    if path_raw is not None:
        xr, yr, zr = path_raw[:,0], path_raw[:,1], path_raw[:,2]
        ax.plot(xr, yr, zr, '-', color='white', linewidth=3.4, zorder=85)   # 细白描边
        line_raw, = ax.plot(xr, yr, zr, '-', color='blue',  linewidth=1.9, zorder=90, label='Raw Path')
        if 'Raw Path' not in labels:
            handles.append(line_raw); labels.append('Raw Path')

    if path_smooth is not None:
        xs, ys, zs = path_smooth[:,0], path_smooth[:,1], path_smooth[:,2]
        ax.plot(xs, ys, zs, '-', color='white', linewidth=3.8, zorder=95)   # 细白描边
        line_smooth, = ax.plot(xs, ys, zs, '-', color='red',   linewidth=2.3, zorder=100, label='Smoothed Path')
        if 'Smoothed Path' not in labels:
            handles.append(line_smooth); labels.append('Smoothed Path')

    ax.legend(handles, labels, loc='upper right')
    plt.tight_layout(); plt.savefig(png_path, dpi=dpi)
    plt.close(fig)


def show_interactive_mpl(env, path_raw, path_smooth):
    import matplotlib, matplotlib.pyplot as plt
    print("[interactive] entering show_interactive_mpl, backend=", matplotlib.get_backend())
    try:
        plt.ioff()
        matplotlib.rcParams["interactive"] = False
    except Exception:
        pass
    # 兜底：确保 GUI 后端
    _ensure_gui_backend()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    filled, colors = prepare_voxels(env)
    draw_static(ax, env, filled, colors)

    # 原始路径（蓝）
    xs_r, ys_r, zs_r = path_raw[:, 0], path_raw[:, 1], path_raw[:, 2]
    ax.plot(xs_r, ys_r, zs_r, '-', color='blue', linewidth=1.5, label='Raw Path', zorder=50)

    # 平滑路径（红）
    xs_s, ys_s, zs_s = path_smooth[:, 0], path_smooth[:, 1], path_smooth[:, 2]
    ax.plot(xs_s, ys_s, zs_s, '-', color='red', linewidth=2.5, label='Smoothed Path', zorder=60)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(0, env.size_x); ax.set_ylim(0, env.size_y); ax.set_zlim(0, env.size_z)
    ax.set_box_aspect((env.size_x, env.size_y, env.size_z))
    ax.legend(loc='upper right')

    # 把窗口唤醒到最前（Qt/Tk 适用；失败忽略）
    try:
        mgr = plt.get_current_fig_manager()
        if hasattr(mgr, "window"):
            mgr.window.activateWindow()
            mgr.window.raise_()
    except Exception:
        pass

    # 关键：阻塞，直到你关闭窗口
    plt.show(block=True)
    print("[interactive] window closed")

