from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import savemat
from path_planner.env import GridWorld3DEnv
from experiments.configurable_agent import ConfigurableDQNAgent, AgentConfig

SIZE_X, SIZE_Y, SIZE_Z = 60, 80, 27
MAX_STEPS = 5 * (SIZE_X + SIZE_Y + SIZE_Z)  # 835
MAX_EXEC_TICKS = int(np.ceil(MAX_STEPS * 1.5))


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception: pass


def make_env(seed=None):
    return GridWorld3DEnv(size_x=SIZE_X, size_y=SIZE_Y, size_z=SIZE_Z,
                          max_steps=MAX_STEPS, cube1=None, cube2=None, seed=seed)


def load_proposed_agent(checkpoint: Path):
    payload = torch.load(checkpoint, map_location='cpu')
    env = make_env(0)
    agent = ConfigurableDQNAgent(env.observation_space.shape[0], env.action_space.n,
                                 device='cpu', use_per=True, config=AgentConfig())
    agent.load_checkpoint(payload, load_optimizer=False)
    return agent


def rollout_policy(env, agent, max_steps=MAX_STEPS):
    state, _ = env.reset()
    path = [tuple(env.pos)]
    collision_attempts = 0
    for _ in range(max_steps):
        before = tuple(env.pos)
        action = agent.select_action(state, eps=0.0)
        dx,dy,dz = env.moves[action]
        nx,ny,nz = before[0]+dx, before[1]+dy, before[2]+dz
        blocked = not (0 <= nx < env.size_x and 0 <= ny < env.size_y and 0 <= nz < env.size_z) or \
                  (0 <= nx < env.size_x and 0 <= ny < env.size_y and 0 <= nz < env.size_z and env.grid[nz,ny,nx] > 0)
        if blocked: collision_attempts += 1
        state, _, terminated, truncated, _ = env.step(action)
        path.append(tuple(env.pos))
        if terminated or truncated: break
    arr = np.asarray(path, dtype=float)
    success = bool(env.wp_reached) and env._distance_to_target(env.pos, env.goal) <= 0.5
    return {'path':arr, 'success':int(success), 'collision_attempts':collision_attempts,
            'final_distance':float(env._distance_to_target(env.pos, env.goal))}


def plan_from_current_state(env, agent, max_steps=MAX_STEPS):
    saved = dict(pos=tuple(env.pos), wp_reached=bool(env.wp_reached), steps=int(env.steps),
                 last_move_dir=env.last_move_dir, prev_action=env.prev_action,
                 best_dist_to_wp=env.best_dist_to_wp, best_dist_to_goal=env.best_dist_to_goal,
                 steps_since_wp=env.steps_since_wp)
    try:
        state = env._norm_state(env.pos)
        path=[tuple(env.pos)]
        for _ in range(max_steps):
            a=agent.select_action(state, eps=0.0)
            state,_,terminated,truncated,_=env.step(a)
            path.append(tuple(env.pos))
            if terminated or truncated: break
        return np.asarray(path,dtype=float)
    finally:
        for k,v in saved.items(): setattr(env,k,v)
        env._refresh_padded_grid()


def build_segment_directions(path):
    path=np.asarray(path,dtype=int)
    if len(path)==0: return np.zeros((0,3),dtype=int)
    dirs=np.zeros_like(path,dtype=int)
    for i in range(len(path)-1): dirs[i]=np.clip(path[i+1]-path[i],-1,1)
    return dirs


def find_waypoint_index(path, waypoint):
    d=np.linalg.norm(np.asarray(path,float)-np.asarray(waypoint,float)[None,:],axis=1)
    idx=np.where(d<=1.0)[0]
    return int(idx[0]) if len(idx) else 0


def valid_cube_around_point(env, point, r_xy=1, r_z=1):
    x,y,z=map(int,point)
    return {'min':(max(0,x-r_xy),max(0,y-r_xy),max(0,z-r_z)),
            'max':(min(env.size_x,x+r_xy+1),min(env.size_y,y+r_xy+1),min(env.size_z,z+r_z+1)),
            'value':1}


def path_length(P):
    P=np.asarray(P,float)
    return float(np.linalg.norm(np.diff(P,axis=0),axis=1).sum()) if len(P)>1 else 0.0


def save_path(outdir: Path, stem: str, P):
    P=np.asarray(P,float)
    pd.DataFrame(P,columns=['x','y','z']).to_csv(outdir/f'{stem}.csv',index=False)
    savemat(outdir/f'{stem}.mat',{stem:P})


def save_grid(outdir: Path, env, stem='grid_world_with_new_obstacle'):
    np.save(outdir/f'{stem}.npy', env.grid)
    savemat(outdir/f'{stem}.mat', {'grid':env.grid, 'size_x':env.size_x, 'size_y':env.size_y,
                                   'size_z':env.size_z, 'start':np.asarray(env.start),
                                   'goal':np.asarray(env.goal), 'waypoint':np.asarray(env.waypoint)})


def write_metrics(outdir: Path, metrics: dict):
    pd.DataFrame([metrics]).to_csv(outdir/'metrics.csv',index=False)
    (outdir/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
