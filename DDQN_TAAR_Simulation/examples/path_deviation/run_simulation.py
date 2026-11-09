from pathlib import Path
import shutil
import numpy as np
from sim_common import *

SEED = 10
WIND_AIR_FRACTION = 0.85
DEVIATION = np.asarray([2,2,-2],dtype=int)
DEVIATION_THRESHOLD = 0.5
MAX_REPLANS_EVENT = 3

ROOT=Path(__file__).resolve().parent
CKPT=ROOT/'checkpoint'/'proposed_PER_DDQN_seed10.pth'
RESULTS=ROOT/'results'


def distance_to_path(pos,path):
    pts=np.asarray(path,float)
    return float(np.min(np.linalg.norm(pts-np.asarray(pos,float)[None,:],axis=1))) if len(pts) else 0.0


def choose_late_aerial_tick(path, fraction):
    path=np.asarray(path,float)
    air_idx=np.where(path[:,2]>0)[0]
    if len(air_idx)==0: return max(1,min(len(path)-2,int(0.8*len(path))))
    j=min(len(air_idx)-1,max(0,int(round(float(np.clip(fraction,0,1))*(len(air_idx)-1)))))
    return int(max(1,min(len(path)-2,air_idx[j])))


def execute(agent,use_behavior_management: bool):
    seed_everything(40_000+SEED)
    planning_env=make_env(SEED)
    initial=rollout_policy(planning_env,agent)
    if not initial['success']:
        raise RuntimeError('The selected PER-enhanced DDQN checkpoint fails the nominal task.')
    initial_path=initial['path']

    env=make_env(SEED); env.reset(seed=SEED)
    reference_path=initial_path.copy(); final_reference_path=initial_path.copy()
    seg_dirs=build_segment_directions(reference_path); seg_idx=0
    traj=[tuple(env.pos)]
    event_tick=choose_late_aerial_tick(initial_path,WIND_AIR_FRACTION)
    disturbance_applied=False; disturbed_point=None; replans=0; replan_point=None; replan_tick=None

    for tick in range(1,MAX_EXEC_TICKS+1):
        if env._distance_to_target(env.pos,env.goal)<=0.5 and env.wp_reached: break
        if (not env.wp_reached) and env._distance_to_target(env.pos,env.waypoint)<=1.0: env.wp_reached=True

        if len(seg_dirs)>0:
            seg_idx=min(seg_idx,len(seg_dirs)-1)
            dx,dy,dz=map(int,seg_dirs[seg_idx])
            cx,cy,cz=map(int,env.pos); nx,ny,nz=cx+dx,cy+dy,cz+dz
            feasible=(0<=nx<env.size_x and 0<=ny<env.size_y and 0<=nz<env.size_z and env.grid[nz,ny,nx]==0)
            if feasible:
                env.pos=(nx,ny,nz)
                if seg_idx < len(seg_dirs)-1: seg_idx += 1

        if (not disturbance_applied) and tick==event_tick:
            candidate=np.asarray(env.pos,dtype=int)+DEVIATION
            candidate=np.clip(candidate,[0,0,0],[env.size_x-1,env.size_y-1,env.size_z-1])
            cand=tuple(candidate.tolist())
            if env.grid[cand[2],cand[1],cand[0]]==0: env.pos=cand
            disturbance_applied=True; disturbed_point=np.asarray(env.pos,float)

        if use_behavior_management:
            ddev=distance_to_path(env.pos,reference_path)
            if ddev > DEVIATION_THRESHOLD and replans < MAX_REPLANS_EVENT:
                new_path=plan_from_current_state(env,agent)
                replans += 1
                if replan_point is None:
                    replan_point=np.asarray(env.pos,float); replan_tick=tick
                if len(new_path)>=2:
                    reference_path=new_path
                    final_reference_path=new_path.copy()
                    seg_dirs=build_segment_directions(reference_path); seg_idx=0

        traj.append(tuple(env.pos))

    traj=np.asarray(traj,float)
    success=bool(env.wp_reached) and env._distance_to_target(env.pos,env.goal)<=0.5
    return dict(initial_path=initial_path,final_reference_path=final_reference_path,executed_trajectory=traj,
                env=env,event_tick=int(event_tick),disturbed_point=disturbed_point,replan_tick=replan_tick,
                replan_point=replan_point,replan_count=int(replans),success=int(success),
                final_distance=float(env._distance_to_target(env.pos,env.goal)))


def save_case(name,data):
    out=RESULTS/name
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    save_path(out,'initial_nominal_path',data['initial_path'])
    save_path(out,'final_reference_path',data['final_reference_path'])
    save_path(out,'executed_trajectory',data['executed_trajectory'])
    save_grid(out,data['env'],stem='grid_world')
    meta={'seed':SEED,'behavior_management':name=='with_behavior_management','success':data['success'],
          'replan_count':data['replan_count'],'final_distance':data['final_distance'],
          'initial_path_length':path_length(data['initial_path']),
          'final_reference_path_length':path_length(data['final_reference_path']),
          'executed_path_length':path_length(data['executed_trajectory']),
          'disturbance_tick':data['event_tick'],'disturbance_vector':DEVIATION.tolist(),
          'disturbance_magnitude':float(np.linalg.norm(DEVIATION)),
          'disturbed_point':None if data['disturbed_point'] is None else data['disturbed_point'].tolist(),
          'replan_tick':data['replan_tick'],'replan_point':None if data['replan_point'] is None else data['replan_point'].tolist()}
    write_metrics(out,meta)


def main():
    RESULTS.mkdir(exist_ok=True)
    agent=load_proposed_agent(CKPT)
    with_mgr=execute(agent,True)
    without_mgr=execute(agent,False)
    save_case('with_behavior_management',with_mgr)
    save_case('without_behavior_management',without_mgr)
    print('\n=== Path-deviation case ===')
    for label,d in [('With behavior management',with_mgr),('Without behavior management',without_mgr)]:
        print(f'{label}: success={d["success"]}, replans={d["replan_count"]}, final_distance={d["final_distance"]:.3f}')
    print('Saved paths under:',RESULTS)

if __name__=='__main__': main()
