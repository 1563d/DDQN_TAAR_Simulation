from __future__ import annotations
import argparse, csv, json, math, shutil, time
from pathlib import Path
import numpy as np
import torch
from scipy.io import savemat
from src.real_scale_env import RealScaleOffshoreEnv
from src.ddqn_per import DDQNPERAgent
from src.scenario_suite import load_suite, scenario_row

METHOD_NAME='Proposed (DDQN+PER)'


def compress(path):
    out=[]
    for p in path:
        if not out or tuple(p)!=tuple(out[-1]): out.append(tuple(p))
    return out


def metrics_for(path,env,collisions,wall_time,inference_time,success):
    ground=air=asc=desc=0.0; cost=0.0
    modes=[0 if p[2]==0 else 1 for p in path]
    transitions=sum(int(a!=b) for a,b in zip(modes,modes[1:]))
    for a,b in zip(path,path[1:]):
        dx,dy,dz=b[0]-a[0],b[1]-a[1],b[2]-a[2]; L=math.sqrt(dx*dx+dy*dy+dz*dz); Lm=L*env.resolution
        if b[2]==0: ground+=Lm; cost+=1.0*L
        else: air+=Lm; cost+=1.6*L
        if dz>0: asc+=dz*env.resolution; cost+=2.0*dz
        elif dz<0: desc+=(-dz)*env.resolution; cost+=0.4*(-dz)
    return {
        'success':int(success),'path_length_m':ground+air,'ground_distance_m':ground,'aerial_distance_m':air,
        'ascent_m':asc,'descent_m':desc,'mode_transitions':transitions,'normalized_motion_cost':cost,
        'collision_attempts':int(collisions),'planning_wall_time_s':float(wall_time),'policy_inference_time_s':float(inference_time),
        'path_points':len(path),'final_distance_m':float(env._distance(env.pos,env.goal)*env.resolution),
    }


def write_csv(path,rows):
    if not rows: return
    keys=[]
    for row in rows:
        for k in row:
            if k not in keys: keys.append(k)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def mean_std(values):
    a=np.asarray(values,dtype=float)
    if len(a)==0: return None,None
    return float(a.mean()), float(a.std(ddof=1)) if len(a)>1 else 0.0


def make_paper_summary(rows):
    n=len(rows); succ=[r for r in rows if int(r['success'])==1]; ns=len(succ)
    summary={
        'method':METHOD_NAME,'n_test_scenarios':n,'n_success':ns,'success_rate_percent':100.0*ns/n if n else 0.0,
        'continuous_path_metrics_population':'successful scenarios only',
        'runtime_metrics_population':'all test scenarios',
    }
    successful_metrics=['path_length_m','ground_distance_m','aerial_distance_m','ascent_m','descent_m','mode_transitions','normalized_motion_cost']
    all_metrics=['planning_wall_time_s','policy_inference_time_s','collision_attempts','final_distance_m']
    for k in successful_metrics:
        m,s=mean_std([r[k] for r in succ]); summary[k+'_mean']=m; summary[k+'_std']=s
    for k in all_metrics:
        m,s=mean_std([r[k] for r in rows]); summary[k+'_mean']=m; summary[k+'_std']=s
    return summary


def summary_rows(summary):
    rows=[{'metric':'Success rate','population':'all test scenarios','n':summary['n_test_scenarios'],'mean':summary['success_rate_percent'],'std':'','unit':'%'}]
    labels=[
        ('path_length_m','Path length','m','successful scenarios only'),('ground_distance_m','Ground distance','m','successful scenarios only'),
        ('aerial_distance_m','Aerial distance','m','successful scenarios only'),('mode_transitions','Mode transitions','count','successful scenarios only'),
        ('normalized_motion_cost','Normalized motion cost','a.u.','successful scenarios only'),('planning_wall_time_s','Planning wall time','s','all test scenarios'),
        ('policy_inference_time_s','Policy inference time','s','all test scenarios'),('collision_attempts','Collision attempts','count','all test scenarios'),
        ('final_distance_m','Final distance','m','all test scenarios')]
    for key,label,unit,pop in labels:
        n=summary['n_success'] if pop.startswith('successful') else summary['n_test_scenarios']
        rows.append({'metric':label,'population':pop,'n':n,'mean':summary.get(key+'_mean'),'std':summary.get(key+'_std'),'unit':unit})
    return rows


def run_one(env,agent,scenario):
    env.apply_scenario(scenario,validate_ground=True); s,_=env.reset(); raw=[env.pos]; total=0.0; inf=0.0; zero_valid_events=0; valid_counts=[]
    t0=time.perf_counter()
    for _ in range(env.max_steps):
        valid_mask=env.valid_action_mask()
        n_valid=int(np.count_nonzero(valid_mask)); valid_counts.append(n_valid)
        if n_valid==0: zero_valid_events+=1
        ti=time.perf_counter(); a=agent.select_action(s,0.0,valid_mask=valid_mask); inf+=time.perf_counter()-ti
        s,r,done,_,_=env.step(a); total+=r; raw.append(env.pos)
        if done: break
    wall=time.perf_counter()-t0; path=compress(raw); success=env.wp_reached and env._distance(env.pos,env.goal)<=0.5
    m=metrics_for(path,env,env.collision_attempts,wall,inf,success)
    m.update({
        'action_masking_enabled':bool(agent.use_action_masking),
        'mean_valid_actions':float(np.mean(valid_counts)) if valid_counts else 0.0,
        'min_valid_actions':int(min(valid_counts)) if valid_counts else 0,
        'zero_valid_action_events':int(zero_valid_events),
    })
    return path,raw,total,m


def save_one(out,env,scenario,path,raw,total,metrics,checkpoint,checkpoint_meta=None):
    out.mkdir(parents=True,exist_ok=True)
    cells=np.asarray(path,dtype=float); meters=cells*env.resolution; raw_cells=np.asarray(raw,dtype=float)
    np.savetxt(out/'planned_path_cells.csv',cells,delimiter=',',header='x,y,z',comments='')
    np.savetxt(out/'planned_path_m.csv',meters,delimiter=',',header='x_m,y_m,z_m',comments='')
    np.savetxt(out/'raw_execution_cells.csv',raw_cells,delimiter=',',header='x,y,z',comments='')
    savemat(out/'planned_path.mat',{'planned_path_cells':cells,'planned_path_m':meters,'start':np.asarray(env.start),'waypoint':np.asarray(env.waypoint),'goal':np.asarray(env.goal),'voxel_resolution_m':env.resolution})
    np.savez_compressed(out/'scene_snapshot.npz',**env.scene_snapshot())
    (out/'scenario.json').write_text(json.dumps(scenario,indent=2),encoding='utf-8')
    m=dict(metrics); m.update({'method':METHOD_NAME,'scenario_id':scenario['scenario_id'],'scenario_seed':scenario['source_seed'],'return':float(total),'checkpoint':str(Path(checkpoint).resolve()),'final_cell':list(map(int,env.pos)),'goal_cell':list(env.goal)})
    if checkpoint_meta:
        m.update(checkpoint_meta)
    (out/'metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    return m



def warmup_policy(agent, env, repeats=8):
    """Run unmeasured forward passes so first-use PyTorch overhead is excluded."""
    s,_=env.reset()
    mask=env.valid_action_mask()
    with torch.no_grad():
        for _ in range(int(repeats)):
            agent.select_action(s,0.0,valid_mask=mask)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='config_real_scale.json')
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--device',default='auto')
    ap.add_argument('--scenario-file',default='test_scenarios.json')
    ap.add_argument('--scenario-id',help='Optional single scenario id, e.g. test_01. Default: run all 20.')
    ap.add_argument('--output',default='results/proposed_ddqn_per_20_scenarios')
    args=ap.parse_args()

    cfg=json.load(open(args.config,'r',encoding='utf-8')); dev=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    env=RealScaleOffshoreEnv(cfg); agent=DDQNPERAgent(env.state_dim,env.action_dim,dev,cfg['agent'])
    payload=torch.load(args.checkpoint,map_location=dev)
    trained_with_mask=payload.get('agent_config',{}).get('feasibility_action_masking',False)
    if not bool(trained_with_mask):
        raise SystemExit('Checkpoint was not trained with feasibility_action_masking=True. Retrain with the revised training project; do not mix old unmasked checkpoints with this planner.')
    agent.load_checkpoint(payload)
    warmup_policy(agent,env,repeats=8)
    checkpoint_meta={
        'checkpoint_training_seed':payload.get('seed'),
        'checkpoint_episode':payload.get('episode'),
        'checkpoint_converged':payload.get('converged'),
        'checkpoint_convergence_episode':payload.get('convergence_episode'),
        'checkpoint_action_masking':payload.get('agent_config',{}).get('feasibility_action_masking'),
    }
    scenarios=load_suite(args.scenario_file)
    if args.scenario_id:
        scenarios=[s for s in scenarios if s['scenario_id']==args.scenario_id]
        if not scenarios: raise SystemExit(f'No scenario named {args.scenario_id}')

    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    # Freeze the exact manifest alongside the results. Future baselines must reuse the same test_scenarios.json.
    shutil.copy2(args.scenario_file,root/'test_scenarios_used.json')
    rows=[]
    print(f'Method={METHOD_NAME} | device={dev} | scenarios={len(scenarios)} | checkpoint={args.checkpoint}')
    print(f'Feasibility-aware action masking={agent.use_action_masking}; timing uses an unmeasured 8-forward-pass warm-up.')
    for i,sc in enumerate(scenarios,start=1):
        path,raw,total,m=run_one(env,agent,sc)
        scenario_out=root/sc['scenario_id']
        m=save_one(scenario_out,env,sc,path,raw,total,m,args.checkpoint,checkpoint_meta=checkpoint_meta)
        row={'method':METHOD_NAME}; row.update(scenario_row(sc)); row.update(m); rows.append(row)
        print(f'[{i:02d}/{len(scenarios):02d}] {sc["scenario_id"]}: success={m["success"]} L={m["path_length_m"]:.2f} m cost={m["normalized_motion_cost"]:.2f} time={m["planning_wall_time_s"]:.4f} s')

    write_csv(root/'all_scenarios_metrics.csv',rows)
    # This per-scenario file is the one to merge with Fan/A* later for paired statistical tests.
    write_csv(root/'paired_results_proposed.csv',rows)
    summary=make_paper_summary(rows)
    (root/'paper_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    write_csv(root/'paper_summary.csv',summary_rows(summary))
    print('\nPaper summary:')
    print(json.dumps(summary,indent=2))
    print(f'All outputs saved to {root.resolve()}')

if __name__=='__main__': main()
