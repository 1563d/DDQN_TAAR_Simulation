from __future__ import annotations
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import torch
from src.real_scale_env import RealScaleOffshoreEnv
from src.ddqn_per import DDQNPERAgent
from src.scenario_suite import load_suite, generate_feasible_scenario, scenario_signature, scenario_row


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def device_auto(): return 'cuda' if torch.cuda.is_available() else 'cpu'


def _capture_rng_state():
    s={'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.random.get_rng_state()}
    if torch.cuda.is_available(): s['torch_cuda']=torch.cuda.get_rng_state_all()
    return s


def _restore_rng_state(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.random.set_rng_state(s['torch_cpu'])
    if torch.cuda.is_available() and 'torch_cuda' in s: torch.cuda.set_rng_state_all(s['torch_cuda'])


def greedy_eval_suite(env, agent, scenarios):
    """Greedy evaluation on a frozen validation suite, not on the final test suite."""
    rng_state=_capture_rng_state(); trials=[]; representative_path=None; representative_scenario=None
    try:
        t_all=time.perf_counter()
        for j,sc in enumerate(scenarios, start=1):
            env.apply_scenario(sc, validate_ground=True)
            s,_=env.reset(); total=0.0; path=[env.pos]; t0=time.perf_counter()
            for _ in range(env.max_steps):
                valid_mask=env.valid_action_mask()
                a=agent.select_action(s,0.0,valid_mask=valid_mask)
                s,r,done,_,_=env.step(a); total+=r; path.append(env.pos)
                if done: break
            success=bool(env.wp_reached and env._distance(env.pos,env.goal)<=0.5)
            row={
                'trial':j,'scenario_id':sc['scenario_id'],'scenario_seed':sc['source_seed'],
                'success':int(success),'return':float(total),'steps':int(env.steps),
                'collisions':int(env.collision_attempts),
                'final_distance_m':float(env._distance(env.pos,env.goal)*env.resolution),
                'wall_time_s':float(time.perf_counter()-t0),
            }
            row.update(scenario_row(sc)); trials.append(row)
            if representative_path is None and success:
                representative_path=list(path); representative_scenario=sc['scenario_id']
        elapsed=time.perf_counter()-t_all
    finally:
        _restore_rng_state(rng_state)

    success=np.asarray([r['success'] for r in trials],float)
    returns=np.asarray([r['return'] for r in trials],float)
    steps=np.asarray([r['steps'] for r in trials],float)
    collisions=np.asarray([r['collisions'] for r in trials],float)
    finald=np.asarray([r['final_distance_m'] for r in trials],float)
    return {
        'success_rate':float(success.mean()),
        'return_mean':float(returns.mean()),'return_std':float(returns.std(ddof=1)) if len(returns)>1 else 0.0,
        'steps_mean':float(steps.mean()),'steps_std':float(steps.std(ddof=1)) if len(steps)>1 else 0.0,
        'collisions_mean':float(collisions.mean()),
        'final_distance_mean_m':float(finald.mean()),
        'eval_trials':len(trials),'wall_time_s':float(elapsed),
        'path':representative_path,'representative_scenario_id':representative_scenario,
    }, trials


def write_csv(path: Path, rows):
    if not rows: return
    keys=[]
    for row in rows:
        for k in row:
            if k not in keys: keys.append(k)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='config_real_scale.json')
    ap.add_argument('--episodes',type=int,help='Override maximum training episodes.')
    ap.add_argument('--seed',type=int)
    ap.add_argument('--device',default='auto')
    ap.add_argument('--no-early-stop',action='store_true')
    args=ap.parse_args()

    cfg=json.load(open(args.config,'r',encoding='utf-8')); tc=cfg['training']; rcfg=cfg['scenario_randomization']
    max_episodes=int(args.episodes or tc.get('max_episodes',3000)); seed=int(args.seed if args.seed is not None else tc['seed'])
    set_seed(seed); dev=device_auto() if args.device=='auto' else args.device

    env=RealScaleOffshoreEnv(cfg); agent=DDQNPERAgent(env.state_dim,env.action_dim,dev,cfg['agent'])
    validation_scenarios=load_suite(tc.get('validation_scenario_file','validation_scenarios.json'))
    heldout_test_scenarios=load_suite(tc.get('reserved_test_scenario_file','test_scenarios.json'))
    reserved={scenario_signature(s) for s in validation_scenarios+heldout_test_scenarios}

    out=Path('outputs')/f'seed_{seed:02d}'; ck=out/'checkpoints'; ck.mkdir(parents=True,exist_ok=True)
    # Copy the exact manifests used by this run for auditability.
    (out/'validation_scenarios_used.json').write_text(Path(tc.get('validation_scenario_file','validation_scenarios.json')).read_text(encoding='utf-8'),encoding='utf-8')
    (out/'heldout_test_manifest_reserved.json').write_text(Path(tc.get('reserved_test_scenario_file','test_scenarios.json')).read_text(encoding='utf-8'),encoding='utf-8')

    rows=[]; eval_rows=[]; eval_trial_rows=[]; best_score=(-1.0,-1e18); best_path=None; best_path_scenario=None
    eps0=float(tc['epsilon_start']); eps1=float(tc['epsilon_end']); decay=max(1,int(tc.get('epsilon_decay_episodes',900)))
    eval_interval=int(tc.get('eval_interval',20)); conv_sr=float(tc.get('convergence_success_rate',0.95)); conv_required=int(tc.get('convergence_consecutive_evals',5)); checkpoint_interval=int(tc.get('checkpoint_interval',100))
    train_base=int(rcfg.get('train_scenario_seed_base',30000)); stride=int(rcfg.get('train_seed_stride',100000))
    shared_schedule=bool(rcfg.get('shared_training_scenario_schedule',True))
    require_train_3d=bool(rcfg.get('training_scenario_requires_3d_path',True))

    consecutive=0; max_consecutive=0; converged=False; convergence_episode=None; actual_final_episode=0
    print(f'Device={dev} | max_episodes={max_episodes} | 20 held-out test scenarios are RESERVED and never used for checkpoint selection')
    print(f'Training: one randomized feasible scenario per episode; validation: {len(validation_scenarios)} frozen scenarios every {eval_interval} episodes')
    print(f'Training scenario schedule: {"SHARED across formal seeds" if shared_schedule else "seed-specific"}; 3-D feasibility check={require_train_3d}')
    print(f'Convergence: validation SR >= {conv_sr:.2f} for {conv_required} consecutive evaluations; epsilon {eps0:.2f}->{eps1:.2f} over {decay} episodes')
    print(f'Feasibility-aware action masking: {agent.use_action_masking} (applied to epsilon-greedy exploration, greedy evaluation, and masked DDQN next-action selection)')

    for ep in range(1,max_episodes+1):
        actual_final_episode=ep; eps=eps1+(eps0-eps1)*max(0.0,1.0-(ep-1)/decay)
        # Common-random-number protocol: by default all formal training seeds see the
        # same randomized scenario at the same episode. Only network initialization,
        # epsilon-greedy actions, replay sampling, and optimizer stochasticity differ.
        scenario_seed=(train_base+ep) if shared_schedule else (train_base+seed*stride+ep)
        sc=generate_feasible_scenario(cfg,env,scenario_seed,f'train_ep{ep:04d}',reserved_signatures=reserved,require_3d_path=require_train_3d)
        env.apply_scenario(sc,validate_ground=True); s,_=env.reset()
        total=0.0; losses=[]; t0=time.perf_counter(); zero_valid_events=0; valid_action_counts=[]
        for _ in range(env.max_steps):
            valid_mask=env.valid_action_mask()
            n_valid=int(np.count_nonzero(valid_mask)); valid_action_counts.append(n_valid)
            if n_valid==0: zero_valid_events+=1
            a=agent.select_action(s,eps,valid_mask=valid_mask)
            ns,r,done,_,_=env.step(a)
            next_valid_mask=env.valid_action_mask()
            agent.buf.push(s,a,r,ns,done,next_valid_mask=next_valid_mask)
            s=ns; total+=r
            v=agent.step_and_maybe_update()
            if v is not None: losses.append(v)
            if done: break
        success=int(env.wp_reached and env._distance(env.pos,env.goal)<=0.5)
        tr={
            'episode':ep,'epsilon':eps,'return':total,'success':success,'steps':env.steps,'collisions':env.collision_attempts,
            'final_distance_m':env._distance(env.pos,env.goal)*env.resolution,
            'loss':float(np.mean(losses)) if losses else np.nan,'wall_time_s':time.perf_counter()-t0,
            'action_masking_enabled':int(agent.use_action_masking),
            'mean_valid_actions':float(np.mean(valid_action_counts)) if valid_action_counts else 0.0,
            'min_valid_actions':int(min(valid_action_counts)) if valid_action_counts else 0,
            'zero_valid_action_events':int(zero_valid_events),
        }
        tr.update(scenario_row(sc)); rows.append(tr)

        if (ep%eval_interval==0) or ep==max_episodes:
            ev,trials=greedy_eval_suite(env,agent,validation_scenarios); ev['episode']=ep
            clean={k:v for k,v in ev.items() if k!='path'}; eval_rows.append(clean)
            for r in trials: r=dict(r); r['episode']=ep; eval_trial_rows.append(r)
            score=(ev['success_rate'],ev['return_mean'])
            # Update the convergence streak BEFORE printing it. The previous version
            # printed the old streak, so a failed validation could misleadingly show
            # e.g. `valSR=0.80 pass=2/5` even though the streak was reset immediately after.
            consecutive=consecutive+1 if ev['success_rate']>=conv_sr else 0
            max_consecutive=max(max_consecutive,consecutive)
            failed_ids=[r['scenario_id'] for r in trials if int(r['success'])==0]
            clean['pass_streak']=int(consecutive)
            clean['max_pass_streak']=int(max_consecutive)
            clean['n_failed_validation_scenarios']=int(len(failed_ids))
            clean['failed_validation_scenario_ids']=';'.join(failed_ids)
            # Replace the just-appended aggregate row with the augmented diagnostics.
            eval_rows[-1]=clean
            print(f'[ep {ep:4d}] trainR={total:8.2f} eps={eps:.3f} trainSucc={success} | valSR={ev["success_rate"]:.2f} valR={ev["return_mean"]:.2f}+/-{ev["return_std"]:.2f} pass={consecutive}/{conv_required} fail={failed_ids}')
            if score>best_score:
                best_score=score; best_path=ev['path']; best_path_scenario=ev['representative_scenario_id']
                torch.save(agent.checkpoint(seed=seed,episode=ep,config=cfg,best_validation=clean),ck/'best_model.pt')
            if consecutive>=conv_required:
                converged=True; convergence_episode=ep
                torch.save(agent.checkpoint(seed=seed,episode=ep,config=cfg,convergence_validation=clean),ck/'converged_model.pt')
                print(f'Convergence satisfied at episode {ep}.')
                if not args.no_early_stop:
                    print('Early stopping enabled.'); break
        if ep%checkpoint_interval==0:
            torch.save(agent.checkpoint(seed=seed,episode=ep,config=cfg),ck/f'episode_{ep:04d}.pt')

    torch.save(agent.checkpoint(seed=seed,episode=actual_final_episode,config=cfg,converged=converged,convergence_episode=convergence_episode),ck/'final_model.pt')
    write_csv(out/'training_log.csv',rows); write_csv(out/'fixed_validation_log.csv',eval_rows); write_csv(out/'fixed_validation_trials.csv',eval_trial_rows)
    if best_path:
        arr=np.asarray(best_path,float); np.savetxt(out/'best_validation_path_cells.csv',arr,delimiter=',',header='x,y,z',comments=''); np.savetxt(out/'best_validation_path_m.csv',arr*env.resolution,delimiter=',',header='x_m,y_m,z_m',comments='')
    summary={
        'training_seed':seed,'max_episodes':max_episodes,'actual_final_episode':actual_final_episode,'converged':bool(converged),'convergence_episode':convergence_episode,
        'early_stopping_enabled':not args.no_early_stop,'validation_suite_size':len(validation_scenarios),'heldout_test_suite_size':len(heldout_test_scenarios),
        'test_suite_used_for_training_or_selection':False,
        'test_manifest_reused_after_method_revision':True,'convergence_success_rate_threshold':conv_sr,'convergence_consecutive_evaluations':conv_required,
        'eval_interval':eval_interval,'epsilon_decay_episodes':decay,'best_validation_success_rate':float(best_score[0]),'best_validation_return_mean':float(best_score[1]),
        'best_validation_representative_scenario_id':best_path_scenario,
        'max_validation_pass_streak':int(max_consecutive),
        'shared_training_scenario_schedule':bool(shared_schedule),
        'training_scenario_requires_3d_path':bool(require_train_3d),
        'feasibility_action_masking':bool(agent.use_action_masking),
        'action_mask_definition':'destination voxel must be in-bounds and unoccupied; no search/look-ahead heuristic',
        'training_scenario_seed_rule':'train_scenario_seed_base + episode' if shared_schedule else 'train_scenario_seed_base + training_seed*stride + episode',
    }
    (out/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'Finished: {out.resolve()} | converged={converged} at {convergence_episode}')

if __name__=='__main__': main()
