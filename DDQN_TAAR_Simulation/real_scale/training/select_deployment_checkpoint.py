from __future__ import annotations
import argparse, json, shutil
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description='Select a deployment checkpoint using validation data only.')
    ap.add_argument('--outputs',default='outputs')
    ap.add_argument('--seeds',nargs='+',type=int,default=[10,11,12,13,14])
    ap.add_argument('--destination',default='deployment_checkpoint/proposed_validation_selected.pt')
    args=ap.parse_args()

    candidates=[]
    for seed in args.seeds:
        root=Path(args.outputs)/f'seed_{seed:02d}'
        sp=root/'training_summary.json'
        cp=root/'checkpoints'/'best_model.pt'
        if not sp.exists() or not cp.exists():
            print(f'SKIP seed {seed}: missing {sp} or {cp}')
            continue
        s=json.loads(sp.read_text(encoding='utf-8'))
        candidates.append({
            'seed':seed,
            'converged':bool(s.get('converged',False)),
            'best_validation_success_rate':float(s.get('best_validation_success_rate',-1.0)),
            'best_validation_return_mean':float(s.get('best_validation_return_mean',-1e30)),
            'convergence_episode':s.get('convergence_episode'),
            'checkpoint':str(cp),
        })
    if not candidates:
        raise SystemExit('No eligible seed outputs found.')

    # Predeclared selection rule: converged policies first; then highest fixed-validation
    # success rate; then highest fixed-validation return; then lower seed only as a
    # deterministic final tie-breaker. The fixed test set is never read here.
    chosen=max(candidates,key=lambda x: (int(x['converged']),x['best_validation_success_rate'],x['best_validation_return_mean'],-x['seed']))
    dest=Path(args.destination); dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(chosen['checkpoint'],dest)
    report={
        'selection_basis':'validation only; fixed test_scenarios.json not read',
        'selection_rule':'converged first; highest best validation success rate; highest best validation return; lower seed as final deterministic tie-breaker',
        'candidates':candidates,
        'selected':chosen,
        'copied_to':str(dest),
    }
    (dest.parent/'selection_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
