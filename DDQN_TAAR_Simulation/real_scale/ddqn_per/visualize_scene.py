import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.real_scale_env import RealScaleOffshoreEnv

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config_real_scale.json'); ap.add_argument('--path'); ap.add_argument('--output',default='scene_or_plan.png'); args=ap.parse_args()
    cfg=json.load(open(args.config,'r',encoding='utf-8')); env=RealScaleOffshoreEnv(cfg); res=env.resolution
    fig=plt.figure(figsize=(10,8)); ax=fig.add_subplot(111,projection='3d')
    cx,cy=env.scene_params['tower_center']; hz=env.scene_params['hub_z']; rr=env.scene_params['rotor_radius']
    ax.plot([cx*res,cx*res],[cy*res,cy*res],[0,hz*res],linewidth=5)
    import math
    for ang in (0,120,240):
        th=math.radians(ang); ax.plot([cx*res,(cx+rr*math.cos(th))*res],[cy*res,cy*res],[hz*res,(hz+rr*math.sin(th))*res],linewidth=3)
    s=np.asarray(env.start)*res; w=np.asarray(env.waypoint)*res; g=np.asarray(env.goal)*res
    ax.scatter(*s,s=80,marker='s'); ax.scatter(*w,s=80,marker='^'); ax.scatter(*g,s=120,marker='*')
    if args.path:
        P=np.loadtxt(args.path,delimiter=',',skiprows=1); P=np.atleast_2d(P); ax.plot(P[:,0],P[:,1],P[:,2],linewidth=2)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]'); ax.set_xlim(0,cfg['physical_extent_m'][0]); ax.set_ylim(0,cfg['physical_extent_m'][1]); ax.set_zlim(0,cfg['physical_extent_m'][2]); fig.tight_layout(); fig.savefig(args.output,dpi=300); print(Path(args.output).resolve())
if __name__=='__main__': main()
