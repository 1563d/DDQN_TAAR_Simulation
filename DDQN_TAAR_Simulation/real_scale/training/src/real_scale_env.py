from __future__ import annotations
from collections import deque
import heapq
import math
import numpy as np

class RealScaleOffshoreEnv:
    """Real-scale offshore inspection environment for PER-enhanced DDQN.

    The static offshore geometry is fixed. A scenario changes only the small-range
    task conditions used in the paper comparison: start, ground waypoint,
    inspection goal, and vessel-deck ground-obstacle layout.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.resolution = float(cfg['voxel_resolution_m'])
        self.size_x = int(cfg['size_x']); self.size_y = int(cfg['size_y']); self.size_z = int(cfg['size_z'])
        self.default_start = tuple(map(int, cfg['task']['start_cell']))
        self.default_waypoint = tuple(map(int, cfg['task']['ground_waypoint_cell']))
        self.default_goal = tuple(map(int, cfg['task']['inspection_goal_cell']))
        self.max_steps = int(cfg['task'].get('max_steps', 500))
        self.moves = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1) if (dx,dy,dz)!=(0,0,0)]
        self.action_dim = len(self.moves); self.state_dim = 46
        self.grid = np.zeros((self.size_z,self.size_y,self.size_x), dtype=np.int8)
        self.scene_params = {}
        self._build_static_scene()
        self._static_grid = self.grid.copy()
        self.current_scenario = self._default_scenario()
        self.apply_scenario(self.current_scenario, validate_ground=True)
        self.reset()

    def _build_static_scene(self):
        self.grid.fill(0)
        X,Y,Z = self.size_x,self.size_y,self.size_z
        # Sea surface is non-traversable in ground mode; only the two decks are cleared.
        self.grid[0,:,:] = 1
        self.grid[0,78:91,8:23] = 0
        self.grid[0,42:59,37:54] = 0
        # Turbine tower / nacelle / blades.
        cx,cy = map(int,self.cfg['turbine']['tower_center_cell'])
        hubz = int(round(float(self.cfg['turbine']['hub_height_m'])/self.resolution))
        rr = int(self.cfg['turbine']['rotor_radius_cells']); tr = int(self.cfg['turbine'].get('tower_radius_cells',1))
        self.grid[0:hubz,max(0,cy-tr):min(Y,cy+tr+1),max(0,cx-tr):min(X,cx+tr+1)] = 1
        self.grid[max(0,hubz-1):min(Z,hubz+2),max(0,cy-1):min(Y,cy+3),max(0,cx-2):min(X,cx+4)] = 1
        for val,ang in zip((2,3,4),(0,120,240)):
            self._create_blade(cx,cy,hubz,ang,rr,val)
        # Fixed maintenance crane. It is not part of the randomized ground layout.
        self.grid[1:12,69:72,28:31] = 1
        self.grid[10:12,62:72,28:31] = 1
        self.scene_params = {
            'tower_center': (cx,cy), 'hub_z': hubz, 'rotor_radius': rr,
            'vessel_deck': (8,22,78,90), 'turbine_deck': (37,53,42,58),
        }

    def _default_scenario(self):
        rcfg = self.cfg.get('scenario_randomization', {})
        obs=[]
        for spec in rcfg.get('ground_obstacles', []):
            omin=list(map(int,spec['base_min_cell'])); size=list(map(int,spec['size_cells']))
            obs.append({'name':spec['name'],'min_cell':omin,'max_cell':[omin[i]+size[i] for i in range(3)]})
        if not obs:
            obs=[
                {'name':'deck_equipment_A','min_cell':[12,80,0],'max_cell':[15,84,2]},
                {'name':'deck_equipment_B','min_cell':[16,85,0],'max_cell':[20,88,2]},
            ]
        return {
            'scenario_id':'base', 'source_seed':-1,
            'start_cell':list(self.default_start),
            'ground_waypoint_cell':list(self.default_waypoint),
            'inspection_goal_cell':list(self.default_goal),
            'ground_obstacles':obs,
        }

    def _create_blade(self,cx,cy,cz,angle,length,val,width=1):
        th=math.radians(angle); c,s=math.cos(th),math.sin(th)
        n=max(20,int(length*5))
        for u in np.linspace(2.0,float(length),n):
            x=int(round(cx+u*c)); z=int(round(cz+u*s))
            for yy in range(cy,min(self.size_y,cy+1)):
                for xi in range(x-width,x+width+1):
                    for zi in range(z-width,z+width+1):
                        if 0<=xi<self.size_x and 0<=yy<self.size_y and 0<=zi<self.size_z and self.grid[zi,yy,xi]==0:
                            self.grid[zi,yy,xi]=val

    def _refresh(self):
        self.grid_pad=np.pad(self.grid,((1,1),(1,1),(1,1)),mode='constant',constant_values=1)

    def apply_scenario(self, scenario, validate_ground=True):
        self.grid = self._static_grid.copy()
        self.current_scenario = scenario
        self.start = tuple(map(int, scenario['start_cell']))
        self.waypoint = tuple(map(int, scenario['ground_waypoint_cell']))
        self.goal = tuple(map(int, scenario['inspection_goal_cell']))
        for o in scenario.get('ground_obstacles', []):
            a=list(map(int,o['min_cell'])); b=list(map(int,o['max_cell']))
            x0,y0,z0=a; x1,y1,z1=b
            if not (0<=x0<x1<=self.size_x and 0<=y0<y1<=self.size_y and 0<=z0<z1<=self.size_z):
                raise ValueError(f'Invalid obstacle bounds: {o}')
            self.grid[z0:z1,y0:y1,x0:x1] = 1
        self._refresh()
        for name,p in [('start',self.start),('waypoint',self.waypoint),('goal',self.goal)]:
            x,y,z=p
            if not(0<=x<self.size_x and 0<=y<self.size_y and 0<=z<self.size_z):
                raise ValueError(f'{name} out of bounds: {p}')
            if self.grid[z,y,x] > 0:
                raise ValueError(f'{name} lies in occupied voxel: {p}')
        if validate_ground and not self._has_ground_path(self.start[:2], self.waypoint[:2]):
            raise ValueError('No ground path from randomized start to randomized ground waypoint')
        return self

    def reset(self, seed=None, scenario=None):
        # seed is accepted for API compatibility, but does NOT reseed global NumPy.
        if scenario is not None:
            self.apply_scenario(scenario, validate_ground=True)
        self.pos=tuple(self.start); self.steps=0; self.wp_reached=False; self.last_move_dir=None; self.collision_attempts=0
        return self._state(self.pos),{}

    def _distance(self,a,b): return float(math.dist(tuple(a),tuple(b)))

    def _has_ground_path(self,start_xy,target_xy,max_expansions=12000):
        sx,sy=map(int,start_xy); gx,gy=map(int,target_xy)
        if not(0<=sx<self.size_x and 0<=sy<self.size_y and 0<=gx<self.size_x and 0<=gy<self.size_y): return False
        if self.grid[0,sy,sx]>0 or self.grid[0,gy,gx]>0: return False
        q=deque([(sx,sy)]); vis=np.zeros((self.size_y,self.size_x),dtype=np.uint8); vis[sy,sx]=1; n=0
        while q and n<max_expansions:
            x,y=q.popleft(); n+=1
            if (x,y)==(gx,gy): return True
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx,ny=x+dx,y+dy
                if 0<=nx<self.size_x and 0<=ny<self.size_y and not vis[ny,nx] and self.grid[0,ny,nx]==0:
                    vis[ny,nx]=1; q.append((nx,ny))
        return False

    def _has_3d_path(self,start,goal,max_expansions=150000):
        start=tuple(map(int,start)); goal=tuple(map(int,goal))
        pq=[(math.dist(start,goal),0.0,start)]; g={start:0.0}; n=0
        while pq and n<max_expansions:
            _,base,p=heapq.heappop(pq); n+=1
            if base != g.get(p): continue
            if p==goal: return True
            for dx,dy,dz in self.moves:
                q=(p[0]+dx,p[1]+dy,p[2]+dz); x,y,z=q
                if not(0<=x<self.size_x and 0<=y<self.size_y and 0<=z<self.size_z): continue
                if self.grid[z,y,x]>0: continue
                ng=base+math.sqrt(dx*dx+dy*dy+dz*dz)
                if ng < g.get(q,1e30):
                    g[q]=ng; heapq.heappush(pq,(ng+math.dist(q,goal),ng,q))
        return False

    def _state(self,pos):
        x,y,z=map(int,pos); gx,gy,gz=(self.waypoint if not self.wp_reached else self.goal)
        dx,dy,dz=gx-x,gy-y,gz-z; ed=math.sqrt(dx*dx+dy*dy+dz*dz)+1e-9; md=abs(dx)+abs(dy)+abs(dz)
        blocked=0.0
        if z==0:
            cnt=0
            for qx,qy in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if not(0<=qx<self.size_x and 0<=qy<self.size_y) or self.grid[0,qy,qx]>0: cnt+=1
            blocked=cnt/4.0
        diag=math.sqrt((self.size_x-1)**2+(self.size_y-1)**2+(self.size_z-1)**2)
        man=max(1,(self.size_x-1)+(self.size_y-1)+(self.size_z-1))
        global20=np.array([
            x/max(1,self.size_x-1),y/max(1,self.size_y-1),z/max(1,self.size_z-1),
            gx/max(1,self.size_x-1),gy/max(1,self.size_y-1),gz/max(1,self.size_z-1),
            dx/ed,dy/ed,dz/ed,min(ed/diag,1.0),min(md/man,1.0),float(self.wp_reached),float(z==0),blocked,
            float(x==gx),float(y==gy),float(z==gz),float(abs(dx)>abs(dy)) if z==gz else 0.0,float(abs(dy)>=abs(dx)) if z==gz else 0.0,float(z!=gz)
        ],dtype=np.float32)
        cz,cy,cx=z+1,y+1,x+1; nb=self.grid_pad[cz-1:cz+2,cy-1:cy+2,cx-1:cx+2].reshape(-1); local=(np.delete(nb,13)>0).astype(np.float32)
        return np.concatenate([global20,local]).astype(np.float32)


    def valid_action_mask(self, pos=None):
        """Return a Boolean mask for actions executable from ``pos``.

        The mask mirrors the environment's hard feasibility checks used by
        :meth:`step`: an action is valid iff its destination voxel is inside
        the map and unoccupied. Because the sea surface is encoded as occupied
        except on traversable decks, the same mask also enforces the
        terrestrial/aerial domain feasibility represented by the voxel map.

        No look-ahead search or path heuristic is used here; this is only a
        one-step safety/feasibility filter over the original 26-action set.
        """
        if pos is None:
            pos = self.pos
        px, py, pz = map(int, pos)
        mask = np.zeros(self.action_dim, dtype=np.bool_)
        for i, (dx, dy, dz) in enumerate(self.moves):
            nx, ny, nz = px + dx, py + dy, pz + dz
            if not (0 <= nx < self.size_x and 0 <= ny < self.size_y and 0 <= nz < self.size_z):
                continue
            if self.grid[nz, ny, nx] > 0:
                continue
            mask[i] = True
        return mask

    def step(self,action):
        R_COLLISION=-5.0; R_WP=100.0; R_GOAL=300.0
        C_G=1.0; C_A=1.6; C_UP=2.0; C_DOWN=0.4; P_DIAG=-1.5; P_TURN=1.2; P_FLY=-6.0; B_TAKE=2.0; P_HIGH=-0.60; K1=2.5; K2=3.5
        self.steps+=1; dx,dy,dz=self.moves[int(action)]; px,py,pz=self.pos; nx,ny,nz=px+dx,py+dy,pz+dz
        oob=not(0<=nx<self.size_x and 0<=ny<self.size_y and 0<=nz<self.size_z); hit=False if oob else self.grid[nz,ny,nx]>0
        if oob or hit:
            self.collision_attempts+=1; r=R_COLLISION-(1.0 if pz==0 and hit else 0.0); done=self.steps>=self.max_steps
            return self._state(self.pos),float(r),done,False,{'collision':True,'wp_reached':self.wp_reached,'success':False}
        prev=self.pos; self.pos=(nx,ny,nz); move=(dx,dy,dz); L=math.sqrt(dx*dx+dy*dy+dz*dz); r=-(C_G if nz==0 else C_A)*L
        if dz>0: r-=C_UP*abs(dz)
        elif dz<0: r-=C_DOWN*abs(dz)
        if pz==0 and nz==0 and abs(dx)==1 and abs(dy)==1: r+=P_DIAG
        if self.last_move_dir is not None:
            pv=np.asarray(self.last_move_dir,dtype=float); cv=np.asarray(move,dtype=float); ca=np.clip(np.dot(pv,cv)/(np.linalg.norm(pv)*np.linalg.norm(cv)+1e-9),-1,1); r-=P_TURN*(math.acos(float(ca))/math.pi)
        if not self.wp_reached:
            gp=self._has_ground_path((px,py),(self.waypoint[0],self.waypoint[1]))
            if gp:
                if nz>0 or dz>0: r+=P_FLY*(1.0 if dz>0 else 0.5)
                r+=P_HIGH*max(nz,0)
            elif pz==0 and dz>0: r+=B_TAKE
            d0=self._distance(prev,self.waypoint); d1=self._distance(self.pos,self.waypoint); r+=K1*(d0-d1)
            if d1<=1.0:
                self.wp_reached=True; r+=R_WP+(15.0 if nz==0 else 0.0); self.last_move_dir=None
        else:
            d0=self._distance(prev,self.goal); d1=self._distance(self.pos,self.goal); r+=K2*(d0-d1)
            if d1<=0.5: r+=R_GOAL
        self.last_move_dir=move
        success=self.wp_reached and self._distance(self.pos,self.goal)<=0.5
        done=bool(success or self.steps>=self.max_steps); r=float(np.clip(r,-100.0,500.0))
        return self._state(self.pos),r,done,False,{'collision':False,'wp_reached':self.wp_reached,'success':success}

    def physical_xyz(self,p): return tuple(float(v)*self.resolution for v in p)

    def scene_snapshot(self):
        obs=[]
        for o in self.current_scenario.get('ground_obstacles',[]):
            obs.append(list(o['min_cell'])+list(o['max_cell']))
        return {
            'grid':self.grid.copy(),
            'start':np.asarray(self.start,dtype=np.int32),
            'waypoint':np.asarray(self.waypoint,dtype=np.int32),
            'goal':np.asarray(self.goal,dtype=np.int32),
            'ground_obstacles':np.asarray(obs,dtype=np.int32),
            'resolution_m':np.asarray(self.resolution),
        }
