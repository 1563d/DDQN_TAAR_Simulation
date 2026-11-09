import json
from src.real_scale_env import RealScaleOffshoreEnv
from src.scenario_suite import load_suite, scenario_signature

cfg=json.load(open('config_real_scale.json','r',encoding='utf-8'))
env=RealScaleOffshoreEnv(cfg)
print('state_dim=',env.state_dim,'action_dim=',env.action_dim,'grid=',env.grid.shape,'resolution_m=',env.resolution)
all_sigs=set()
for fname in ('validation_scenarios.json','test_scenarios.json'):
    suite=load_suite(fname)
    print(f'Validating {fname}: {len(suite)} scenarios')
    assert len(suite)==20
    for s in suite:
        env.apply_scenario(s,validate_ground=True)
        assert env._has_ground_path(env.start[:2],env.waypoint[:2]), s['scenario_id']
        assert env._has_3d_path(env.waypoint,env.goal), s['scenario_id']
        sig=scenario_signature(s)
        assert sig not in all_sigs, f'duplicate scenario: {s["scenario_id"]}'
        all_sigs.add(sig)
    print('  PASS')
print('All 40 validation/test scenarios are feasible and mutually unique.')
print('SCENARIO SUITE VALIDATION PASSED')
