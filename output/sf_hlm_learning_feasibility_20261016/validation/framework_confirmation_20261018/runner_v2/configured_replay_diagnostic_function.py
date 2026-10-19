def raw_audit(a,p):
    f,fa,fp=verify_runner(a,p)
    if (a.results_root/'raw_audit.json').exists():raise FileExistsError('Completed raw audit is immutable')
    import numpy as np
    from scripts.analysis.sf_hlm_framework_statistics import summary
    episodes=all_episodes(a,p);recomputed=summary(p,episodes);prior=b.read(a.results_root/'analysis/summary.json')
    for key in recomputed:
        if recomputed[key]!=prior[key]:raise ValueError('Full block statistics do not reproduce')
    old=f.original_args(fa,fp,p['blocks'][0]);r=f.followup(fa);runner=r.runner(old);original,rollout=r.initialize(old,runner);public_modules(a)
    models,actual_source,checkpoint,state=r.basis(old,runner,original,fp)
    import config
    from utils.sf_hlm_matching import feedback
    from marl_models.sf_hlm_v2 import service_view
    from utils.sf_hlm_software_delay import SoftwareServiceDelay
    from utils.canonical_experiment import exogenous_frame
    replay_steps=0;replay_requests=0;replay_records=[]
    for task in p['physical_replay_tasks']:
        receipt,steps=checked_steps(a,p,task);env,obs=runner.episode_env(task[2],task[3])
        with SoftwareServiceDelay(env,'measured') as hook:
            for row in steps:
                if payload_hash(exogenous_frame(env))!=row['exogenous_hash'] or sum(u.latent_request[0]<config.NUM_SERVICES for u in env.ues)!=row['potential']:raise ValueError('Physical replay external stream differs')
                _,masks,requests,rows=runner.base.prepare(env)
                if [q[2] for q in requests]!=row['request_ids'] or payload_hash(rows)!=row['public_rows_hash']:raise ValueError('Physical replay public predecision state differs')
                service_view(env,feedback(rows,row['assignment'],5,energy_reference=config.OFFLOAD_ENERGY_NORM_REF));actions=runner.base.actions_from_assignment(requests,row['assignment'],masks);motion=np.asarray(row['motion'],dtype=row['motion_dtype']);hook.arm(row['request_ids'],row['policy_seconds']);next_obs,rewards,metrics=env.step(motion,offloading_actions=actions);outcome=hook.finish()
                state=payload_hash([b.normal(next_obs),b.normal(rewards),b.normal(metrics),outcome['requests']])
                if b.normal(metrics)!=row['metrics'] or outcome['requests']!=row['requests'] or state!=row['state_hash']:raise ValueError('Full physical replay diverges from sealed timed trajectory')
                obs=np.asarray(next_obs,np.float32);replay_steps+=1;replay_requests+=len(row['requests'])
        replay_records.append(dict(task=task,steps_sha256=receipt['steps_sha256'],steps=len(steps)))
        b.write(a.results_root/'raw_audit_progress.json',dict(status='AUDITING',replayed_episodes=len(replay_records),expected=len(p['physical_replay_tasks']),replayed_steps=replay_steps))
    report=dict(status='DIAGNOSTIC_CONFIGURED_REPLAY_CONFIRMED_NOT_FORMAL_PASS',episodes=len(replay_records),steps=replay_steps,requests=replay_requests,source_hash=actual_source['source_hash'],config_hash=actual_source['config_hash'],only_change='Invoke original basis consumer configure before physical replay')
    b.write(a.validation_root/'replay_configuration_diagnostic.json',report);print(json.dumps(report))
