import gym
import os
import sunblaze_envs


def make_env(env_id, process_index=0, outdir=None):
    env = sunblaze_envs.make(env_id)
    if outdir:
        env = sunblaze_envs.MonitorParameters(
            env,
            output_filename=os.path.join(outdir, "env-parameters-{}.json".format(process_index))
        )

    return env
