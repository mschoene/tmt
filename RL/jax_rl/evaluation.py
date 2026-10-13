from typing import Dict

import flax.linen as nn
import gymnasium as gym
import numpy as np
import jax.numpy as jnp
from jax import jit,vmap
from functools import partial

def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats

def rel_err(a,b):
    return jnp.sqrt(((a-b)**2).mean())/(jnp.sqrt((a**2).mean())+jnp.sqrt((b**2).mean()))#

# @partial(jit, static_argnames=('mean_fn',))
# @partial(jit, static_argnames=('mean_fn', 'kwargs'))
# @partial(jit, static_argnames=('mean_fn', 'kwargs', 'reprhos'))
# @partial(jit, static_argnames=('mean_fn', 'reprhos'))
def test_equivariance(x, params, mean_fn, kwargs, reprhos):
# def test_equivariance(x, params, mean_fn, kwargs, reprhos):
# def test_equivariance(x,params,mean_fn,kwargs,reprhos):
    st,inv_st = kwargs['state_transform'],kwargs['inv_state_transform']
    at,inv_at = kwargs['action_transform'],kwargs['inv_action_transform']
    srep = kwargs['state_rep']
    arep = kwargs['action_rep']
    # G = kwargs['symmetry_group']
    # g = G.samples(x.shape[0])
    # ring = vmap(srep.rho_dense)(g)
    # routg = vmap(arep.rho_dense)(g)
    # import pdb; pdb.set_trace()
    ring,routg = reprhos(x)
    gx = inv_st((ring@st(x)[...,None])[...,0])
    m1 = mean_fn(params,gx) #T=1
    m2 = mean_fn(params,x)
    return rel_err(m1,inv_at((routg@at(m2)[...,None])[...,0]))
    # logit_mean = dist
    # print(f"x shape {x.shape}")
    # print(f"Transformed x {st(x).shape} and f^-1(f(x))={inv_st(st(x)).shape}")
    # print(f"mean shape {logit_mean.shape}")
    # #TODO, use state transform and inv state transform
    # assert False


# JIT version for use in evaluation (standalone calls)
# @partial(jit, static_argnames=('mean_fn',))
# def test_equivariance(x, params, mean_fn, kwargs, reprhos):
#     """JIT-compiled version for evaluation."""
#     return test_equivariance_core(x, params, mean_fn, kwargs, reprhos)

def rpp_evaluate(agent: nn.Module,mean_fn, env: gym.Env,
             num_episodes: int,kwargs,reprhos, encoder=None) -> Dict[str, float]:
    #mean_fn = jit(lambda p,x: agent.sac.actor.apply_fn.apply({'params':p},x)._distribution._loc)
    stats = {'return': [], 'length': [],'equiv_err':[]}
    
    if hasattr(agent, 'sac'):
        params = agent.sac.actor.params
    else:
        params = agent.actor.params
    
    for _ in range(num_episodes):
        #equiv_errs = []
        observations = []
        observation, done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            # observation, _, done, info = env.step(action)
            ret = env.step(action)
            if len(ret) == 5:
                observation, reward, terminated, truncated, info = ret
                done = terminated or truncated
            else:
                observation, reward, done, info = ret

            observations.append(observation)
            #equiv_errs.append(test_equivariance(observation,agent.sac.actor.params,mean_fn,kwargs))

        for k in ['return','length']:
            stats[k].append(info['episode'][k])
        # stats['equiv_err'].append(test_equivariance(jnp.stack(observations,axis=0),\
        #             agent.sac.actor.params,mean_fn,kwargs,reprhos))
        # import pdb; pdb.set_trace()
        
        if encoder is not None: 
            observations = jnp.stack(observations, axis=0)

            observations = encoder(params, observations)
        
        stats['equiv_err'].append(test_equivariance(jnp.stack(observations,axis=0),\
                    params,mean_fn,kwargs,reprhos))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats