from emlp.reps import V,T,Rep
from emlp.groups import Z,S,SO,Group
from scipy.spatial.transform import Rotation
import jax.numpy as jnp
import numpy as np
import jax

# NumPy 2.0 compatibility: provide deprecated alias used by emlp
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

class PseudoScalar(Rep):
    is_regular=False
    def __init__(self,G=None):
        self.G=G
        # self.concrete = (self.G is not None)
    @property
    def concrete(self):
        return self.G is not None
    def __call__(self,G):
        return PseudoScalar(G)
    def size(self):
        return 1
    def __str__(self):
        return "P"
    def rho(self,M):
        sign = jnp.linalg.slogdet(M@jnp.eye(M.shape[0]))[0]
        return sign*jnp.eye(1)
    def __eq__(self,other):
        return type(self)==type(other) and self.G==other.G
    def __hash__(self):
        return hash((type(self),self.G))
    @property
    def T(self):
        return self

# M = M@jnp.eye(M.shape[0])
        # i1 = (M[:2,:2]==0).all().astype(int)
        # M2 = M[:2,:2] if i1==0 else M[2:,:2]
        # i2 = M2[0,0]
        # M1 = jnp.eye(2)[[i1,1-i1]]

class SwimmerState(Rep):
    is_regular=False
    def __init__(self):
        self.G = Z(2)*Z(2)
    def __call__(self,G):
        assert G==self.G
        return self
    def size(self):
        return 10
    def __str__(self):
        return "SwimmerX"
    def rho(self,M):
        M = M@jnp.eye(M.shape[0])
        i1 = (M[:2,:2]==0).all().astype(int)
        # jnp.where instead of Python if — safe under vmap/jit
        M2 = jnp.where(i1 == 0, M[:2,:2], M[2:,:2])
        _swap = jnp.array([[0.,1.],[1.,0.]])
        M1 = jnp.where(i1 == 0, jnp.eye(2), _swap)
        s1  =2*M1[0,0]-1
        s2 = 2*M2[0,0]-1
        s = jnp.eye(1)
        p = s1*jnp.eye(1)
        rhos = [s,p,s1*M2,s,p,s,p,s1*M2]
        return jax.scipy.linalg.block_diag(*rhos)
    def __eq__(self,other):
        return type(self)==type(other) and self.G==other.G
    def __hash__(self):
        return hash((type(self),self.G))
    @property
    def T(self):
        return self

class SwimmerStateAngle(Rep):
    """Raw 8-dim Swimmer obs — no cos/sin expansion needed for Z(2)×Z(2).

    Layout: [θ_torso, θ1, θ2, vx, vy, ω_torso, ω1, ω2]

    gen0 (y-reflection): diag(-1,-1,-1,+1,-1,-1,-1,-1)
    gen1 (link swap):    block_diag(1, swap₂, 1, 1, 1, swap₂)

    Works with `state_transform=Id` — no preprocessing required.
    Equivariant ground truth in angle space:
      G0(s) = [-θ_torso, -θ1, -θ2,  vx, -vy, -ω_torso, -ω1, -ω2]
      G1(s) = [ θ_torso,  θ2,  θ1,  vx,  vy,  ω_torso,  ω2,  ω1]
    """
    is_regular = False

    def __init__(self):
        self.G = Z(2)*Z(2)

    def __call__(self, G):
        assert G == self.G
        return self

    def size(self):
        return 8

    def rho(self, M):
        M = M @ jnp.eye(M.shape[0])
        i1 = (M[:2, :2] == 0).all().astype(int)
        M2 = jnp.where(i1 == 0, M[:2, :2], M[2:, :2])
        _swap = jnp.array([[0., 1.], [1., 0.]])
        M1 = jnp.where(i1 == 0, jnp.eye(2), _swap)
        s1 = 2 * M1[0, 0] - 1   # -1 for gen0 (y-flip), +1 for gen1 (link swap)
        s = jnp.eye(1)
        return jax.scipy.linalg.block_diag(
            s1 * s,   # θ_torso: pseudoscalar
            s1 * M2,  # [θ1, θ2]: negate (gen0) or swap (gen1)
            s,        # vx: true scalar
            s1 * s,   # vy: pseudoscalar
            s1 * s,   # ω_torso: pseudoscalar
            s1 * M2,  # [ω1, ω2]: same as [θ1, θ2]
        )

    def __str__(self):
        return "SwimmerStateAngle"

    def __eq__(self, other):
        return type(self) == type(other) and self.G == other.G

    def __hash__(self):
        return hash((type(self), self.G))

    @property
    def T(self):
        return self


class SwimmerAction(SwimmerState):
    def size(self):
        return 2
    def __str__(self):
        return "SwimmerA"
    def rho(self, M):
        M = M @ jnp.eye(M.shape[0])
        i1 = (M[:2, :2] == 0).all().astype(int)
        M2 = jnp.where(i1 == 0, M[:2, :2], M[2:, :2])
        _swap = jnp.array([[0., 1.], [1., 0.]])
        M1 = jnp.where(i1 == 0, jnp.eye(2), _swap)
        s1 = 2*M1[0, 0] - 1
        return s1 * M2

class SwimmerActionStd(SwimmerAction):

    def size(self):
        return 2
    def __str__(self):
        return "SwimmerAstd"
    def rho(self, M):
        M = M @ jnp.eye(M.shape[0])
        i1 = (M[:2, :2] == 0).all().astype(int)
        M2 = jnp.where(i1 == 0, M[:2, :2], M[2:, :2])
        return M2

class D(Group):
    def __init__(self,k):
        translation = np.eye(4)[np.array([3,0,1,2])][None]
        reflection = np.eye(4)[np.array([1,0,3,2])][None]
        self.discrete_generators = np.concatenate((translation,reflection))
        super().__init__(k)

# can this cause a bug because of pointing to the same rep object?


import numpy as np
import jax.numpy as jnp
from jax import jit
def vector_dual(v):
    v1 = jnp.stack([0*v[...,0],-v[...,2],v[...,1]],axis=-1)
    v2 = jnp.stack([v[...,2],0*v[...,1],-v[...,0]],axis=-1)
    v3 = jnp.stack([-v[...,1],v[...,0],0*v[...,2]],axis=-1)
    return jnp.stack([v1,v2,v3],-2)

def quat2rot(q):
    q = q/jnp.sqrt((q**2).sum(-1))[...,None]
    q0,v = jnp.split(q,[1],axis=-1)
    v_cross = vector_dual(v)
    R = jnp.eye(3)-2*q0[...,None]*v_cross+2*v_cross@v_cross
    return R.T



def ant_state_transform(x):
    """ Converts the quaternion in state vector to a rotation matrix"""
    z,q,angs,vcom,w,angv,forces = jnp.split(x,[1,5,13,16,19,27],axis=-1)
    R  =quat2rot(q).reshape(*q.shape[:-1],-1)
    #R = Rotation.from_quat(q).as_matrix().reshape(*q.shape[:-1],-1)
    #Rw = vector_dual(w).reshape(*w.shape[:-1],-1)
    return jnp.concatenate([z,R,angs,vcom,w,angv],-1) #remove forces

def ant_inv_state_transform(x):
    """ converts the """
    z,R,angs,vcom,w,angv,forces = jnp.split(x,[1,5+5,13+5,16+5,19+5,27+5],axis=-1)
    R = R.reshape(*R.shape[:-1],3,3)
    q = np.roll(Rotation.from_matrix(R).as_quat(),1,axis=-1)
    return jnp.concatenate([z,q,angs,vcom,w,angv,forces],-1)


ant_state_perm = np.array([0,1,2,3,4,5,7,9,11,6,8,10,12,13,14,15,16,17,18,
                            19,21,23,25,20,22,24,26])
inv_ant_state_perm = np.argsort(ant_state_perm)
ant_action_perm = np.array([0,2,4,6,1,3,5,7])
inv_ant_action_perm = np.argsort(ant_action_perm)

def ant_state_transform2(x):
    """ groups legs into 4"""
    flips = np.ones(27)
    return x[...,ant_state_perm]*flips
def ant_inv_state_transform2(x):
    y = x[...,inv_ant_state_perm]
    flips = np.ones(27)
    return y*flips

def ant_action_transform2(a):
    """ groups legs into 4 """
    return a[...,ant_action_perm]
def ant_inv_action_transform2(a):
    return a[...,inv_ant_action_perm]

# Ant-v5: obs=105 = qpos(13) + qvel(14) + cfrc_ext(78, 13 bodies x6)
# Bodies: torso(1) | FL_hip(2) FL_aux(3) FL_foot(4) | FR_hip(5) FR_aux(6) FR_foot(7)
#       | BL_hip(8) BL_aux(9) BL_foot(10) | BR_hip(11) BR_aux(12) BR_foot(13)
# cfrc starts at obs index 27; group same link types across 4 legs for Z(4)
def _build_ant_v5_state_perm():
    base = list(ant_state_perm)  # first 27: qpos+qvel already grouped
    torso = list(range(27, 33))
    hip  = list(range(33,39)) + list(range(51,57)) + list(range(69,75)) + list(range(87,93))
    aux  = list(range(39,45)) + list(range(57,63)) + list(range(75,81)) + list(range(93,99))
    foot = list(range(45,51)) + list(range(63,69)) + list(range(81,87)) + list(range(99,105))
    return np.array(base + torso + hip + aux + foot)

ant_v5_state_perm = _build_ant_v5_state_perm()
inv_ant_v5_state_perm = np.argsort(ant_v5_state_perm)

def ant_v5_state_transform(x):
    return x[..., ant_v5_state_perm]
def ant_v5_inv_state_transform(x):
    return x[..., inv_ant_v5_state_perm]

# Humanoid-v5: obs=348 = qpos(22)+qvel(23)+cinert(130)+cvel(78)+qfrc_actuator(17)+cfrc_ext(78)
# Z(2) bilateral symmetry: group right/left limb pairs consecutively.
# Joints: abdomen_z/y/x(3) | r_hip_x/z/y,r_knee(4) | l_hip_x/z/y,l_knee(4) | r_s1/s2/e(3) | l_s1/s2/e(3)
# Bodies: torso(0) lwaist(1) pelvis(2) [invariant] | r_thigh(3)↔l_thigh(6) | r_shin(4)↔l_shin(7)
#       | r_foot(5)↔l_foot(8) | r_ua(9)↔l_ua(11) | r_la(10)↔l_la(12)
def _build_humanoid_v5_state_perm():
    qpos_p = [0, 1,2,3,4, 5,6,7, 8,12, 9,13, 10,14, 11,15, 16,19, 17,20, 18,21]
    qvel_p = [22+i for i in [0,1,2, 3,4,5, 6,7,8, 9,13, 10,14, 11,15, 12,16, 17,20, 18,21, 19,22]]
    inv_b = [0,1,2]
    pairs = [(3,6),(4,7),(5,8),(9,11),(10,12)]
    def block_perm(start, n):
        p = []
        for b in inv_b:
            p += list(range(start + b*n, start + b*n + n))
        for r,l in pairs:
            p += list(range(start + r*n, start + r*n + n))
            p += list(range(start + l*n, start + l*n + n))
        return p
    cinert_p = block_perm(45, 10)
    cvel_p   = block_perm(175, 6)
    qfrc_p   = [253+i for i in [0,1,2, 3,7, 4,8, 5,9, 6,10, 11,14, 12,15, 13,16]]
    cfrc_p   = block_perm(270, 6)
    return np.array(qpos_p + qvel_p + cinert_p + cvel_p + qfrc_p + cfrc_p)

humanoid_v5_state_perm = _build_humanoid_v5_state_perm()
inv_humanoid_v5_state_perm = np.argsort(humanoid_v5_state_perm)
humanoid_v5_action_perm = np.array([0,1,2, 3,7, 4,8, 5,9, 6,10, 11,14, 12,15, 13,16])
inv_humanoid_v5_action_perm = np.argsort(humanoid_v5_action_perm)

def humanoid_v5_state_transform(x):
    return x[..., humanoid_v5_state_perm]
def humanoid_v5_inv_state_transform(x):
    return x[..., inv_humanoid_v5_state_perm]
def humanoid_v5_action_transform(a):
    return a[..., humanoid_v5_action_perm]
def humanoid_v5_inv_action_transform(a):
    return a[..., inv_humanoid_v5_action_perm]

def _walker_state_transform(x):
    """ groups left right joint correspondences together """
    (y,orient,rh,rk,ra,lh,lk,la,vcomx,\
        vcomy,angvel,vrh,vrk,vra,vlh,vlk,vla) = jnp.split(x,np.arange(1,x.shape[-1]),axis=-1)
    reordered_tuple = (y,orient,rh,lh,rk,lk,ra,la,vcomx,\
        vcomy,angvel,vrh,vlh,vrk,vlk,vra,vla)
    return jnp.concatenate(reordered_tuple,-1)

walker_perm = _walker_state_transform(np.arange(17))
inv_walker_perm = np.argsort(walker_perm)

def walker_state_transform(x):
    return x[...,walker_perm]

def inv_walker_state_transform(x):
    return x[...,inv_walker_perm]

def _inv_walker_action_transform(a):
    rh,lh,rk,lk,ra,la= jnp.split(a,np.arange(1,a.shape[-1]),axis=-1)
    return jnp.concatenate([rh,rk,ra,lh,lk,la],-1)

inv_walker_action_perm = _inv_walker_action_transform(np.arange(6))
walker_action_perm = np.argsort(inv_walker_action_perm)

def inv_walker_action_transform(a):
    return a[...,inv_walker_action_perm]

def walker_action_transform(a):
    return a[...,walker_action_perm]

def humanoid_state_transform(x):
    z,q,x_rest,extra_info = jnp.split(x,[1,5,45],axis=-1)
    R  =quat2rot(q).reshape(*q.shape[:-1],-1)
    out = jnp.concatenate([z,R,x_rest],-1)
    return out

def inv_humanoid_state_transform(x):
    z,R,x_rest,extra_info = jnp.split(x,[1,5+5,45+5],axis=-1)
    R = R.reshape(*R.shape[:-1],3,3)
    q = np.roll(Rotation.from_matrix(R).as_quat(),1,axis=-1)
    return jnp.concatenate([z,q,x_rest],-1)

leg_arm_perm = np.array([0,4,1,5,2,6,3,7,8,11,9,12,10,13])
def _humanoid_state_perm2(x):
    stuff,legsarms,vs,legsvarmsv = jnp.split(x,[13,13+14,27+6+3])
    return jnp.concatenate([stuff,legsarms[leg_arm_perm],vs,legsvarmsv[leg_arm_perm]])

humanoid_state_perm = _humanoid_state_perm2(np.arange(50))
inv_humanoid_state_perm = np.argsort(humanoid_state_perm)
humanoid_action_perm = np.concatenate([np.arange(3),3+leg_arm_perm])
inv_humanoid_action_perm = np.argsort(humanoid_action_perm)
#print(humanoid_state_perm.shape,humanoid_action_perm.shape)


def swimmer_state_transform2(x):
    """ Expands the orientations into vectors"""
    orient_vec = jnp.stack([jnp.cos(x[...,0]),jnp.sin(x[...,0])],-1)
    orient_vel = jnp.stack([-jnp.sin(x[...,0])*x[...,5],jnp.cos(x[...,0])*x[...,5]],-1)
    return jnp.concatenate([orient_vec,x[...,1:5],orient_vel,x[...,6:]],-1)

def swimmer_inv_state_transform2(x):
    """ Expands the orientations into vectors"""
    orient_angle = jnp.arctan2(x[...,1],x[...,0])[...,None]
    orient_angvel = (x[...,6]/(-x[...,1]))[...,None]#(x[...,6]**2+x[...,7]**2)
    return jnp.concatenate([orient_angle,x[...,2:6],orient_angvel,x[...,8:]],-1)



# ============================================================================
# Reusable Symmetry Transformation Utilities
# ============================================================================

def rotation_matrix_2d(theta):
    """Returns 2D rotation matrix for angle theta (works with JAX)"""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s], [s, c]])

def apply_so2_rotation_state(state, theta):
    """
    Apply SO(2) rotation to 2D state (works for batched states)
    
    Args:
        state: [x, y] or [batch, x, y] or [x, y, vx, vy] etc.
               Assumes state is pairs of 2D vectors
        theta: rotation angle in radians
    
    Returns:
        rotated state with same shape
    """
    R = rotation_matrix_2d(theta)
    
    # Handle different input shapes
    if state.ndim == 1:
        # Single state: reshape to pairs of 2D vectors
        n_pairs = state.shape[0] // 2
        state_reshaped = state.reshape(n_pairs, 2)
        rotated = (R @ state_reshaped.T).T
        return rotated.reshape(-1)
    else:
        # Batched states
        n_pairs = state.shape[-1] // 2
        state_reshaped = state.reshape(*state.shape[:-1], n_pairs, 2)
        rotated = jnp.einsum('ij,...kj->...ki', R, state_reshaped)
        return rotated.reshape(*state.shape)

def apply_so2_rotation_action(action, theta):
    """
    Apply SO(2) rotation to 2D action
    
    Args:
        action: [fx, fy] or [batch, fx, fy]
        theta: rotation angle in radians
    
    Returns:
        rotated action with same shape
    """
    R = rotation_matrix_2d(theta)
    
    if action.ndim == 1:
        return R @ action
    else:
        return jnp.einsum('ij,...j->...i', R, action)

def apply_z2_reflection(x):
    """Apply Z(2) reflection symmetry: x → -x"""
    return -x

def verify_so2_equivariance(state, action, next_state, theta, dynamics_fn):
    """
    Verify SO(2) equivariance: rotating inputs should rotate outputs
    
    Args:
        state, action, next_state: trajectory data
        theta: rotation angle to test
        dynamics_fn: function (state, action) → next_state
    
    Returns:
        max error across all dimensions
    """
    # Rotate inputs
    state_rot = apply_so2_rotation_state(state, theta)
    action_rot = apply_so2_rotation_action(action, theta)
    
    # Apply dynamics
    next_state_rot = dynamics_fn(state_rot, action_rot)
    
    # Rotate expected output
    next_state_expected = apply_so2_rotation_state(next_state, theta)
    
    # Compare
    error = jnp.max(jnp.abs(next_state_rot - next_state_expected))
    return error

def verify_z2_equivariance(state, action, next_state, dynamics_fn):
    """
    Verify Z(2) equivariance: flipping inputs should flip outputs
    
    Args:
        state, action, next_state: trajectory data
        dynamics_fn: function (state, action) → next_state
    
    Returns:
        max error across all dimensions
    """
    # Flip inputs
    state_flipped = apply_z2_reflection(state)
    action_flipped = apply_z2_reflection(action)
    
    # Apply dynamics
    next_state_flipped = dynamics_fn(state_flipped, action_flipped)
    
    # Flip expected output
    next_state_expected = apply_z2_reflection(next_state)
    
    # Compare
    error = jnp.max(jnp.abs(next_state_flipped - next_state_expected))
    return error

#####
### Representation Definitions for Environments
#####

P = PseudoScalar()
vector3 = T(1)+T(0)
matrix3 = vector3**2
pseodovector3 = P*vector3
s = T(0)
# # L/R symmetries only
# environment_symmetries['Humanoid-v2'] = {
#     'state_rep':T(0)+matrix3+17*T(0)+vector3+pseodovector3+\
#             17*T(0),
#     'state_transform':humanoid_state_transform,
#     'action_rep':17*T(0),
#     'inv_action_transform':Id,
#     'symmetry_group':SO(2),
#     'action_space':"continuous",
# }
Id = lambda x:x 

#import collections
# environment_symmetries = collections.defaultdict(lambda: {
#     'state_transform':Id,
#     'inv_state_transform':Id,
#     'action_transform':Id,
#     'inv_action_transform':Id})
legarmrep = P*T(1)+3*T(1)+P*T(1)+2*T(1)
environment_symmetries={
    # 'Humanoid-v2': {
    #     'state_rep':T(0)+matrix3+17*T(0)+vector3+pseodovector3+\
    #         17*T(0),
    #     'state_transform':humanoid_state_transform,
    #     'inv_state_transform':inv_humanoid_state_transform,
    #     'action_rep':17*T(0),
    #     'action_transform':Id,
    #     'inv_action_transform':Id,
    #     'symmetry_group':SO(2),
    #     'action_space':"continuous",
    # },
    'Humanoid-v2': {
        'state_rep':s+matrix3+s+s+P+legarmrep+\
            vector3+pseodovector3+s+s+P+legarmrep,
        'state_transform':lambda x: humanoid_state_transform(x)[...,humanoid_state_perm],
        'inv_state_transform':lambda x: inv_humanoid_state_transform(x[...,inv_humanoid_state_perm]),
        'action_rep':2*s+P+legarmrep,
        'action_transform':lambda a: a[...,humanoid_action_perm],
        'inv_action_transform':lambda a: a[...,inv_humanoid_action_perm],
        'symmetry_group':Z(2),
        'action_space':"continuous",
        'middle_rep':136*T(0)+60*T(1),
    },
    'Ant-v2': {
        'state_rep':T(0)+4*T(0)+2*T(1)+6*T(0)+\
            2*T(1),#+14*vector3+14*pseodovector3,
        'state_transform':ant_state_transform2,
        'inv_state_transform':ant_inv_state_transform2,
        'action_rep':2*T(1),
        'action_transform':ant_action_transform2,
        'inv_action_transform':ant_inv_action_transform2,
        'symmetry_group':Z(4),
        'action_space':"continuous",
        'middle_rep':136*T(0)+30*T(1),
    },
    # 'Swimmer-v2':{ # Focus just on LR symmetry now, to add front back later
    #     'state_rep':P+P+P+2*T(0)+P+P+P, # shoud vcom swap?
    #     'state_transform':Id,
    #     'inv_state_transform':Id,
    #     'action_rep':2*P,
    #     'action_std_rep':2*T(0),
    #     'action_transform':Id,
    #     'inv_action_transform':Id,
    #     'symmetry_group':Z(2),
    #     'action_space':"continuous",
    #     'middle_rep':126*T(0)+55*T(1)+5*T(2),
    # },
    'Swimmer-v2':{ # Focus just on LR symmetry now, to add front back later
        'state_rep':SwimmerState(), # shoud vcom swap?
        'state_transform':swimmer_state_transform2,
        'inv_state_transform':swimmer_inv_state_transform2,
        'action_rep':SwimmerAction(),
        'action_std_rep':SwimmerActionStd(),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2)*Z(2),
        'action_space':"continuous",
        #'middle_rep':108*T(0)+25*T(1)+3*T(2),
    },
    'Swimmer-v5':{ # same obs/act structure as Swimmer-v2; Z(2)×Z(2) bilateral symmetry
        'state_rep':SwimmerState(),
        'state_transform':swimmer_state_transform2,
        'inv_state_transform':swimmer_inv_state_transform2,
        'action_rep':SwimmerAction(),
        'action_std_rep':SwimmerActionStd(),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2)*Z(2),
        'action_space':"continuous",
    },
    # Angle-space variant: 8-dim raw obs, no cos/sin expansion.
    # Equivariant ground truth matches the untransformed state directly.
    # Use --env_name=Swimmer-v5-angle together with this entry.
    'Swimmer-v5-angle':{
        'state_rep':SwimmerStateAngle(),
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':SwimmerAction(),
        'action_std_rep':SwimmerActionStd(),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2)*Z(2),
        'action_space':"continuous",
    },
    'Walker2d-v2':{
        'state_rep':2*T(0)+3*T(1)+3*T(0)+3*T(1),
        'state_transform':walker_state_transform,
        'inv_state_transform':inv_walker_state_transform,
        'action_rep':3*T(1),
        'action_transform':walker_action_transform,
        'inv_action_transform':inv_walker_action_transform,
        'symmetry_group':Z(2),
        'action_space':"continuous",
        # 'middle_rep':136*T(0)+60*T(1),
        'middle_rep':76*T(0)+90*T(1),
    },    
    'Walker2d-v5':{
        'state_rep':2*T(0)+3*T(1)+3*T(0)+3*T(1), # T0 trivial repr, T1 sign flip or was it permutation
        'state_transform':walker_state_transform,
        'inv_state_transform':inv_walker_state_transform,
        'action_rep':3*T(1),
        # 'action_std_rep':6*T(0),  # added to match 6-dim action space
        'action_transform':walker_action_transform,
        'inv_action_transform':inv_walker_action_transform,
        'symmetry_group':Z(2),
        'action_space':"continuous",
        # 'middle_rep':136*T(0)+60*T(1),
    },
    # Ant-v5: obs=105 (qpos13+qvel14+cfrc_ext78), act=8
    # Z(4) cyclic permutation of 4 legs (FL→FR→BR→BL).
    # T(1) is the 4D cyclic-permutation representation of Z(4).
    # state_rep: z(T0) + quat(4*T0, approx) + 2 joint-type groups(2*T1=8)
    #          + body-vel(6*T0) + 2 joint-vel groups(2*T1=8)
    #          + torso cfrc(6*T0) + 3 leg-link-type cfrc groups(18*T1=72)
    'Ant-v5': {
        'state_rep': T(0) + 4*T(0) + 2*T(1) + 6*T(0) + 2*T(1) + 6*T(0) + 18*T(1),
        'state_transform': ant_v5_state_transform,
        'inv_state_transform': ant_v5_inv_state_transform,
        'action_rep': 2*T(1),
        'action_transform': ant_action_transform2,
        'inv_action_transform': ant_inv_action_transform2,
        'symmetry_group': Z(4),
        'action_space': "continuous",
    },
    # Humanoid-v5: obs=348, act=17. Z(2) bilateral (left↔right).
    # state_rep pieces (T0=1D, T1=2D swap pair):
    #   qpos: z(T0)+quat(4*T0,approx)+abdomen(3*T0)+7 LR joint-pairs(7*T1)
    #   qvel: xyz_vel(3*T0)+ang_vel(3*T0)+abd_vel(3*T0)+7 LR vel-pairs(7*T1)
    #   cinert: 3 invariant bodies(30*T0) + 5 LR body-pairs(50*T1)
    #   cvel:   3 invariant(18*T0) + 5 LR pairs(30*T1)
    #   qfrc:   abdomen(3*T0) + 7 LR pairs(7*T1)
    #   cfrc:   3 invariant(18*T0) + 5 LR pairs(30*T1)
    'Humanoid-v5': {
        'state_rep': (T(0) + 4*T(0) + 3*T(0) + 7*T(1)   # qpos=22
                    + 3*T(0) + 3*T(0) + 3*T(0) + 7*T(1)  # qvel=23
                    + 30*T(0) + 50*T(1)                    # cinert=130
                    + 18*T(0) + 30*T(1)                    # cvel=78
                    + 3*T(0) + 7*T(1)                      # qfrc=17
                    + 18*T(0) + 30*T(1)),                  # cfrc=78
        'state_transform': humanoid_v5_state_transform,
        'inv_state_transform': humanoid_v5_inv_state_transform,
        'action_rep': 3*T(0) + 7*T(1),
        'action_transform': humanoid_v5_action_transform,
        'inv_action_transform': humanoid_v5_inv_action_transform,
        'symmetry_group': Z(2),
        'action_space': "continuous",
    },
    # HalfCheetah-v5 and Hopper-v5 have identical obs/act dims to v2
    'HalfCheetah-v5':{
        'state_rep':T(0) + 8*P + T(0) + 7*P,
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':6*P,
        'action_std_rep':6*T(0),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"continuous",
    },
    'Hopper-v5':{
        'state_rep':T(0)+4*P+(P+T(0)) + 4*P,
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':3*P,
        'action_std_rep':3*T(0),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"continuous",
    },
    'Hopper-v2':{ #
        'state_rep':T(0)+4*P+(P+T(0)) + 4*P, # shoud vcom swap?
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':3*P,
        'action_std_rep':3*T(0),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"continuous",
        'middle_rep':126*T(0)+55*T(1)+5*T(2),
    },
    'HalfCheetah-v2':{
        'state_rep':T(0) + 8*P + T(0) + 7*P,
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':6*P,
        'action_std_rep':6*T(0),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"continuous",
        'middle_rep':126*T(0)+55*T(1)+5*T(2),
    },
    'InclinedCartpole-v0':{
        'state_rep':P + P + P + P,
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':T(1),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"discrete",
    },
    'CartPole-v0':{
        'state_rep':P + P + P + P,
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':T(1),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"discrete",
    },
    'CartPole-v1':{
        'state_rep':P + P + P + P,
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':T(1),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"discrete",
    },
}

environment_symmetries.update({
    'HalfCheetahFull-v0':{
        'state_rep':P+T(0) + 7*P +P+ T(0) + 7*P,
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':6*P,
        'action_std_rep':6*T(0),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"continuous",
        #'angular_dims':[2,3,4,5,6,7,8],
    },
    'HopperFull-v0':{ #
        'state_rep':(P+T(0))+4*P+(P+T(0)) + 4*P, # shoud vcom swap?
        'state_transform':Id,
        'inv_state_transform':Id,
        'action_rep':3*P,
        'action_std_rep':3*T(0),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2),
        'action_space':"continuous",
        #'angular_dims':[2,3,4,5],
    },
    'Walker2dFull-v0':{
        'state_rep':3*T(0)+3*T(1)+3*T(0)+3*T(1),
        'state_transform':walker_state_transform,
        'inv_state_transform':inv_walker_state_transform,
        'action_rep':3*T(1),
        'action_transform':walker_action_transform,
        'inv_action_transform':inv_walker_action_transform,
        'symmetry_group':Z(2),
        'action_space':"continuous",
        #'angular_dims':[2,3,4,5,6,7,8],
    },
    'SwimmerFull-v0':{ # Focus just on LR symmetry now, to add front back later
        'state_rep':SwimmerState(), # shoud vcom swap?
        'state_transform':swimmer_state_transform2,
        'inv_state_transform':swimmer_inv_state_transform2,
        'action_rep':SwimmerAction(),
        'action_std_rep':SwimmerActionStd(),
        'action_transform':Id,
        'inv_action_transform':Id,
        'symmetry_group':Z(2)*Z(2),
        'action_space':"continuous",
        #'angular_dims':[2,3,4],
    },


        'ParticleCircle2D-v0': {
        'state_rep': 2*V,   # [x,y] position vector + [vx,vy] velocity vector # V is the standard 2D vector representation that rotates under SO(2)
        'state_transform': Id,
        'inv_state_transform': Id,
        'action_rep': V,    # [fx,fy] force vector
        'action_std_rep': 2*T(0),
        'action_transform': Id,
        'inv_action_transform': Id,
        'symmetry_group': SO(2),
        'action_space': "continuous",
    },

    # 3D particle in sphere: state is position+velocity -> 2 copies of the 3D vector rep
    # action is 3D vector (even if env ignores it, policy still outputs it)
    "ParticleCircle3D-v0": {
        "symmetry_group": SO(3),
        "state_rep": lambda G: 2 * V(G),     # (x,y,z) and (vx,vy,vz)
        "action_rep": lambda G: V(G),        # (ax,ay,az)
        "state_transform": lambda x: x,
        "inv_state_transform": lambda x: x,
        "action_transform": lambda u: u,
        "inv_action_transform": lambda u: u,
        "action_space": "continuous",
    },

    # Reacher2D obs is end-effector position (x,y): 2D vector rep under SO(2)
    # action is joint torques: DOES NOT transform under world rotations -> scalars
    # "Reacher2D-v0": {
    #     "symmetry_group": SO(2),
    #     "state_rep": lambda G: V(G),         # ee position (x,y)
    #     "action_rep": lambda G: 2 * T(0),    # (tau1, tau2) invariant scalars
    #     "state_transform": lambda x: x,
    #     "inv_state_transform": lambda x: x,
    #     "action_transform": lambda u: u,
    #     "inv_action_transform": lambda u: u,
    #     "action_space": "continuous",
    # },
        "Reacher2D-v0": {
        "symmetry_group": SO(2),
        "state_rep": lambda G: 3 * V(G) + 2 * T(0), # goal position, joint positions and torques
        "action_rep": lambda G: 2 * T(0),
        "state_transform": lambda x: x,
        "inv_state_transform": lambda x: x,
        "action_transform": lambda u: u,
        "inv_action_transform": lambda u: u,
        "action_space": "continuous",
    },
    # Reacher-v5 (MuJoCo): 10-dim obs, 2-dim action, SO(2) symmetry.
    # Raw obs: [cosθ1, sinθ1, cosθ2, sinθ2, tgt_x, tgt_y, dθ1, dθ2, ee_x-tgt_x, ee_y-tgt_y]
    # Vectors under world rotation: (cosθ1,sinθ1), (tgt_x,tgt_y), (ee_x-tgt_x,ee_y-tgt_y)
    # Scalars (invariant): cosθ2, sinθ2 (relative angle), dθ1, dθ2 (angular vels)
    # Reorder for contiguous EMLP blocks: [vectors..., scalars...]
    "Reacher-v5": {
        "symmetry_group": SO(2),
        "state_rep": lambda G: 3 * V(G) + 4 * T(0),
        # Gymnasium layout: [cos_q0, cos_q1, sin_q0, sin_q1, tgt_x, tgt_y, vel0, vel1, ee_x-tgt_x, ee_y-tgt_y]
        # Reorder to contiguous vector pairs: (cos_q0,sin_q0), (tgt_x,tgt_y), (ee_x-tgt_x,ee_y-tgt_y), then scalars
        "state_transform": lambda x: x[..., np.array([0,2, 4,5, 8,9, 1,3, 6,7])],
        "inv_state_transform": lambda x: x[..., np.argsort(np.array([0,2, 4,5, 8,9, 1,3, 6,7]))],
        "action_rep": lambda G: 2 * T(0),   # joint torques: scalars
        "action_transform": lambda u: u,
        "inv_action_transform": lambda u: u,
        "action_space": "continuous",
    },
})

# ---------------------------------------------------------------------------
# TriFingerPro — Z(3) cyclic-finger symmetry
# State: pos(9) + vel(9) + torque(9) + tip_force(3) = 30 dims
# Action: torque(9)
# Joints are stored finger-major [f0j0,f0j1,f0j2, f1j0,...]; we reorder to
# joint-type-major [j0_f0,j0_f1,j0_f2, j1_f0,...] so each T(1) block is
# contiguous (one joint type across all three fingers, cycling as Z(3)).
# ---------------------------------------------------------------------------
_trifinger_joint_perm = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8])  # finger-major → type-major

def _build_trifinger_state_perm():
    p = np.concatenate([
        _trifinger_joint_perm,           # pos   [0:9]
        _trifinger_joint_perm + 9,       # vel   [9:18]
        _trifinger_joint_perm + 18,      # torq  [18:27]
        np.array([27, 28, 29]),          # tip   [27:30] already finger-ordered
    ])
    return p

trifinger_state_perm     = _build_trifinger_state_perm()
inv_trifinger_state_perm = np.argsort(trifinger_state_perm)
trifinger_action_perm    = _trifinger_joint_perm
inv_trifinger_action_perm = np.argsort(trifinger_action_perm)

def trifinger_state_transform(x):
    return x[..., trifinger_state_perm]

def inv_trifinger_state_transform(x):
    return x[..., inv_trifinger_state_perm]

def trifinger_action_transform(a):
    return a[..., trifinger_action_perm]

def inv_trifinger_action_transform(a):
    return a[..., inv_trifinger_action_perm]

environment_symmetries.update({
    # obs = pos(9)+vel(9)+torque(9)+tip_force(3) = 30; act = torque(9)
    # T(1) for Z(3) is the 3-dim cyclic-permutation rep.
    # After reordering to type-major each 3-element block is one T(1).
    "TriFingerPro-v0": {
        "state_rep":  10 * T(1),  # 30 = 10 × 3
        "action_rep":  3 * T(1),  # 9  =  3 × 3
        "state_transform":     trifinger_state_transform,
        "inv_state_transform": inv_trifinger_state_transform,
        "action_transform":    trifinger_action_transform,
        "inv_action_transform": inv_trifinger_action_transform,
        "symmetry_group": Z(3),
        "action_space": "continuous",
    },
    # Position-control variant: same state (30-dim), action = target joint positions (9-dim).
    # Joint limits differ per joint type but are identical across fingers, so Z(3) holds.
    # Actions normalised to [-1,1] per joint using (pos - center) / half_range.
    "TriFingerPro-position-v0": {
        "state_rep":  10 * T(1),
        "action_rep":  3 * T(1),
        "state_transform":     trifinger_state_transform,
        "inv_state_transform": inv_trifinger_state_transform,
        "action_transform":    trifinger_action_transform,
        "inv_action_transform": inv_trifinger_action_transform,
        "symmetry_group": Z(3),
        "action_space": "continuous",
    },
})

# halfcheetah_rep2 = {
#     'state_rep':2*T(0)+3*T(1)+3*T(0)+3*T(1),
#     'state_transform':walker_state_transform,
#     'action_rep':3*T(1),
#     'inv_action_transform':inv_walker_action_transform,
#     'symmetry_group':Z(2),
# }
# comment back in to use the 2nd rep version for half cheetah
# environment_symmetries['HalfCheetah-v2']=halfcheetah_rep2


# ---------------------------------------------------------------------------
# Ground-truth Lie algebra generators and LieSD evaluation metrics
# ---------------------------------------------------------------------------

_J2 = np.array([[0., -1.], [1., 0.]])  # so(2) generator

_SO3_GENERATORS = [
    np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]]),   # Jx
    np.array([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]]),   # Jy
    np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]]),   # Jz
]

_SO2_ENVS = frozenset({
    'ParticleCircle2D-v0', 'Reacher2D-v0',
})
_SO3_ENVS = frozenset({'ParticleCircle3D-v0'})


def _block_diag_generators(base_gens, n_dim, block_size):
    """Tile each generator in base_gens as block-diagonal on an n_dim-dim space."""
    K = len(base_gens)
    result = np.zeros((K, n_dim, n_dim))
    n_blocks = n_dim // block_size
    for k, J in enumerate(base_gens):
        for i in range(n_blocks):
            s = i * block_size
            result[k, s:s + block_size, s:s + block_size] = J
    return result


def get_ground_truth_discrete_symmetries(env_name: str, n_s: int, n_a: int):
    """Return list of (P_s, P_a) ground-truth discrete symmetry matrices, or None.

    ParticleCircle2D (state=[x,y,vx,vy], action=[fx,fy]) has three discrete
    symmetries from the full O(2) symmetry group:
      - sign flip x  : (x,y,vx,vy,fx,fy) → (-x, y,-vx, vy,-fx, fy)
      - sign flip y  : (x,y,vx,vy,fx,fy) → ( x,-y, vx,-vy, fx,-fy)
      - permutation  : (x,y,vx,vy,fx,fy) → ( y, x, vy, vx, fy, fx)
    """
    if env_name == 'ParticleCircle2D-v0' and n_s == 4 and n_a == 2:
        P_s_flip_x = np.diag([-1., 1., -1., 1.])
        P_a_flip_x = np.diag([-1., 1.])
        P_s_flip_y = np.diag([1., -1., 1., -1.])
        P_a_flip_y = np.diag([1., -1.])
        P_s_perm   = np.array([[0.,1.,0.,0.],[1.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]])
        P_a_perm   = np.array([[0.,1.],[1.,0.]])
        return [
            (P_s_flip_x, P_a_flip_x),
            (P_s_flip_y, P_a_flip_y),
            (P_s_perm,   P_a_perm),
        ]
    # ParticleCircle3D: state=[x,y,z,vx,vy,vz] (6D), action=[ax,ay,az] (3D).
    # Full group is SO(3); finite discrete subgroup = hyperoctahedral B₃ (signed
    # permutations of 3 axes, order 48).  Generators listed here:
    #   flip_x, flip_y, flip_z, swap_xy
    if env_name == 'ParticleCircle3D-v0' and n_s == 6 and n_a == 3:
        def _s6(d): return np.diag(d).astype(np.float32)
        def _swap6(i, j):
            P = np.eye(6, dtype=np.float32)
            P[i, i] = P[j, j] = 0.;  P[i, j] = P[j, i] = 1.
            P[i+3, i+3] = P[j+3, j+3] = 0.;  P[i+3, j+3] = P[j+3, i+3] = 1.
            return P
        def _swap3(i, j):
            P = np.eye(3, dtype=np.float32)
            P[i, i] = P[j, j] = 0.;  P[i, j] = P[j, i] = 1.
            return P
        P_s_flip_x = _s6([-1., 1., 1., -1., 1., 1.]);  P_a_flip_x = np.diag([-1., 1., 1.]).astype(np.float32)
        P_s_flip_y = _s6([ 1.,-1., 1.,  1.,-1., 1.]);  P_a_flip_y = np.diag([ 1.,-1., 1.]).astype(np.float32)
        P_s_flip_z = _s6([ 1., 1.,-1.,  1., 1.,-1.]);  P_a_flip_z = np.diag([ 1., 1.,-1.]).astype(np.float32)
        P_s_swap_xy = _swap6(0, 1);                     P_a_swap_xy = _swap3(0, 1)
        return [
            (P_s_flip_x,  P_a_flip_x),
            (P_s_flip_y,  P_a_flip_y),
            (P_s_flip_z,  P_a_flip_z),
            (P_s_swap_xy, P_a_swap_xy),
        ]
    if env_name in ('TriFingerPro-v0', 'TriFingerPro-position-v0') and n_s == 30 and n_a == 9:
        # Z(3) cyclic permutation: one finger-cycle step in original coords.
        # Permutation matrices are in the original (finger-major) coordinate system.
        def _perm_matrix(perm):
            n = len(perm)
            P = np.zeros((n, n))
            for i, j in enumerate(perm):
                P[i, j] = 1.0
            return P
        # one cyclic step: f0→f1→f2→f0 in original finger-major layout
        s_perm_one = np.concatenate([
            np.array([3,4,5, 6,7,8, 0,1,2]),        # pos
            np.array([3,4,5, 6,7,8, 0,1,2]) + 9,    # vel
            np.array([3,4,5, 6,7,8, 0,1,2]) + 18,   # torq
            np.array([28, 29, 27]),                  # tip
        ])
        a_perm_one = np.array([3,4,5, 6,7,8, 0,1,2])
        P_s1 = _perm_matrix(s_perm_one)
        P_a1 = _perm_matrix(a_perm_one)
        P_s2 = P_s1 @ P_s1  # two steps = inverse one step
        P_a2 = P_a1 @ P_a1
        return [(P_s1, P_a1), (P_s2, P_a2)]
    # Swimmer-v5: obs=[θ_body, θ_r1, θ_r2, ẋ, ẏ, θ̇_body, θ̇_r1, θ̇_r2], act=[τ_r1, τ_r2]
    #
    # gen0 — y-reflection (mirror across x-axis):
    #   Angles are CCW-positive; reflecting across x maps CCW→CW, so every angle
    #   negates. ẋ (forward speed) is preserved; ẏ negates. Both torques negate to
    #   produce the mirrored motion. Pure diagonal, no link swap:
    #     θ_body→-θ_body, θ_r1→-θ_r1, θ_r2→-θ_r2, ẋ→ẋ, ẏ→-ẏ,
    #     θ̇_body→-θ̇_body, θ̇_r1→-θ̇_r1, θ̇_r2→-θ̇_r2, τ_r1→-τ_r1, τ_r2→-τ_r2
    #
    # gen1 — link permutation (swap r1↔r2, no sign change):
    #   Both links are identical (same mass/length/damping), so relabeling them
    #   while swapping their torques leaves the instantaneous rigid-body dynamics
    #   invariant. Note: this is not a spatial transformation — it has the character
    #   of a time-reversal of the joint sequence (in a periodic gait, r1 and r2
    #   play swapped roles half a cycle apart). Whether MuJoCo respects this as an
    #   exact symmetry depends on whether the joint ordering affects any asymmetric
    #   internal term (e.g. joint damping applied in sequence); for identical links
    #   it should hold, but treat with caution for non-random policies.
    #
    # gen0*gen1 — combined flip+swap:
    #   θ_r1↔-θ_r2, θ_r2↔-θ_r1, etc. This is the "swimming in the mirrored
    #   direction with swapped link roles" symmetry.
    if env_name in ('Swimmer-v5', 'Swimmer-v5-angle') and n_s == 8 and n_a == 2:
        # gen0: y-reflection — pure negation, no swap
        g0_s = np.diag([-1., -1., -1., 1., -1., -1., -1., -1.]).astype(np.float32)
        g0_a = np.diag([-1., -1.]).astype(np.float32)
        # gen1: link swap — θ1↔θ2, ω1↔ω2, τ1↔τ2 (no sign change)
        g1_s = np.eye(8, dtype=np.float32)
        g1_s[1, 1] = g1_s[2, 2] = 0.; g1_s[1, 2] = g1_s[2, 1] = 1.
        g1_s[6, 6] = g1_s[7, 7] = 0.; g1_s[6, 7] = g1_s[7, 6] = 1.
        g1_a = np.array([[0., 1.], [1., 0.]], dtype=np.float32)
        # gen0*gen1: combined flip+swap
        g01_s = g0_s @ g1_s
        g01_a = g0_a @ g1_a
        return [(g0_s, g0_a), (g1_s, g1_a), (g01_s, g01_a)]
    # Walker2d-v5: obs=[y, orient, rh, rk, ra, lh, lk, la, vcomx, vcomy, angvel,
    #                   vrh, vrk, vra, vlh, vlk, vla], act=[rh, rk, ra, lh, lk, la]
    # Z(2) bilateral: swap right↔left legs.  Pure permutation, no sign flips
    # (2D sagittal-plane walker — bilateral is a relabeling, not a reflection).
    if env_name == 'Walker2d-v5' and n_s == 17 and n_a == 6:
        s_perm = np.arange(17)
        for r, l in [(2, 5), (3, 6), (4, 7), (11, 14), (12, 15), (13, 16)]:
            s_perm[r], s_perm[l] = l, r
        P_s = np.zeros((17, 17), dtype=np.float32)
        for i, j in enumerate(s_perm):
            P_s[i, j] = 1.0
        a_perm = np.array([3, 4, 5, 0, 1, 2])
        P_a = np.zeros((6, 6), dtype=np.float32)
        for i, j in enumerate(a_perm):
            P_a[i, j] = 1.0
        return [(P_s, P_a)]
    # CartPole-v1: state=[x, ẋ, θ, θ̇], action=scalar ∈ {0,1} centered to {-0.5,+0.5}.
    # Z(2) left-right reflection: negate all state dims, negate centered action.
    if env_name == 'CartPole-v1' and n_s == 4 and n_a == 1:
        return [(-np.eye(4, dtype=np.float32), -np.eye(1, dtype=np.float32))]
    return None


def compute_discrete_symmetry_error(candidates, gt_symmetries, threshold: float = 0.5):
    """Evaluate discovered discrete symmetries against ground truth.

    For each GT symmetry, compute the minimum normalised Frobenius distance to
    any discovered candidate (using hard_P_s / hard_P_a).  A GT symmetry is
    "covered" when that distance is below `threshold` (normalised by sqrt(n_s)
    and sqrt(n_a) respectively).

    Two different ±1-valued matrices have a normalised distance ≥ 1.0, so
    threshold=0.5 accepts near-exact matches only.

    Returns dict with:
      eval/discrete_recall        – fraction of GT symmetries covered
      eval/discrete_count_error   – |num_found - num_gt|
      eval/discrete_mean_min_dist – average min distance over GT symmetries
    """
    n_gt = len(gt_symmetries)
    n_found = len(candidates)
    if n_gt == 0:
        return {}
    if n_found == 0:
        return {
            "eval/discrete_recall":         0.0,
            "eval/discrete_count_error":    float(n_gt),
            "eval/discrete_mean_min_dist":  2.0,
            "eval/discrete_max_min_dist":   2.0,
            "eval/discrete_combined_error": 2.0 + float(n_gt),
        }

    min_dists = []
    for P_s_gt, P_a_gt in gt_symmetries:
        P_s_gt = np.asarray(P_s_gt, dtype=np.float64)
        P_a_gt = np.asarray(P_a_gt, dtype=np.float64)
        n_s, n_a = P_s_gt.shape[0], P_a_gt.shape[0]
        best = float("inf")
        for cand in candidates:
            P_s = np.asarray(cand["hard_P_s"], dtype=np.float64)
            P_a = np.asarray(cand["hard_P_a"], dtype=np.float64)
            d_s = np.linalg.norm(P_s - P_s_gt) / np.sqrt(n_s)
            d_a = np.linalg.norm(P_a - P_a_gt) / np.sqrt(n_a)
            best = min(best, 0.5 * (d_s + d_a))
        min_dists.append(best)

    recall = float(sum(d < threshold for d in min_dists)) / n_gt
    mean_min_dist = float(np.mean(min_dists))
    max_min_dist  = float(np.max(min_dists))   # worst-case GT element
    count_error = float(abs(n_found - n_gt))
    return {
        "eval/discrete_recall":         recall,
        "eval/discrete_count_error":    count_error,
        "eval/discrete_mean_min_dist":  mean_min_dist,
        "eval/discrete_max_min_dist":   max_min_dist,
        "eval/discrete_combined_error": mean_min_dist + count_error,
    }


def get_ground_truth_generators(env_name: str, n_s: int, n_a: int):
    """Return (L_s_gt, L_a_gt) ground-truth Lie algebra generators.

    Each is an ndarray of shape (K_gt, n, n).  Returns (None, None) for
    discrete-only groups (Z(2)) where no continuous generator exists.
    """
    if env_name in _SO2_ENVS:
        L_s_gt = _block_diag_generators([_J2], n_s, block_size=2)   # (1, n_s, n_s)
        L_a_gt = _block_diag_generators([_J2], n_a, block_size=2)   # (1, n_a, n_a)
        return L_s_gt, L_a_gt
    if env_name in _SO3_ENVS:
        L_s_gt = _block_diag_generators(_SO3_GENERATORS, n_s, block_size=3)  # (3, n_s, n_s)
        L_a_gt = _block_diag_generators(_SO3_GENERATORS, n_a, block_size=3)  # (3, n_a, n_a)
        return L_s_gt, L_a_gt
    if env_name == 'Reacher-v5' and n_s == 10 and n_a == 2:

        L_s_gt = np.zeros((1, 10, 10), dtype=np.float64)
        # Gymnasium Reacher-v5 obs: [cos_q0, cos_q1, sin_q0, sin_q1, tgt_x, tgt_y, vel0, vel1, ee_x-tgt_x, ee_y-tgt_y]
        # [0, 2]  (cos_q0, sin_q0): absolute shoulder angle, rotates under world SO(2)
        L_s_gt[0][np.ix_([0, 2], [0, 2])] = _J2
        # [2:4]  (cos θ2, sin θ2): relative angle, INVARIANT — do NOT add _J2
        # [4:6]  (tgt_x, tgt_y): world-frame target, rotates
        L_s_gt[0, 4:6, 4:6] = _J2
        # [6:8]  (dθ1/dt, dθ2/dt): scalar angular velocities, invariant
        # [8:10] (ee_x - tgt_x, ee_y - tgt_y): world-frame 2D error vector, rotates
        L_s_gt[0, 8:10, 8:10] = _J2
        L_a_gt = np.zeros((1, 2, 2), dtype=np.float64)  # torques invariant

        # # Reacher-v5 obs: [cos θ1, sin θ1, cos θ2, sin θ2, tgt_x, tgt_y,
        # #                  dθ1/dt, dθ2/dt, ee_x-tgt_x, ee_y-tgt_y]
        # # SO(2) acts on the four 2D position/error blocks; angular velocities
        # # [6:8] are scalars (invariant). Actions (joint torques) are also scalars.
        # L_s_gt = np.zeros((1, 10, 10), dtype=np.float64)
        # for i in [0, 2, 4, 8]:  # vector-rep blocks; skip [6:8] (angular velocities)
        #     L_s_gt[0, i:i + 2, i:i + 2] = _J2
        # L_a_gt = np.zeros((1, 2, 2), dtype=np.float64)  # torques are scalars
        return L_s_gt, L_a_gt
    return None, None


def _generators_to_matrix(L):
    """L: (K, n, n) → V: (n^2, K) with unit-norm columns."""
    K = L.shape[0]
    V = L.reshape(K, -1).T.astype(np.float64)          # (n^2, K)
    norms = np.linalg.norm(V, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return V / norms

def _principal_angle_sines(L_disc: np.ndarray, L_gt: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Return sin(θ_i) for all principal angles between the two generator subspaces.
    Handles dimension mismatch: extra dimensions in either subspace appear as sin=1.
    """
    V  = _generators_to_matrix(L_disc)
    Vs = _generators_to_matrix(L_gt)
    Q  = _orth(V,  tol)
    Qs = _orth(Vs, tol)
    sv = np.linalg.svd(Q.T @ Qs, compute_uv=False)
    sv = np.clip(sv, 0.0, 1.0)
    r, r_gt = Q.shape[1], Qs.shape[1]
    n_angles = max(r, r_gt)
    cos2 = np.zeros(n_angles)
    cos2[:len(sv)] = sv ** 2
    return np.sqrt(np.maximum(0.0, 1.0 - cos2))  # sin(θ_i)


def compute_space_error(L_disc: np.ndarray, L_gt: np.ndarray, tol: float = 1e-8) -> float:
    """Subspace distance: sum of squared sines of principal angles.

    Invariant to generator ordering, redundant generators, different K, scaling.
    """
    sins = _principal_angle_sines(L_disc, L_gt, tol)
    return float(np.sum(sins ** 2))


def compute_sin_theta_max(L_disc: np.ndarray, L_gt: np.ndarray, tol: float = 1e-8) -> float:
    """Max principal angle (sin) between discovered and ground-truth subspaces."""
    sins = _principal_angle_sines(L_disc, L_gt, tol)
    return float(sins.max()) if len(sins) > 0 else 1.0


def compute_sin_theta_avg(L_disc: np.ndarray, L_gt: np.ndarray, tol: float = 1e-8) -> float:
    """Mean principal angle (sin) between discovered and ground-truth subspaces."""
    sins = _principal_angle_sines(L_disc, L_gt, tol)
    return float(sins.mean()) if len(sins) > 0 else 1.0


def compute_sin_theta_min_best(L_disc: np.ndarray, L_gt: np.ndarray, tol: float = 1e-8) -> float:
    """Min sin_theta over individual discovered generators vs GT subspace.

    For each single generator k, compute sin_theta_max(L_disc[k:k+1], L_gt) and
    return the minimum across all k.  Measures whether ANY single discovered
    generator aligns with the GT subspace, independent of extra spurious generators.
    """
    best = 1.0
    for k in range(L_disc.shape[0]):
        s = compute_sin_theta_max(L_disc[k:k+1], L_gt, tol)
        if s < best:
            best = s
    return best


def _orth(M: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, s > tol]


# def compute_space_error(L_disc: np.ndarray, L_gt: np.ndarray) -> float:
#     """LieSD space error between discovered generators L_disc and ground truth L_gt.

#     E_space = min_X ||V X - V*||_F^2 + min_Y ||V* Y - V||_F^2

#     Solved via least-squares (Moore-Penrose pseudoinverse).
#     Both inputs have shape (K, n, n).
#     """
#     V = _generators_to_matrix(L_disc)      # (n^2, K)
#     Vs = _generators_to_matrix(L_gt)       # (n^2, K_gt)
#     X, _, _, _ = np.linalg.lstsq(V, Vs, rcond=None)
#     Y, _, _, _ = np.linalg.lstsq(Vs, V, rcond=None)
#     err = np.linalg.norm(V @ X - Vs) ** 2 + np.linalg.norm(Vs @ Y - V) ** 2
#     return float(err)


def compute_orth_error(L_disc: np.ndarray) -> float:
    """LieSD orthogonality error: sum_{i<j} |<v_i, v_j>| over unit-norm columns."""
    V = _generators_to_matrix(L_disc)      # (n^2, K)
    K = V.shape[1]
    err = 0.0
    for i in range(K - 1):
        for j in range(i + 1, K):
            err += abs(float(V[:, i] @ V[:, j]))
    return err


# ---------------------------------------------------------------------------
# Save discovered symmetries (written by the discovery scripts; loading is
# the caller's responsibility, e.g. train_dynamics_model.py reads the json/
# npz directly for the equivariance-penalty check)
# ---------------------------------------------------------------------------

def save_discovered_symmetry(save_dir: str, step: int,
                             env_name: str, n_s: int, n_a: int,
                             L_s: np.ndarray, L_a: np.ndarray,
                             W: np.ndarray = None,
                             extra_meta: dict = None):
    """Save discovered generators + metadata for later EMLP use.

    W is optional — only saved when a linear intertwiner was actually learnt.
    Writes:
      <save_dir>/discovered_symmetry_step<step>.npz  – arrays
      <save_dir>/discovered_symmetry_step<step>.json – metadata
    """
    import json, os
    os.makedirs(save_dir, exist_ok=True)

    stem = os.path.join(save_dir, f"discovered_symmetry_step{step}")
    arrays = dict(L_s=L_s, L_a=L_a)
    has_intertwiner = W is not None
    if has_intertwiner:
        arrays["W"] = W
    np.savez(stem + ".npz", **arrays)

    meta = dict(env_name=env_name, n_s=int(n_s), n_a=int(n_a),
                K=int(L_s.shape[0]), step=int(step),
                has_intertwiner=has_intertwiner,
                npz_path=stem + ".npz")
    if extra_meta:
        meta.update(extra_meta)
    with open(stem + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    return stem


def save_discovered_discrete_symmetry(save_dir: str, subgroup: dict,
                                      env_name: str, n_s: int, n_a: int,
                                      extra_meta: dict = None) -> str:
    """Save discrete subgroup elements for later EMLP RL use.

    subgroup : dict with key 'elements', each elem has 'D_s' and 'D_a' arrays.
    Writes:
      <save_dir>/discovered_discrete_symmetry.npz  – stacked D_s / D_a arrays
      <save_dir>/discovered_discrete_symmetry.json – metadata
    Returns the stem path.
    """
    import json as _json, os as _os
    _os.makedirs(save_dir, exist_ok=True)

    elements = subgroup.get("elements", [])
    if not elements:
        return None

    Ds = np.stack([np.asarray(e["D_s"], dtype=np.float32) for e in elements])
    Da = np.stack([np.asarray(e["D_a"], dtype=np.float32) for e in elements])

    stem = _os.path.join(save_dir, "discovered_discrete_symmetry")
    np.savez(stem + ".npz", D_s=Ds, D_a=Da)

    meta = dict(
        env_name=env_name, n_s=int(n_s), n_a=int(n_a),
        num_elements=int(len(elements)),
        npz_path=stem + ".npz",
        kind="discrete",
    )
    if extra_meta:
        meta.update(extra_meta)
    with open(stem + ".json", "w") as f:
        _json.dump(meta, f, indent=2)

    return stem
