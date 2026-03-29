# REFERENCE PYTHON FILE FOR PLOTS AND ORGANIZATION OF OWN FILES
# REFERENCE PYTHON FILE FOR PLOTS AND ORGANIZATION OF OWN FILES
# REFERENCE PYTHON FILE FOR PLOTS AND ORGANIZATION OF OWN FILES
# REFERENCE PYTHON FILE FOR PLOTS AND ORGANIZATION OF OWN FILES
# REFERENCE PYTHON FILE FOR PLOTS AND ORGANIZATION OF OWN FILES

"""
Earth-Moon-Satellite-Asteroid N-body orbital mechanics.
Integrators: RK4 (4th-order) and Leapfrog/Velocity-Verlet (2nd-order, symplectic).
Entry point: engine = Solver(...) then engine.solve()
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import time

# ── Constants ─────────────────────────────────────────────────────────────────
G  = 6.67430e-11;  g0 = 9.80665
r_E = 6.371e6;  m_E = 5.972e24;  m_M = 7.348e22

const_G = G;  const_m_Earth = m_E;  const_m_Moon = m_M
const_r_Earth = r_E;  const_r_Moon = 3.844e8
const_r_sat_init = r_E + 400e3

def find_v_sqrt_GM_r(M, r): return np.sqrt(G * M / r)

# ── Physics ───────────────────────────────────────────────────────────────────
def compute_accelerations(masses, positions):
    N = len(masses); acc = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if i == j: continue
            dr = positions[j] - positions[i]
            r2 = dr @ dr + 1e-12
            acc[i] += G * masses[j] / (r2 * np.sqrt(r2)) * dr
    return acc

def total_energy(masses, pos, vel):
    N = len(masses)
    KE = 0.5 * np.sum(masses[:, None] * vel**2)
    PE = sum(-G * masses[i] * masses[j] / max(np.linalg.norm(pos[j] - pos[i]), 1.0)
             for i in range(N) for j in range(i + 1, N))
    return KE + PE

# ── Integrators (uniform interface) ──────────────────────────────────────────
# Both: (pos, vel, masses, dt, thrust_fn=None, t=0.0, sat_idx=None) → (pos_new, vel_new)
# thrust_fn(t, pos_sat, vel_sat) → Δa added to satellite only

def rk4_step(pos, vel, masses, dt, thrust_fn=None, t=0.0, sat_idx=None):
    def _a(p, v, t_):
        a = compute_accelerations(masses, p)
        if thrust_fn is not None and sat_idx is not None:
            a[sat_idx] += thrust_fn(t_, p[sat_idx], v[sat_idx])
        return a
    k1v, k1a = vel,            _a(pos,             vel,            t      )
    k2v, k2a = vel+.5*dt*k1a, _a(pos+.5*dt*k1v,  vel+.5*dt*k1a, t+.5*dt)
    k3v, k3a = vel+.5*dt*k2a, _a(pos+.5*dt*k2v,  vel+.5*dt*k2a, t+.5*dt)
    k4v, k4a = vel+dt*k3a,    _a(pos+dt*k3v,      vel+dt*k3a,    t+dt   )
    return (pos + dt/6*(k1v+2*k2v+2*k3v+k4v),
            vel + dt/6*(k1a+2*k2a+2*k3a+k4a))

def leapfrog_step(pos, vel, masses, dt, thrust_fn=None, t=0.0, sat_idx=None):
    """Velocity Verlet / Störmer–Verlet — symplectic, 2nd-order."""
    a0 = compute_accelerations(masses, pos)
    if thrust_fn is not None and sat_idx is not None:
        a0[sat_idx] += thrust_fn(t, pos[sat_idx], vel[sat_idx])
    vh = vel + .5*dt*a0
    pn = pos + dt*vh
    a1 = compute_accelerations(masses, pn)
    if thrust_fn is not None and sat_idx is not None:
        a1[sat_idx] += thrust_fn(t+dt, pn[sat_idx], vh[sat_idx])
    return pn, vh + .5*dt*a1

def propagate(pos, vel, masses, dt, T, method='rk4',
              thrust_fn=None, sat_idx=None, store_E=False):
    fn = rk4_step if method == 'rk4' else leapfrog_step
    p, v = pos.copy(), vel.copy()
    ptraj, vtraj, ts = [p.copy()], [v.copy()], [0.0]
    Es = [total_energy(masses, p, v)] if store_E else None
    t = 0.0
    while t < T - 1e-10:
        p, v = fn(p, v, masses, dt, thrust_fn, t, sat_idx)
        t += dt
        ptraj.append(p.copy()); vtraj.append(v.copy()); ts.append(t)
        if store_E: Es.append(total_energy(masses, p, v))
    out = (np.array(ptraj), np.array(vtraj), np.array(ts))
    return out + (np.array(Es),) if store_E else out

# ── Analytical 2-body circular-orbit reference ────────────────────────────────
def _analytical_orbit(r0, t_arr, mu=G*m_E):
    r = np.linalg.norm(r0)
    phi = np.arctan2(r0[1], r0[0]) + np.sqrt(mu/r**3) * t_arr
    return np.c_[r*np.cos(phi), r*np.sin(phi), np.zeros_like(phi)]

# ── Stumpff functions ─────────────────────────────────────────────────────────
def _stumpff(z):
    if z > 1e-6:
        s=np.sqrt(z); return (1-np.cos(s))/z, (s-np.sin(s))/z**1.5
    if z < -1e-6:
        s=np.sqrt(-z); return (np.cosh(s)-1)/(-z), (np.sinh(s)-s)/(-z)**1.5
    return 0.5, 1.0/6.0

# ── Lambert 2-body solver (universal variables, BME) ─────────────────────────
def lambert_2body(r1v, r2v, tof, mu=G*m_E):
    r1, r2 = np.linalg.norm(r1v), np.linalg.norm(r2v)
    cdv = np.clip(r1v @ r2v / (r1*r2), -1.0, 1.0)
    A   = np.sin(np.arccos(cdv)) * np.sqrt(r1*r2 / (1 - cdv + 1e-30))
    if np.cross(r1v, r2v)[2] < 0: A = -A
    if abs(A) < 1.0: raise ValueError("Lambert: degenerate geometry")

    def _y(z): C,S=_stumpff(z); return r1+r2+A*(z*S-1)/max(C**.5, 1e-15)
    def _F(z):
        yn=_y(z)
        if yn<=0: return 1e20
        C,S=_stumpff(z); return (yn/max(C,1e-15))**1.5*S+A*np.sqrt(yn)-np.sqrt(mu)*tof

    z = 0.0
    for _ in range(600):
        fv=_F(z); dz=max(abs(z)*1e-5, 1e-4)
        df=(_F(z+dz)-fv)/dz
        if abs(df)<1e-30: break
        z=np.clip(z - np.clip(fv/df, -2*np.pi, 2*np.pi), -4*np.pi**2+.1, 4*np.pi**2-.1)
        if abs(fv/df)<1e-9 and abs(fv)<1.0: break

    C,S=_stumpff(z); yn=_y(z)
    if yn<=0 or abs(C)<1e-15: raise ValueError("Lambert: no convergence")
    f=1-yn/r1; g=A*np.sqrt(yn/mu); gd=1-yn/r2
    if abs(g)<1e-15: raise ValueError("Lambert: degenerate g")
    return (r2v-f*r1v)/g, (gd*r2v-r1v)/g

# ── Thrust model (Tsiolkovsky finite burn) ────────────────────────────────────
def make_thrust_fn(dv_vec, m0, mdot, Isp):
    """Returns (thrust_fn, burn_duration_s). thrust_fn=None if burn not possible."""
    dv=np.linalg.norm(dv_vec)
    if dv<1.0 or mdot<=0 or Isp<=0: return None, 0.0
    mf  = m0 * np.exp(-dv/(Isp*g0))    # Tsiolkovsky final mass
    t_b = (m0 - mf) / mdot
    d   = dv_vec / dv;  F = mdot*Isp*g0
    def _fn(t, pos, vel):
        if t >= t_b: return np.zeros(3)
        return F / max(m0-mdot*t, mf) * d
    return _fn, t_b

# ── Intercept search (Lambert sweep over candidate times) ─────────────────────
def find_intercept(masses, pos0, vel0, sat_idx, asteroid_idx,
                   T_search=3e6, dt_probe=500.0):
    """Sweep candidate intercept times; pick min-ΔV 2-body Lambert arc."""
    mu = G*masses[0]    # central body = Earth (index 0)
    r_s, v_s = pos0[sat_idx], vel0[sat_idx]
    keep = [i for i in range(len(masses)) if i != sat_idx]
    ai   = keep.index(asteroid_idx)
    traj_k, _, ts_k = propagate(pos0[keep], vel0[keep], masses[keep],
                                dt_probe, T_search)
    best_dv, best_t, best_v1 = np.inf, None, None
    N=len(ts_k); step=max(1, N//100)
    for i in range(step, N, step):
        try:
            v1, _ = lambert_2body(r_s, traj_k[i, ai], ts_k[i], mu)
            dv    = np.linalg.norm(v1 - v_s)
            if dv < best_dv: best_dv, best_t, best_v1 = dv, ts_k[i], v1
        except Exception: continue
    if best_v1 is None:
        raise RuntimeError("Intercept search failed — no valid Lambert arc found.")
    print(f"  Optimal intercept: t = {best_t/3600:.2f} h,  ΔV = {best_dv:.1f} m/s")
    return best_v1 - v_s, best_t, best_dv

# ── Integrator comparison ─────────────────────────────────────────────────────
_CMP_DTS = [1000, 500, 250, 125, 50]

def _run_comparison(m2, p2, v2, T_cmp):
    """RK4 vs Leapfrog at each timestep; 2-body Earth+Satellite for analytical reference."""
    res = {}
    for mname in ('rk4', 'leapfrog'):
        res[mname] = {}
        for dt in _CMP_DTS:
            t0 = time.perf_counter()
            traj, _, ts, Es = propagate(p2, v2, m2, dt, T_cmp,
                                        method=mname, store_E=True)
            elapsed = time.perf_counter() - t0
            ts_arr = np.array(ts)
            anl    = _analytical_orbit(p2[1], ts_arr)   # exact circular-orbit positions
            errs   = np.linalg.norm(traj[:, 1, :] - anl, axis=1)
            e0     = Es[0]
            res[mname][dt] = {'wall_t': elapsed, 'errs': errs,
                              'drift': (Es-e0)/(abs(e0)+1e-50), 'ts': ts_arr}
    return res

def _plot_comparison(res, T_cmp):
    dts = _CMP_DTS
    clr = {'rk4': '#1565C0', 'leapfrog': '#BF360C'}
    lbl = {'rk4': 'RK4 (4th-order)', 'leapfrog': 'Leapfrog (2nd-order, symplectic)'}

    # ── Figure 1: summary ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Integration Comparison  (T = {T_cmp/86400:.0f} days, 2-body Earth–Satellite)',
                 fontweight='bold')

    ax = axes[0]
    x = np.arange(len(dts)); w = 0.35
    for k, m in enumerate(('rk4', 'leapfrog')):
        ax.bar(x+(k-.5)*w, [res[m][d]['wall_t'] for d in dts], w,
               label=lbl[m], color=clr[m], alpha=0.85)
    ax.set(xticks=x, xticklabels=dts, xlabel='Timestep (s)', ylabel='Wall time (s)',
           title='Computation Time')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    for m in ('rk4', 'leapfrog'):
        ax.loglog(dts, [res[m][d]['errs'][-1]/1e3 for d in dts],
                  'o-', label=lbl[m], color=clr[m])
    dt_arr = np.array(dts, float)
    for m, slope, scale in (('rk4',4,1e-9), ('leapfrog',2,0.01)):
        ax.loglog(dt_arr, scale*(dt_arr/dts[-1])**slope, '--',
                  color=clr[m], alpha=0.4, lw=1, label=f'O(dt^{slope})')
    ax.set(xlabel='Timestep (s)', ylabel='Final position error (km)',
           title='Accuracy vs. Analytical Orbit (log–log)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which='both')

    ax = axes[2]
    for m in ('rk4', 'leapfrog'):
        ax.semilogy(dts, [np.max(np.abs(res[m][d]['drift'])) for d in dts],
                    's-', label=lbl[m], color=clr[m])
    ax.set(xlabel='Timestep (s)', ylabel='max |ΔE/E₀|', title='Peak Energy Drift')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_summary.png', dpi=150, bbox_inches='tight'); plt.show()

    # ── Figure 2: position error over time ────────────────────────────────────
    fig, axes = plt.subplots(2, len(dts), figsize=(18, 7), sharey='row',
                              constrained_layout=True)
    fig.suptitle('Position Error vs. Analytical Circular Orbit', fontweight='bold')
    for col, dt in enumerate(dts):
        for row, m in enumerate(('rk4', 'leapfrog')):
            ax = axes[row][col]
            ax.plot(res[m][dt]['ts']/3600, res[m][dt]['errs']/1e3,
                    lw=0.8, color=clr[m])
            if row == 0: ax.set_title(f'dt = {dt} s', fontsize=9)
            if col == 0: ax.set_ylabel(f"{lbl[m][:3]}\nerror (km)", fontsize=8)
            ax.grid(alpha=0.3)
    fig.supxlabel('Time (h)')
    plt.savefig('comparison_accuracy.png', dpi=150, bbox_inches='tight'); plt.show()

    # ── Figure 3: energy drift over time ──────────────────────────────────────
    fig, axes = plt.subplots(2, len(dts), figsize=(18, 7), sharey='row',
                              constrained_layout=True)
    fig.suptitle('Energy Drift ΔE/E₀ Over Time  '
                 '(Leapfrog is symplectic → bounded oscillation; RK4 may drift)',
                 fontweight='bold')
    for col, dt in enumerate(dts):
        for row, m in enumerate(('rk4', 'leapfrog')):
            ax = axes[row][col]
            ax.plot(res[m][dt]['ts']/3600, res[m][dt]['drift'],
                    lw=0.8, color=clr[m])
            if row == 0: ax.set_title(f'dt = {dt} s', fontsize=9)
            if col == 0: ax.set_ylabel(f"{lbl[m][:3]}\nΔE/E₀", fontsize=8)
            ax.grid(alpha=0.3)
    fig.supxlabel('Time (h)')
    plt.savefig('comparison_energy.png', dpi=150, bbox_inches='tight'); plt.show()

# ── 3D Trajectory Plot ────────────────────────────────────────────────────────
def _plot_3d(traj, names, dt, t_burn=0.0):
    sc  = 1e6   # m → Mm
    clr = {'Earth':'royalblue', 'Moon':'slategrey',
           'Satellite':'crimson', 'Asteroid':'#2E7D32'}

    fig = plt.figure(figsize=(13, 11))
    ax  = fig.add_subplot(111, projection='3d')

    for i, name in enumerate(names):
        tr = traj[:, i, :] / sc; c = clr.get(name, 'purple')
        ax.plot(tr[:,0], tr[:,1], tr[:,2], color=c, lw=1.2, alpha=0.85, label=name)
        ax.scatter(*tr[-1], color=c, s=50, zorder=5)
        ax.scatter(*tr[0],  color=c, s=20, marker='^', zorder=5)

    # burnout marker
    if 'Satellite' in names and t_burn > 0:
        bi = min(int(t_burn/dt), len(traj)-1)
        bp = traj[bi, names.index('Satellite')] / sc
        ax.scatter(*bp, color='orange', s=120, marker='*', zorder=6)

    # Earth solid sphere + 200 km exclusion shell
    u, v = np.mgrid[0:2*np.pi:24j, 0:np.pi:13j]
    cu, sv, cv = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
    ax.plot_surface(r_E/sc*cu, r_E/sc*sv, r_E/sc*cv,
                    color='royalblue', alpha=0.5, linewidth=0)
    r_ex = (r_E+200e3)/sc
    ax.plot_surface(r_ex*cu, r_ex*sv, r_ex*cv,
                    color='cyan', alpha=0.08, linewidth=0)

    # Moon sphere at final position
    if 'Moon' in names:
        mp = traj[-1, names.index('Moon')] / sc;  rm = 1.737e6/sc
        ax.plot_surface(mp[0]+rm*cu, mp[1]+rm*sv, mp[2]+rm*cv,
                        color='slategrey', alpha=0.55, linewidth=0)

    ax.set_xlabel('X (Mm)'); ax.set_ylabel('Y (Mm)'); ax.set_zlabel('Z (Mm)')
    ax.set_title('Earth–Moon–Satellite–Asteroid Trajectory', fontsize=13, fontweight='bold')

    leg  = [mlines.Line2D([],[],color=clr.get(n,'purple'),lw=2,label=n) for n in names]
    leg += [mpatches.Patch(facecolor='cyan',alpha=0.4,label='200 km excl. zone')]
    if t_burn>0: leg.append(mlines.Line2D([],[],marker='*',color='orange',lw=0,ms=10,label='Burnout'))
    ax.legend(handles=leg, loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig('trajectory_3d.png', dpi=150, bbox_inches='tight'); plt.show()

# ── Classes ────────────────────────────────────────────────────────────────────
class body:
    def __init__(self, m, keepout_r, pos, vel):
        self.m, self.keepout_r = m, keepout_r
        self.pos, self.vel = np.array(pos, float), np.array(vel, float)

class Satellite:
    def __init__(self, m, keepout_r, pos, vel, mdot, Isp):
        self.m, self.keepout_r = m, keepout_r
        self.pos, self.vel = np.array(pos, float), np.array(vel, float)
        self.mdot, self.Isp = mdot, Isp

class Solver:
    _T_CMP  = 30*86400   # comparison window: 30 days
    _T_SRCH = 3e6        # intercept search horizon (s)
    _DT_PB  = 500.0      # probe dt for asteroid trajectory in search

    def __init__(self, bodies_input, satellite_input, dt):
        """
        bodies_input : list of [name, mass, keepout_r, pos, vel, accel(ignored)]
        satellite_input : [mass, keepout_r, [lat_deg, lon_deg, alt_km], v0, a0(ignored), mdot, Isp]
        dt           : main integration timestep (s)
        """
        self.dt = float(dt)
        self._bodies = {}

        for entry in bodies_input:
            name, m, r, pos, vel, _ = entry
            self._bodies[name] = body(m, r, pos, vel)

        m_s,r_s,lla,v0,_,mdot,Isp = satellite_input
        lat, lon = np.deg2rad(lla[0]), np.deg2rad(lla[1]);  alt = lla[2]*1e3
        rm    = r_E + alt
        pos_s = np.array([rm*np.cos(lat)*np.cos(lon), rm*np.sin(lat),
                           rm*np.cos(lat)*np.sin(lon)])
        v0a   = np.array(v0, float)
        if np.allclose(v0a, 0.0):    # circular LEO velocity if not specified
            tang  = np.array([-pos_s[1], pos_s[0], 0.0])
            tang /= np.linalg.norm(tang) + 1e-30
            v0a   = find_v_sqrt_GM_r(m_E, rm) * tang
        self._sat = Satellite(m_s, r_s, pos_s, v0a, mdot, Isp)

        # Table 1 defaults when caller passes zero vectors
        _def = {'Moon':    (np.array([3.844e8,0.,0.]), np.array([0.,1.02e3,0.])),
                'Asteroid':(np.array([5e8,5e8,0.]),    np.array([-500.,-200.,0.]))}
        for nm, (dp, dv) in _def.items():
            if nm in self._bodies:
                b = self._bodies[nm]
                if np.allclose(b.pos, 0.0): b.pos = dp.copy()
                if np.allclose(b.vel, 0.0): b.vel = dv.copy()

    def _pack(self, include_sat=True):
        names = list(self._bodies.keys())
        ms = np.array([self._bodies[n].m   for n in names])
        ps = np.vstack([self._bodies[n].pos for n in names])
        vs = np.vstack([self._bodies[n].vel for n in names])
        if include_sat:
            names.append('Satellite')
            ms = np.append(ms, self._sat.m)
            ps = np.vstack([ps, self._sat.pos])
            vs = np.vstack([vs, self._sat.vel])
        return names, ms, ps, vs

    def solve(self, method='rk4', run_comparison=True, run_3d=True):
        """
        Find optimal intercept, propagate full trajectory, generate all plots.
        Returns dict with trajectory data and intercept parameters.
        """
        names, masses, pos0, vel0 = self._pack(include_sat=True)
        sat_idx      = names.index('Satellite')
        asteroid_idx = names.index('Asteroid')

        print("Searching for optimal intercept trajectory...")
        dv_vec, t_int, dv_mag = find_intercept(
            masses, pos0, vel0, sat_idx, asteroid_idx,
            T_search=self._T_SRCH, dt_probe=self._DT_PB)

        thrust_fn, t_burn = make_thrust_fn(
            dv_vec, self._sat.m, self._sat.mdot, self._sat.Isp)

        T_total = t_int * 1.05
        print(f"Propagating  T = {T_total/3600:.1f} h,  dt = {self.dt} s,  method = {method}...")
        traj, vtraj, ts = propagate(
            pos0, vel0, masses, self.dt, T_total,
            method=method, thrust_fn=thrust_fn, sat_idx=sat_idx)

        if run_3d:
            print("Generating 3D trajectory plot...")
            _plot_3d(traj, names, self.dt, t_burn)

        if run_comparison:
            print("Running RK4 vs Leapfrog comparison (2-body Earth–Satellite)...")
            m2 = np.array([masses[0], masses[sat_idx]])
            p2 = np.vstack([pos0[0],  pos0[sat_idx]])
            v2 = np.vstack([vel0[0],  vel0[sat_idx]])
            cmp_res = _run_comparison(m2, p2, v2, self._T_CMP)
            _plot_comparison(cmp_res, self._T_CMP)

        print("Solve complete.")
        return {'traj': traj, 'vel': vtraj, 'times': ts, 'names': names,
                'dv_vec': dv_vec, 't_intercept': t_int,
                'dv_mag': dv_mag,  't_burn': t_burn}