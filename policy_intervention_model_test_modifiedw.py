import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from solvemodel_modifiedw import mu, steadyState

print("=== TWO-TYPE MODEL WITH POLICY INTERVENTION ===")
print("Policy: 50% reduction in search costs during period 5 only")
print("Implementation: Modify k parameter temporarily for both types")
print()

# ============================================================================
# MODIFIED SOLVER FUNCTIONS TO HANDLE TIME-VARYING SEARCH COSTS
# ============================================================================

def predictedMomentsPolicy(xi, b, s, logphi, k_vec=None):
    """Modified version that can handle time-varying k parameters"""
    delta, k, gamma, mu_S, sigma, kappa, pi = xi
    lastperiod = len(b)
    muv = mu(xi, np.arange(1, lastperiod + 1))
    haz = np.zeros(len(b))
    logw_reemp = np.zeros(len(b))

    for t in range(lastperiod - 2, -1, -1):
        omega = (logphi[t + 1] - muv[t]) / sigma
        if omega >= 7:
            omega = 7
        haz[t] = s[t] * (1 - norm.cdf(omega))
        logw_reemp[t] = muv[t] + sigma * norm.pdf(omega) / (1 - norm.cdf(omega))

    haz[-1] = haz[-2]
    logw_reemp[-1] = logw_reemp[-2]

    return haz, logw_reemp

def optimalPathPolicy(xi, b, k_vec=None):
    """Modified version that accepts time-varying search cost vector k_vec"""
    delta, k_base, gamma, mu_S, sigma, kappa, pi = xi
    lastperiod = len(b)
    muv = mu(xi, np.arange(1, lastperiod + 1))
    s = np.zeros(lastperiod)
    logphi = np.zeros(lastperiod)
    
    # Use time-varying k if provided, otherwise use base k
    if k_vec is None:
        k_vec = np.ones(lastperiod) * k_base
    
    # Assuming steadyState is another function to compute steady state
    s[-1], logphi[-1] = steadyState(xi, b[-1])

    for t in range(lastperiod - 2, -1, -1):
        omega = (logphi[t + 1] - muv[t + 1]) / sigma
        if omega > 7:
            omega = 7
        integral = (1 - norm.cdf(omega)) * (
            muv[t + 1] - logphi[t + 1] + sigma * norm.pdf(omega) / (1 - norm.cdf(omega))
        )
        if np.isnan(integral) or integral < 0:
            integral = 0
        
        # Use time-specific k value
        k_t = k_vec[t]
        s[t] = min((1 / k_t * delta / (1 - delta) * integral) ** (1 / gamma), 1)
        logphi[t] = (
            (1 - delta) * (np.log(b[t]) - k_t * (s[t] ** (1 + gamma)) / (1 + gamma))
            + delta * logphi[t + 1]
            + delta * s[t] * integral
        )

    return s, logphi

def solveModelPolicy(xi, institutions, k_vec=None):
    """Modified solver that handles time-varying search costs"""
    T, b = institutions

    s, logphi = optimalPathPolicy(xi, b, k_vec)
    haz, logw_reemp = predictedMomentsPolicy(xi, b, s, logphi, k_vec)
    surv = np.ones(len(b))
    for t in range(1, len(b)):
        surv[t] = surv[t - 1] * (1 - haz[t - 1])
    
    Tminb = T - len(b)
    haz_long = np.concatenate((haz, np.ones(Tminb) * haz[-1]))
    logw_reemp_long = np.concatenate((logw_reemp, np.ones(Tminb) * logw_reemp[-1]))

    surv_long = np.ones(T)
    for t in range(1, T):
        surv_long[t] = surv_long[t - 1] * (1 - haz_long[t - 1])

    D = np.sum(surv_long)
    dens_long = haz_long * surv_long
    if np.sum(dens_long) != 0:
        E_logw_reemp = np.sum(dens_long * logw_reemp_long) / np.sum(dens_long)
    else:
        E_logw_reemp = 0

    return s, logphi, haz, logw_reemp, surv, D, E_logw_reemp

def solveSingleTypeModelPolicy(xi, institutions, k_vec=None):
    """Policy version of single type model solver"""
    s, logphi, haz, logw_reemp, surv, D, E_logw_reemp = solveModelPolicy(xi, institutions, k_vec)
    return s, logphi, haz, logw_reemp, surv, D, E_logw_reemp

# ============================================================================
# MODEL SETUP AND POLICY DEFINITION
# ============================================================================

# Model setup
T = 96  # Long horizon to eliminate finite horizon effects
display_periods = 18  # Show only first 24 months in plots
ben_level = 800 / 30  # Constant benefit level
ben = np.ones(T) * ben_level
inst = (T, ben)

# Base parameters
base_params = [
    0.9493,     # delta
    47.0856,    # k1 (will be updated for each type)
    1,          # gamma  
    4.0364,     # mu1 (will be updated for each type)
    0.01,       # sigma
    25,         # kappa (experience effect starts after 0 periods)
    0.05,       # pi (declining wage offers)
]

# Type-specific parameters
type1_params = base_params.copy()
type1_params[1] = 47.0856   # k1
type1_params[3] = 4.0364   # mu1

type2_params = base_params.copy()
type2_params[1] = 148.35   # k2
type2_params[3] = 4.0364    # mu2  

# Type shares
q1, q2 = 0.5, 0.5

# Policy intervention: 50% reduction in search costs for period 5 (index 4)
intervention_period = 4  # Period 5 (0-indexed as 4)
cost_reduction = 0.35

# Create time-varying cost vectors
k1_vec = np.ones(T) * type1_params[1]  # Base k1 for all periods
k1_vec[intervention_period] = type1_params[1] * (1 - cost_reduction)  # Reduce by 35% in period 5

k2_vec = np.ones(T) * type2_params[1]  # Base k2 for all periods  
k2_vec[intervention_period] = type2_params[1] * (1 - cost_reduction)  # Reduce by 35% in period 5

print(f"Policy intervention details:")
print(f"- Target period: {intervention_period + 1} (period 5)")
print(f"- Cost reduction: {cost_reduction*100}%")
print(f"- Type 1 k: {type1_params[1]:.4f} → {k1_vec[intervention_period]:.4f}")
print(f"- Type 2 k: {type2_params[1]:.4f} → {k2_vec[intervention_period]:.4f}")
print()

# ============================================================================
# SOLVE BASELINE MODEL (NO INTERVENTION)
# ============================================================================

print("=== SOLVING BASELINE MODEL (NO INTERVENTION) ===")
print("Type 1: k={:.4f}, μ={:.4f}, share={:.1f}".format(type1_params[1], type1_params[3], q1))
print("Type 2: k={:.4f}, μ={:.4f}, share={:.1f}".format(type2_params[1], type2_params[3], q2))

# Solve baseline model
s1_base, logphi1_base, haz1_base, logw1_base, surv1_base, D1_base, Ew1_base = solveSingleTypeModelPolicy(type1_params, inst)
s2_base, logphi2_base, haz2_base, logw2_base, surv2_base, D2_base, Ew2_base = solveSingleTypeModelPolicy(type2_params, inst)

# ============================================================================
# SOLVE POLICY MODEL (WITH INTERVENTION)
# ============================================================================

print("\n=== SOLVING POLICY MODEL (WITH INTERVENTION) ===")

# Solve policy model with time-varying costs
s1_policy, logphi1_policy, haz1_policy, logw1_policy, surv1_policy, D1_policy, Ew1_policy = solveSingleTypeModelPolicy(type1_params, inst, k1_vec)
s2_policy, logphi2_policy, haz2_policy, logw2_policy, surv2_policy, D2_policy, Ew2_policy = solveSingleTypeModelPolicy(type2_params, inst, k2_vec)

# ============================================================================
# CALCULATE AGGREGATES FOR BOTH SCENARIOS
# ============================================================================

def calculate_aggregates(s1, s2, logphi1, logphi2, haz1, haz2, logw1, logw2, surv1, surv2, q1, q2, T):
    """Calculate aggregate measures given individual type results"""
    # Aggregate survival
    surv_agg = q1 * surv1 + q2 * surv2
    
    # Type composition over time
    comp1 = np.zeros(T)
    comp2 = np.zeros(T)
    
    for t in range(T):
        if surv_agg[t] > 0:
            comp1[t] = (q1 * surv1[t]) / surv_agg[t]
            comp2[t] = (q2 * surv2[t]) / surv_agg[t]
        else:
            comp1[t] = comp1[t-1] if t > 0 else q1
            comp2[t] = comp2[t-1] if t > 0 else q2
    
    # Aggregate measures (composition-weighted)
    haz_agg = comp1 * haz1 + comp2 * haz2
    s_agg = comp1 * s1 + comp2 * s2
    logphi_agg = comp1 * logphi1 + comp2 * logphi2
    
    # Aggregate reemployment wages (hazard-weighted)
    logw_agg = np.zeros(T)
    for t in range(T):
        if haz_agg[t] > 0:
            logw_agg[t] = (comp1[t] * haz1[t] * logw1[t] + comp2[t] * haz2[t] * logw2[t]) / haz_agg[t]
        else:
            logw_agg[t] = comp1[t] * logw1[t] + comp2[t] * logw2[t]
    
    return surv_agg, comp1, comp2, haz_agg, s_agg, logphi_agg, logw_agg

# Calculate aggregates for baseline
surv_agg_base, comp1_base, comp2_base, haz_agg_base, s_agg_base, logphi_agg_base, logw_agg_base = calculate_aggregates(
    s1_base, s2_base, logphi1_base, logphi2_base, haz1_base, haz2_base, 
    logw1_base, logw2_base, surv1_base, surv2_base, q1, q2, T)

# Calculate aggregates for policy
surv_agg_policy, comp1_policy, comp2_policy, haz_agg_policy, s_agg_policy, logphi_agg_policy, logw_agg_policy = calculate_aggregates(
    s1_policy, s2_policy, logphi1_policy, logphi2_policy, haz1_policy, haz2_policy,
    logw1_policy, logw2_policy, surv1_policy, surv2_policy, q1, q2, T)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n=== POLICY IMPACT SUMMARY ===")
print("Expected Duration Effects:")
print(f"Type 1: {D1_base:.2f} → {D1_policy:.2f} (change: {D1_policy - D1_base:.2f})")
print(f"Type 2: {D2_base:.2f} → {D2_policy:.2f} (change: {D2_policy - D2_base:.2f})")
print(f"Aggregate: {q1*D1_base + q2*D2_base:.2f} → {q1*D1_policy + q2*D2_policy:.2f} (change: {q1*(D1_policy-D1_base) + q2*(D2_policy-D2_base):.2f})")
print()

print("Average Reemployment Wage Effects:")
print(f"Type 1: {Ew1_base:.4f} → {Ew1_policy:.4f} (change: {Ew1_policy - Ew1_base:.4f})")
print(f"Type 2: {Ew2_base:.4f} → {Ew2_policy:.4f} (change: {Ew2_policy - Ew2_base:.4f})")
print(f"Aggregate: {q1*Ew1_base + q2*Ew2_base:.4f} → {q1*Ew1_policy + q2*Ew2_policy:.4f} (change: {q1*(Ew1_policy-Ew1_base) + q2*(Ew2_policy-Ew2_base):.4f})")
print()

print("Period 5 Effects (intervention period):")
print(f"Search effort Type 1: {s1_base[4]:.4f} → {s1_policy[4]:.4f} (change: {s1_policy[4] - s1_base[4]:.4f})")
print(f"Search effort Type 2: {s2_base[4]:.4f} → {s2_policy[4]:.4f} (change: {s2_policy[4] - s2_base[4]:.4f})")
print(f"Exit hazard Type 1: {haz1_base[4]:.4f} → {haz1_policy[4]:.4f} (change: {haz1_policy[4] - haz1_base[4]:.4f})")
print(f"Exit hazard Type 2: {haz2_base[4]:.4f} → {haz2_policy[4]:.4f} (change: {haz2_policy[4] - haz2_base[4]:.4f})")

# ============================================================================
# CREATE MODIFIED PLOTS AS REQUESTED
# ============================================================================

# Calculate cumulative search intensity
s1_base_cum = np.cumsum(s1_base[:display_periods])
s1_policy_cum = np.cumsum(s1_policy[:display_periods])
s2_base_cum = np.cumsum(s2_base[:display_periods])
s2_policy_cum = np.cumsum(s2_policy[:display_periods])

# Calculate means for the first 18 periods
s1_base_mean = np.mean(s1_base[:display_periods])
s1_policy_mean = np.mean(s1_policy[:display_periods])
s2_base_mean = np.mean(s2_base[:display_periods])
s2_policy_mean = np.mean(s2_policy[:display_periods])

# Define colors and styles
colors = {'type1': 'green', 'type2': 'blue'}
line_styles = {'baseline': '-', 'policy': '--'}
line_widths = {'individual': 2.5}

months = np.arange(1, display_periods + 1)

# ============================================================================
# FIGURE 1: Four Panel Plot
# ============================================================================

fig1, axes = plt.subplots(2, 2, figsize=(16, 16))

# Panel 1: Search Intensity Levels
ax1 = axes[0, 0]
ax1.plot(months, s1_base[:display_periods], color=colors['type1'], linestyle=line_styles['baseline'], 
         linewidth=line_widths['individual'], label=f'Type 1 Baseline (mean={s1_base_mean:.3f})')
ax1.plot(months, s1_policy[:display_periods], color=colors['type1'], linestyle=line_styles['policy'], 
         linewidth=line_widths['individual'], label=f'Type 1 Policy (mean={s1_policy_mean:.3f})')
ax1.plot(months, s2_base[:display_periods], color=colors['type2'], linestyle=line_styles['baseline'], 
         linewidth=line_widths['individual'], label=f'Type 2 Baseline (mean={s2_base_mean:.3f})')
ax1.plot(months, s2_policy[:display_periods], color=colors['type2'], linestyle=line_styles['policy'], 
         linewidth=line_widths['individual'], label=f'Type 2 Policy (mean={s2_policy_mean:.3f})')
ax1.axvline(x=5, color='gray', linestyle=':', alpha=0.7)
ax1.set_title('Panel 1: Search Intensity Levels', fontsize=15, fontweight='bold')

ax1.set_ylabel('Search Intensity')
ax1.set_xticks(range(1, display_periods + 1, 3))  # Ticks at 1, 4, 7, 10, 13, 16
ax1.legend(fontsize=14)
ax1.grid(True, alpha=0.3)

# Panel 2: Cumulative Search Intensity
ax2 = axes[0, 1]
ax2.plot(months, s1_base_cum, color=colors['type1'], linestyle=line_styles['baseline'], 
         linewidth=line_widths['individual'], label='Type 1 Baseline')
ax2.plot(months, s1_policy_cum, color=colors['type1'], linestyle=line_styles['policy'], 
         linewidth=line_widths['individual'], label='Type 1 Policy')
ax2.plot(months, s2_base_cum, color=colors['type2'], linestyle=line_styles['baseline'], 
         linewidth=line_widths['individual'], label='Type 2 Baseline')
ax2.plot(months, s2_policy_cum, color=colors['type2'], linestyle=line_styles['policy'], 
         linewidth=line_widths['individual'], label='Type 2 Policy')
ax2.axvline(x=5, color='gray', linestyle=':', alpha=0.7)
ax2.set_title('Panel 2: Cumulative Search Intensity', fontsize=15, fontweight='bold')

ax2.set_ylabel('Cumulative Search Intensity')
ax2.set_xticks(range(1, display_periods + 1, 3))  # Ticks at 1, 4, 7, 10, 13, 16
ax2.legend(fontsize=14)
ax2.grid(True, alpha=0.3)

# Panel 3: Composition of Types
ax3 = axes[1, 0]
ax3.plot(months, comp1_base[:display_periods], color=colors['type1'], linestyle=line_styles['baseline'], 
         linewidth=line_widths['individual'])
ax3.plot(months, comp1_policy[:display_periods], color=colors['type1'], linestyle=line_styles['policy'], 
         linewidth=line_widths['individual'])
ax3.plot(months, comp2_base[:display_periods], color=colors['type2'], linestyle=line_styles['baseline'], 
         linewidth=line_widths['individual'])
ax3.plot(months, comp2_policy[:display_periods], color=colors['type2'], linestyle=line_styles['policy'], 
         linewidth=line_widths['individual'])
ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax3.axvline(x=5, color='gray', linestyle=':', alpha=0.7)
ax3.set_title('Panel 3: Composition of Types', fontsize=15, fontweight='bold')
ax3.set_xlabel('Months of Unemployment')
ax3.set_ylabel('Share of Each Type')
ax3.set_ylim(0, 1)
ax3.set_xticks(range(1, display_periods + 1, 3))  # Ticks at 1, 4, 7, 10, 13, 16
ax3.grid(True, alpha=0.3)

# Panel 4: Reservation Wages
ax4 = axes[1, 1]
ax4.plot(months, logphi1_base[:display_periods], color=colors['type1'], linestyle=line_styles['baseline'], 
         linewidth=line_widths['individual'])
ax4.plot(months, logphi1_policy[:display_periods], color=colors['type1'], linestyle=line_styles['policy'], 
         linewidth=line_widths['individual'])
ax4.plot(months, logphi2_base[:display_periods], color=colors['type2'], linestyle=line_styles['baseline'], 
         linewidth=line_widths['individual'])
ax4.plot(months, logphi2_policy[:display_periods], color=colors['type2'], linestyle=line_styles['policy'], 
         linewidth=line_widths['individual'])
ax4.axvline(x=5, color='gray', linestyle=':', alpha=0.7)
ax4.set_title('Panel 4: Reservation Wages', fontsize=15, fontweight='bold')
ax4.set_xlabel('Months of Unemployment')
ax4.set_ylabel('Log Reservation Wage')
ax4.set_xticks(range(1, display_periods + 1, 3))  # Ticks at 1, 4, 7, 10, 13, 16
ax4.grid(True, alpha=0.3)


plt.tight_layout()

plt.savefig(r'C:\Users\Thijs\Dropbox\Thijs Scheepmaker\test\output\HOLEmodel_policy_intervention_mumain2.png', 
            dpi=300, bbox_inches='tight')

# ============================================================================
# FIGURE 2: Exit Hazard and Survival Rates
# ============================================================================

fig2, axes = plt.subplots(1, 2, figsize=(16, 8))

# Panel 1: Exit Hazard Rates
ax1 = axes[0]
ax1.plot(months, haz1_base[:display_periods], color=colors['type1'], linestyle=line_styles['baseline'], 
         linewidth=line_widths['individual'], label='Type 1 Baseline')
ax1.plot(months, haz1_policy[:display_periods], color=colors['type1'], linestyle=line_styles['policy'], 
         linewidth=line_widths['individual'], label='Type 1 Policy')
ax1.plot(months, haz2_base[:display_periods], color=colors['type2'], linestyle=line_styles['baseline'], 
         linewidth=line_widths['individual'], label='Type 2 Baseline')
ax1.plot(months, haz2_policy[:display_periods], color=colors['type2'], linestyle=line_styles['policy'], 
         linewidth=line_widths['individual'], label='Type 2 Policy')
ax1.axvline(x=5, color='gray', linestyle=':', alpha=0.7, label='Intervention Period')
ax1.set_title('Panel 1: Exit Hazard Rates', fontsize=12, fontweight='bold')
ax1.set_xlabel('Months of Unemployment')
ax1.set_ylabel('Monthly Exit Probability')
ax1.set_xticks(range(1, display_periods + 1, 3))  # Ticks at 1, 4, 7, 10, 13, 16
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: Survival Rates
ax2 = axes[1]
ax2.plot(months, surv1_base[:display_periods], color=colors['type1'], linestyle=line_styles['baseline'], 
         linewidth=line_widths['individual'], label='Type 1 Baseline')
ax2.plot(months, surv1_policy[:display_periods], color=colors['type1'], linestyle=line_styles['policy'], 
         linewidth=line_widths['individual'], label='Type 1 Policy')
ax2.plot(months, surv2_base[:display_periods], color=colors['type2'], linestyle=line_styles['baseline'], 
         linewidth=line_widths['individual'], label='Type 2 Baseline')
ax2.plot(months, surv2_policy[:display_periods], color=colors['type2'], linestyle=line_styles['policy'], 
         linewidth=line_widths['individual'], label='Type 2 Policy')
ax2.axvline(x=5, color='gray', linestyle=':', alpha=0.7, label='Intervention Period')
ax2.set_title('Panel 2: Survival Rates', fontsize=12, fontweight='bold')
ax2.set_xlabel('Months of Unemployment')
ax2.set_ylabel('Survival Probability')
ax2.set_xticks(range(1, display_periods + 1, 3))  # Ticks at 1, 4, 7, 10, 13, 16
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)


plt.tight_layout()

plt.savefig(r'C:\Users\Thijs\Dropbox\Thijs Scheepmaker\test\output\HOLEmodel_policy_intervention_muappendix2.png', 
            dpi=300, bbox_inches='tight')

print("\n=== MODIFIED FIGURES CREATED ===")
print("Figure 1 contains four panels:")
print("  - Panel 1: Search intensity levels with means in legend")
print("  - Panel 2: Cumulative search intensity over unemployment duration")
print("  - Panel 3: Composition of types over time")
print("  - Panel 4: Reservation wages by type and model")
print()
print("Figure 2 contains two panels:")
print("  - Panel 1: Exit hazard rates by type and model")
print("  - Panel 2: Survival rates by type and model")
print()
print("All plots maintain line styling by type (green=Type 1, blue=Type 2)")
print("and by model (solid=Baseline, dashed=Policy)")

