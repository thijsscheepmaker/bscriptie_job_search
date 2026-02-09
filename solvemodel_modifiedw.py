import sys
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import optimagic as em
from numba import njit


momentsfile = "base_moments_Germany_wages.xlsx"

def mu(xi, t):
    """
    Modified wage offer distribution function.
    Formula: ln_t(w) ~ N(μ + π * max{S-t, 0}, σ²)
    
    The wage offer starts at μ + π*S in period 1 and gradually 
    declines by π each period until reaching μ at period S (steady state).
    After period S, it remains at μ.
    
    Parameters:
    - xi: parameter vector [delta, k, gamma, mu_S, sigma, kappa, pi]
    - t: time period(s) (can be array or scalar)
    - kappa: now interpreted as S (steady state period)
    
    Note: pi should be positive for declining wage offers
    """
    delta, k, gamma, mu_S, sigma, kappa, pi = xi
    S = kappa  # kappa now represents the steady state period S
    
    # Calculate max{S-t, 0} for each time period
    # This starts at S-1 when t=1, and reaches 0 when t=S
    decline_term = np.maximum(S - t, np.zeros(len(t)))
    
    return mu_S + pi * decline_term


def predictedMoments(xi, b, s, logphi):
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
        logw_reemp[t] = muv[t] + sigma * norm.pdf(omega) / (
            1 - norm.cdf(omega)
        )

    haz[-1] = haz[-2]
    logw_reemp[-1] = logw_reemp[-2]

    return haz, logw_reemp


def optimalPath(xi, b):
    delta, k, gamma, mu_S, sigma, kappa, pi = xi
    lastperiod = len(b)
    muv = mu(xi, np.arange(1, lastperiod + 1))
    s = np.zeros(lastperiod)
    logphi = np.zeros(lastperiod)

    # Assuming steadyState is another function to compute steady state
    s[-1], logphi[-1] = steadyState(xi, b[-1])

    for t in range(lastperiod - 2, -1, -1):
        omega = (logphi[t + 1] - muv[t + 1]) / sigma
        if omega > 7:
            omega = 7
        integral = (1 - norm.cdf(omega)) * (
            muv[t + 1]
            - logphi[t + 1]
            + sigma
            * norm.pdf(omega)
            / (1 - norm.cdf(omega))
        )
        if np.isnan(integral) or integral < 0:
            integral = 0
        s[t] = min((1 / k * delta / (1 - delta) * integral) ** (1 / gamma), 1)
        logphi[t] = (
            (1 - delta) * (np.log(b[t]) - k * (s[t] ** (1 + gamma)) / (1 + gamma))
            + delta * logphi[t + 1]
            + delta * s[t] * integral
        )

    return s, logphi


# @njit(cache=True)
def clip_gradient(grad, max_grad=1):
    norm = np.linalg.norm(grad)
    if norm > max_grad:
        grad = grad / norm * max_grad
    return grad

# @njit(cache=True)
def check_and_update_bound(func, x, bounds):
    for j, (lower, upper) in enumerate(bounds):
        if lower is not None:
            x_at_lower = np.array(x)
            x_at_lower[j] = lower
            if func(x_at_lower) < func(x):
                x[j] = lower
        if upper is not None:
            x_at_upper = np.array(x)
            x_at_upper[j] = upper
            if func(x_at_upper) < func(x):
                x[j] = upper
    return x

# @njit(cache=True)
def gradient_descent(func, x0, bounds, lr=0.01, max_iter=1000, max_grad=1):
    x = np.array(x0)
    for i in range(max_iter):
        grad = np.zeros_like(x)
        for j in range(len(x)):
            x_plus = np.copy(x)
            x_minus = np.copy(x)
            x_plus[j] += 1e-5
            x_minus[j] -= 1e-5
            grad[j] = (func(x_plus) - func(x_minus)) / (2 * 1e-5)
        
        grad = clip_gradient(grad, max_grad=max_grad)
        x_next = x - lr * grad
        x_next = check_and_update_bound(func, x_next, bounds)
        
        if np.linalg.norm(x_next - x) < 1e-6:
            break
        x = x_next
    
    return x


def steadyState(xi, b_S):
    delta, k, gamma, mu_S, sigma, kappa, pi = xi

    # @njit(cache=True)
    def steadyStateSystem(x):
        s, q = x
        omega = (q - mu_S) / sigma
        if omega > 7:
            omega = 7
        integral = (1 - norm.cdf(omega)) * (
            mu_S
            - q
            + sigma * norm.pdf(omega) / (1 - norm.cdf(omega))
        )
        if np.isnan(integral) or integral < 0:
            integral = 0

        # FOC for search effort
        # f1 = min(s, 1) - (1 / k * delta / (1 - delta) * integral) ** (1 / gamma)
        f1 = s - (1 / k * delta / (1 - delta) * integral) ** (1 / gamma)
        # FOC for reservation wage
        f2 = (
            -q
            + np.log(b_S)
            - k * (min(s, 1) ** (1 + gamma)) / (1 + gamma)
            + delta / (1 - delta) * min(s, 1) * integral
        )

        # Return sum of squares of the deviations
        return f1**2 + f2**2

    # Bounds for s and q
    # bounds = [(0, 1), (0, None)]  # s between 0 and 1, q greater than 0
    bounds = [(0, None), (0, None)]  # s between 0 and 1, q greater than 0

    # Initial guess
    x0 = [0.05, mu_S]

    # Solve the system of equations
    res = minimize(steadyStateSystem, x0, bounds=bounds, method="L-BFGS-B")
    s_S = res.x[0]
    logphi_S = res.x[1]

    # print(res.x)
    # optimized_x = gradient_descent(steadyStateSystem, x0, bounds)
    # s_S = optimized_x[0]
    # logphi_S = optimized_x[1]
    # print(optimized_x)
    
    s_S_cap = min(s_S,1)

    return s_S_cap, logphi_S


def solveModel(xi, institutions):
    T, b = institutions

    s, logphi = optimalPath(xi, b)
    haz, logw_reemp = predictedMoments(xi, b, s, logphi)
    surv = np.ones(len(b))
    for t in range(1, len(b)):
        surv[t] = surv[t - 1] * (1 - haz[t - 1])
    # T=96
    
    Tminb = T - len(b)
    haz_long = np.concatenate((haz, np.ones(Tminb) * haz[-1]))
    logw_reemp_long = np.concatenate((logw_reemp, np.ones(Tminb) * logw_reemp[-1]))

    surv_long = np.ones(T)
    for t in range(1, T):
        surv_long[t] = surv_long[t - 1] * (1 - haz_long[t - 1])

    D = np.sum(surv_long)
    dens_long = haz_long * surv_long
    if np.sum(dens_long)!=0:
        E_logw_reemp = np.sum(dens_long * logw_reemp_long) / np.sum(dens_long)
    else: 
        E_logw_reemp = 0

    # E_logw_reemp = 0
    return s, logphi, haz, logw_reemp, surv, D, E_logw_reemp


# UI benefit level b as function of time
def benefit_path(b1, b2, b3, welfare, T1, T2, T3, T):
    """
    Returns the benefit path given parameter values.
        Arguments:
            b1,b2,b3,welfare (all floats): values at distinct time windows during unemployment
            T1,T2,T3 (int): points in Unemp. spell when benefit levels change.
            T (int): Total number of periods
        Returns:
            benefits (array): Benefit path in unemployment
    """
    benefits = np.zeros(T)
    benefits[0:T1] = b1
    benefits[T1:T2] = b2
    benefits[T2:T3] = b3
    benefits[T3:T] = welfare

    return benefits


# function to solve individual labor supply model
def solveSingleTypeModel(xi, institutions):
    """
    Solves the model for a single-type individual.
        Arguments:
            params (array): Array of structural parameters.
            institutions (array): Array of institional parameters
        Returns:
            valOLF (array): Value function of OLF

    """

    s, logphi, haz, logw_reemp, surv, D, E_logw_reemp = solveModel(xi, institutions)

    return s, logphi, haz, logw_reemp, surv, D, E_logw_reemp


def solveMultiTypeModel(params, institutions):
    """
    Solves the model for a multi-type individual.
        Arguments:
            params (array): Array of structural parameters.
            institutions (array): Array of institional parameters
        Returns:
            Aggregated hazard rate
    """
    delta, k1, gamma, mu1, sigma, kappa, pi, k2, k3, mu2, mu3, q1, q2 = params
    q3 = 1 - q1 - q2

    xi1 = [delta, k1, gamma, mu1, sigma, kappa, pi]
    xi2 = [delta, k2, gamma, mu2, sigma, kappa, pi]
    xi3 = [delta, k3, gamma, mu3, sigma, kappa, pi]

    s1, logphi1, haz1, logw1, surv1, D1, Ew1 = solveModel(xi1, institutions)
    s2, logphi2, haz2, logw2, surv2, D2, Ew2 = solveModel(xi2, institutions)
    s3, logphi3, haz3, logw3, surv3, D3, Ew3 = solveModel(xi3, institutions)

    dens1 = haz1 * surv1
    dens2 = haz2 * surv2
    dens3 = haz3 * surv3

    survival = q1 * surv1 + q2 * surv2 + q3 * surv3
    haz_agg = (q1 * haz1 * surv1 + q2 * haz2 * surv2 + q3 * haz3 * surv3) / survival
    w_reemp_agg = (
        q1 * dens1 * logw1 + q2 * dens2 * logw2 + q3 * dens3 * logw3
    ) / (q1 * dens1 + q2 * dens2 + q3 * dens3)

    D = q1 * D1 + q2 * D2 + q3 * D3

    return haz_agg, w_reemp_agg, survival, D


class gmm:
    def __init__(self, params, target, W, inst1, inst2, disp=False):
        self.target_h12 = target[0]
        self.target_h18 = target[1]
        self.target_w12 = target[2]
        self.target_w18 = target[3]
        self.W = W
        self.inst1 = inst1
        self.inst2 = inst2
        self.params_full = params
        self.disp = disp

    def criterion(self, params):
        self.params_full.update(params)
        params_full_values = np.array(self.params_full["value"])

        haz_agg1, w_reemp_agg1, survival1, D1 = solveMultiTypeModel(
            params_full_values, self.inst1
        )
        haz_agg2, w_reemp_agg2, survival2, D2 = solveMultiTypeModel(
            params_full_values, self.inst2
        )

        target = np.concatenate(
            (self.target_h12, self.target_h18, self.target_w12, self.target_w18)
        )
        predicted = np.concatenate((haz_agg1, haz_agg2, w_reemp_agg1, w_reemp_agg2))
        moment_diff = predicted - target

        if self.disp:
            print(moment_diff)
        return np.dot(moment_diff, np.dot(self.W, moment_diff))


if __name__ == "__main__":
    # Institutional Parameters
    P12 = 12
    P18 = 18
    T = 96
    b_level = 26.7  # Daily UI benefit (800 EUR per month)
    welfare = 16.7  # Daily welfare (500 EUR per month)
    b12 = benefit_path(b_level, b_level, welfare, welfare, P12, P12, T, T)
    b18 = benefit_path(b_level, b_level, welfare, welfare, P18, P18, T, T)
    inst1 = (T, b12)
    inst2 = (T, b18)

    timevec = np.arange(1, 97)
    timevec24 = np.arange(1, 25)

    # Read Empirical Moments
    excel_file = pd.ExcelFile(momentsfile)
    targets = pd.read_excel(momentsfile, sheet_name="Data")

    target_h12 = np.array(targets["h_P12"])
    target_h18 = np.array(targets["h_P18"])
    target_w12 = np.array(targets["lnw_P12"])
    target_w18 = np.array(targets["lnw_P18"])

    blue = "#2E86AB"
    green = "#06A77D"
    red = "#A23B72"
    orange = "#F18F01"
    yellow = "#C73E1D"

    # Define target moments for GMM
    target = [target_h12, target_h18, target_w12, target_w18]

    # Weighting matrix (identity for simplicity)
    W = np.eye(len(target_h12) + len(target_h18) + len(target_w12) + len(target_w18))

    # Initial parameter guesses

    param_dict = {
        "delta": {"value": 0.995, "lower_bound": 0.99, "upper_bound": 0.999},
        "k1": {"value": 150, "lower_bound": 0.1, "upper_bound": 300},
        "gamma": {"value": 0.145, "lower_bound": 0.01, "upper_bound": 3},
        "mu1": {"value": 5.995, "lower_bound": 4, "upper_bound": 9},
        "sigma": {"value": 0.5, "lower_bound": 0.1, "upper_bound": 2},
        "kappa": {"value": 12, "lower_bound": 0, "upper_bound": 60},
        "pi": {"value": 0, "lower_bound": -1, "upper_bound": 1},
        "k2": {"value": 50, "lower_bound": 0.1, "upper_bound": 300},
        "k3": {"value": 5, "lower_bound": 0.1, "upper_bound": 300},
        "mu2": {"value": 2.5, "lower_bound": 2, "upper_bound": 9},
        "mu3": {"value": 2, "lower_bound": 2, "upper_bound": 9},
        "q1": {"value": 1, "lower_bound": 0, "upper_bound": 1},
        "q2": {"value": 0, "lower_bound": 0, "upper_bound": 1},
    }

    params_full = pd.DataFrame(param_dict).T

    params = params_full.copy()
    # params = params.drop(["delta", "gamma", "sigma", "kappa", "pi"])
    params = params.drop(["delta", "gamma", "kappa", "pi", "k2", "k3", "mu2", "mu3", "q1", "q2"])

    if 1:
        # Calculate moments based on guess parameters
        # params_full_values = np.array([0.995, 150, 0.145, 5.995, 0.5, 12, 0, 50, 5, 2.5, 2, 1, 0])
        params_full_values = np.array(params_full["value"])

        haz_agg1, w_reemp_agg1, survival1, D1 = solveMultiTypeModel(
            params_full_values, inst1
        )
        haz_agg2, w_reemp_agg2, survival2, D2 = solveMultiTypeModel(
            params_full_values, inst2
        )

        plt.clf()
        plt.plot(timevec, haz_agg1, label="Hazard, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec, haz_agg2, label="Hazard, P=18", linestyle="dashed", color=red)
        plt.plot(timevec[:24], target_h12[:24], label="Moments Hazard, P=12", linestyle="solid", color=blue)
        plt.plot(timevec[:24], target_h18[:24], label="Moments Hazard, P=18", linestyle="solid", color=red)
        plt.title("Exit Hazard")
        plt.xlabel("Months")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        plt.savefig("./log/fig_haz.pdf", bbox_inches="tight")
        plt.show()
        plt.clf()

        plt.clf()
        plt.plot(timevec[:25], w_reemp_agg1[:25], label="Wage, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec[:25], w_reemp_agg2[:25], label="Wage, P=18", linestyle="dashed", color=red)
        plt.plot(timevec[:24], target_w12[:24], label="Moments Wage, P=12", linestyle="solid", color=blue)
        plt.plot(timevec[:24], target_w18[:24], label="Moments Wage, P=18", linestyle="solid", color=red)
        plt.title("Reemployment Wage")
        plt.xlabel("Months")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        plt.show()
        plt.clf()

    if 0:
        # algo = "tao_pounders"
        algo = "scipy_lbfgsb"
        gmm_object = gmm(params_full.copy(),target, W, inst1, inst2, disp=False)
        # --- Multistart --- 
        res = em.minimize(
            criterion=gmm_object.criterion,
            params=params,
            algorithm=algo,
            multistart=True,
            multistart_options = {
                "n_cores"  : 4,
                "n_samples": 40,
                "sampling_method" : 'latin_hypercube',
                "convergence_max_discoveries" : 2,
                "share_optimizations" : .1
            }
        )

        print('Estimated Parameters:')
        print(res.params)
        print(res.criterion)
        print(res)
        col0 = res.params['value']
        col0.loc['SSE'] = res.criterion
        em.criterion_plot(res,monotone=True)

        print(gmm_object.params_full)
        gmm_object.params_full.update(res.params)
        
        # gmm_object = gmm(params_full.copy(),target, W, inst1, inst2, disp=False)
        
        print(gmm_object.params_full)

        params_result=np.array(gmm_object.params_full['value'])

        h1,w1,S1,D1 = solveMultiTypeModel(params_result,inst1)
        h2,w2,S2,D2 = solveMultiTypeModel(params_result,inst2)


        plt.clf()
        plt.plot(timevec, h1, label="Hazard, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec, h2, label="Hazard, P=18", linestyle="dashed", color=red)
        plt.plot(timevec[:30], target_h12, label="Moments Hazard, P=12", linestyle="solid", color=blue)
        plt.plot(timevec[:30], target_h18, label="Moments Hazard, P=18", linestyle="solid", color=red)
        plt.title("Exit Hazard")
        plt.xlabel("Months")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        plt.savefig("./log/fig_haz.pdf", bbox_inches="tight")
        plt.show()
        plt.clf()

        plt.clf()
        plt.plot(timevec[:25], w1[:25], label="Wage, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec[:25], w2[:25], label="Wage, P=18", linestyle="dashed", color=red)
        plt.plot(timevec[:25], target_w12[:25], label="Moments Wage, P=12", linestyle="solid", color=blue)
        plt.plot(timevec[:25], target_w18[:25], label="Moments Wage, P=18", linestyle="solid", color=red)
        plt.title("Reemployment Wage")
        plt.xlabel("Months")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        plt.savefig("./log/fig_wage.pdf", bbox_inches="tight")
        plt.show()
        plt.clf()

    if 0:

        # Two UI regimes
        # b1 = np.concatenate((np.ones(12) * 190, np.ones(12) * 90))
        # b2 = np.concatenate((np.ones(18) * 190, np.ones(6) * 90))

        # lastperiod = len(b1)

        # Model Parameters
        # xi = [0.995, 150, 0.145, 5.995, 0.5, 12, 0]
        # T = 24
        # P1 = 12
        # P2 = 18
        # ben1 = np.ones(T) * 90
        # ben2 = np.ones(T) * 90
        # ben1[:P1] = 190
        # ben2[:P2] = 190
        # inst1 = T, b1
        # inst2 = T, b2

        # # Solve the model for both benefit regimes
        # s1, logphi1, haz1, logw1, surv1, D1, Ew1 = solveModel(xi, inst1)
        # s2, logphi2, haz2, logw2, surv2, D2, Ew2 = solveModel(xi, inst2)

        # # s1, logphi1, haz1, logw1, surv1, D1, Ew1 = solveModel1(xi, b1)
        # # s2, logphi2, haz2, logw2, surv2, D2, Ew2 = solveModel1(xi, b2)

        # # Plotting
        # fontsize = 16
        # time = np.arange(1, lastperiod + 1)

        # # UI Benefit Paths
        # plt.figure(figsize=(8, 6))
        # plt.plot(time, b1, "r", label="P=12")
        # plt.plot(time, b2, "b", label="P=18")
        # plt.legend(fontsize=fontsize)
        # plt.xlabel("Time", fontsize=fontsize)
        # plt.title("UI Benefit Paths", fontsize=fontsize + 2)
        # plt.ylim(0, 250)  # Adjusted y-axis limit
        # plt.savefig("./log/fig21_bpath.pdf")
        # plt.show()

        # # Search Effort
        # plt.figure(figsize=(8, 6))
        # plt.plot(time, s1, "r", label="P=12")
        # plt.plot(time, s2, "b", label="P=18")
        # plt.legend(fontsize=fontsize)
        # plt.xlabel("Time", fontsize=fontsize)
        # plt.title("Search Effort", fontsize=fontsize + 2)
        # plt.ylim(0, 0.09)  # Adjusted y-axis limit
        # plt.savefig("./log/fig21_s.pdf")
        # plt.show()

        # 3.1 Standard Model 1 type

        delta = 0.995 
        k = 20 
        gamma = 0.145 
        mu1 = 5.995
        sigma = 0.5
        kappa = 12 
        pi = 0
        xi = [delta, k, gamma, mu1, sigma, kappa, pi ]
        # xi = [0.995, 150, 0.145, 5.995, 0.5, 12, 0]

        params = np.array([delta, k, gamma, mu1, sigma, kappa, pi])

        # s1, logphi1, haz1, logw1, surv1, D1, Ew1 = solveSingleTypeModel(params, inst1)
        # s2, logphi2, haz2, logw2, surv2, D2, Ew2 = solveSingleTypeModel(params, inst2)


        # plt.clf()
        # plt.plot(timevec, s1, label="Hazard, P=12", linestyle="dashed", color=blue)
        # plt.plot(timevec, s2, label="Hazard, P=18", linestyle="dashed", color=red)
        # plt.title("Exit Hazard")
        # plt.xlabel("Months")
        # plt.legend(loc="lower left")
        # plt.axvline(x=12, color="grey", linestyle="dashed")
        # plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.10)
        # plt.savefig("./log/fig_31.pdf", bbox_inches="tight")
        # plt.show()
        # plt.clf()

        # plt.clf()
        # plt.plot(timevec[:25], logw1[:25], label="Wage, P=12", linestyle="dashed", color=blue)
        # plt.plot(timevec[:25], logw2[:25], label="Wage, P=18", linestyle="dashed", color=red)
        # plt.title("Reemployment Wage")
        # plt.xlabel("Months")
        # plt.legend(loc="lower left")
        # plt.axvline(x=12, color="grey", linestyle="dashed")
        # plt.axvline(x=18, color="grey", linestyle="dashed")
        # # plt.ylim(bottom=0, top=0.15)
        # plt.savefig("./log/fig_35c.pdf", bbox_inches="tight")
        # plt.show()
        # plt.clf()

        #  Multi Type Aggregate

        k1 = 20
        k2 = 10
        k3 = 0
        mu2 = 2.5
        mu3 = 2
        q1 = 1
        q2 = 0
        paramsMulti = np.array([delta, k1, gamma, mu1, sigma, kappa, pi,k2,k3,mu2,mu3,q1,q2])

        # P=12 group
        haz_agg1, w_reemp_agg1, survival1, D1  = solveMultiTypeModel(paramsMulti, inst1)

        # P=18 group
        haz_agg2, w_reemp_agg2, survival2, D2 = solveMultiTypeModel(
            paramsMulti, inst2
        )

        plt.clf()
        plt.plot(timevec, haz_agg1, label="Hazard, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec, haz_agg2, label="Hazard, P=18", linestyle="dashed", color=red)
        plt.title("Exit Hazard")
        plt.xlabel("Months")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        plt.savefig("./log/fig_35c.pdf", bbox_inches="tight")
        plt.show()
        plt.clf()

        plt.clf()
        plt.plot(timevec, survival1, label="Survival, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec, survival2, label="Survival, P=18", linestyle="dashed", color=red)
        plt.title("Survival in Unemployment")
        plt.xlabel("Months")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        plt.savefig("./log/fig_35c.pdf", bbox_inches="tight")
        plt.show()
        plt.clf()

        plt.clf()
        plt.plot(timevec[:25], w_reemp_agg1[:25], label="Survival, P=12", linestyle="dashed", color=blue)
        plt.plot(timevec[:25], w_reemp_agg2[:25], label="Survival, P=18", linestyle="dashed", color=red)
        plt.title("Reemployment Wage")
        plt.xlabel("Months")
        plt.legend(loc="lower left")
        plt.axvline(x=12, color="grey", linestyle="dashed")
        plt.axvline(x=18, color="grey", linestyle="dashed")
        # plt.ylim(bottom=0, top=0.15)
        plt.savefig("./log/fig_35c.pdf", bbox_inches="tight")
        plt.show()
        plt.clf()

    