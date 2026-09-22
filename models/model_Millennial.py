import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize


const_dict = {
    'Celsius2Kelvin': 273.15,
    'gas_const': 8.31446,
    'change_in_entropy': 710.0,
    'rho': 26.235,
}

param_dict = {
    'p_i': 0.66,
    'p_a': 0.33,
    
    'alpha_pl': 2.5e12,
    'Ea_pl': 64320.0,
    
    # 'Ha_pl': 6.5*10e3,
    'Ha_pl': 5.9*10e3,
    'Hdl_pl': 1.70*10e4,
    'Hdh_pl': 2.23*10e4,
    
    'porosity': 0.6,
    'Km_pl': 10000,
    'k_p2a': 0.02,
    'k_abreak': 0.019,
    'k_leach': 0.0015,
    'k_aff_pl': 10000,
    'p1': 0.186,
    'p2': 0.216,
    'kaff_des': 1.0,
    'sorption_c': 0.86,
    'Km_lb': 290.0,
    
    'alpha_lb': 2.6e12,
    'Ea_lb': 60260.0,
    
    # 'Ha_lb': 6.5*10e3,
    'Ha_lb': 5.5*10e3,
    'Hdl_lb': 1.70*10e4,
    'Hdh_lb': 2.23*10e4,
    
    'k_md': 0.0036,
    'p_b': 0.5,
    'k_m2a': 0.005,
    'cue_ref': 0.60,
    'cue_t': 0.012,
    'tae_ref': 15,
    'matpot': -15.0,
    'lambda': 2.1e-4,
    'kamin': 0.2,
}


def ts__arrhenius(T_soil, alpha, Ea, R=8.3143, T_k0=273.15):
    T_soil_k = T_soil + T_k0
    k = alpha * np.exp(-1.0 * Ea / (R * T_soil_k))
    return k


def ts__arrhenius_biology(T_soil, Ha, Hdl, Hdh, S, rho=26.235, R=8.3143, T_k0=273.15):
    T_soil_k = T_soil + T_k0
    k = 1 * np.exp( rho - Ha / (R * T_soil_k) ) / ( 1.0 + np.exp( (Hdl - S * T_soil_k) / (R * T_soil_k) ) + np.exp( ((S * T_soil_k) - Hdh) / (R * T_soil_k) ) )
    return k


def calc__Vmax_pl(T_soil, const_dict, param_dict):
    # Vmax_pl = param_dict['alpha_pl'] * np.exp(-1.0 * param_dict['Ea_pl'] / (const_dict['gas_const'] * (T_soil + const_dict['Celsius2Kelvin'])))
    # Vmax_pl = ts__arrhenius(T_soil=T_soil, alpha=param_dict['alpha_pl'], Ea=param_dict['Ea_pl'], R=const_dict['gas_const'], T_k0=const_dict['Celsius2Kelvin'])
    Vmax_pl = ts__arrhenius_biology(T_soil=T_soil, Ha=param_dict['Ha_pl'], Hdl=param_dict['Hdl_pl'], Hdh=param_dict['Hdh_pl'], rho=const_dict['rho'], S=const_dict['change_in_entropy'], R=const_dict['gas_const'], T_k0=const_dict['Celsius2Kelvin'])
    return Vmax_pl


def calc__Vmax_lb(T_soil, const_dict, param_dict):
    # Vmax_lb = param_dict['alpha_lb'] * np.exp(-1.0 * param_dict['Ea_lb'] / (const_dict['gas_const'] * (T_soil + const_dict['Celsius2Kelvin'])))
    # Vmax_lb = ts__arrhenius(T_soil=T_soil, alpha=param_dict['alpha_lb'], Ea=param_dict['Ea_lb'], R=const_dict['gas_const'], T_k0=const_dict['Celsius2Kelvin'])
    Vmax_lb = ts__arrhenius_biology(T_soil=T_soil, Ha=param_dict['Ha_lb'], Hdl=param_dict['Hdl_lb'], Hdh=param_dict['Hdh_lb'], rho=const_dict['rho'], S=const_dict['change_in_entropy'], R=const_dict['gas_const'], T_k0=const_dict['Celsius2Kelvin'])
    return Vmax_lb


def calc__scalar_wd(SM, param_dict):
    scalar_wd = (SM / param_dict['porosity'])**0.5
    return scalar_wd


def model_forward(POC, LMWC, AGC, MIC, MAOC, param_dict, const_dict, T_soil, SM, C_input, depth=0.2, pH=7.0, BD=1000, claysilt=80):
    
    Vmax_pl = calc__Vmax_pl(T_soil=T_soil, const_dict=const_dict, param_dict=param_dict)
    
    scalar_wd = calc__scalar_wd(SM=SM, param_dict=param_dict)
    
    scalar_wb = np.exp(param_dict['lambda'] * param_dict['matpot']) * (param_dict['kamin'] + (1 - param_dict['kamin']) * ((param_dict['porosity'] - SM) / param_dict['porosity'])**0.5) * scalar_wd
    
    kaff_lm = np.exp(-1.0 * param_dict['p1'] * pH - param_dict['p2']) * param_dict['kaff_des']
    
    Qmax = depth * BD * claysilt * param_dict['sorption_c']

    Vmax_lb = calc__Vmax_lb(T_soil=T_soil, const_dict=const_dict, param_dict=param_dict)
    
    
    ### decomposition
    # POC -> LMWC
    f_POC_LMWC = 0
    if POC > 0 and MIC > 0:
        f_POC_LMWC = Vmax_pl * scalar_wd * POC * ( MIC / (param_dict['Km_pl'] + MIC) )

    # POC -> AGC
    f_POC_AGC = 0
    if POC > 0:
        f_POC_AGC = param_dict['k_p2a'] * scalar_wd * POC

    # AGC -> MAOC + POC
    f_AGC_break = 0
    if AGC > 0:
        f_AGC_break = param_dict['k_abreak'] * scalar_wd * AGC

    # LMWC -> out of system (leaching)
    f_LMWC_leach = 0
    if LMWC > 0:
        f_LMWC_leach = param_dict['k_leach'] * scalar_wd * LMWC

    # LMWC -> MAOC
    f_LMWC_MAOC = 0
    if LMWC > 0 and MAOC >= 0:
        f_LMWC_MAOC = scalar_wd * kaff_lm * LMWC * max(0.0, (1 - MAOC / Qmax))

    # MAOM -> LMWC
    f_MAOC_LMWC = 0
    if MAOC > 0:
        f_MAOC_LMWC = param_dict['kaff_des'] * MAOC / Qmax
    
    # LMWC -> MIC
    f_LMWC_MIC = 0
    if LMWC > 0 and MIC > 0:
        f_LMWC_MIC = Vmax_lb * scalar_wb * MIC * ( LMWC / (param_dict['Km_lb'] + LMWC) )

    # MIC -> MAOC/LMWC
    f_MIC_dead = 0
    if MIC > 0:
        f_MIC_dead = param_dict['k_md'] * MIC**2.0

    # MAOC -> AGC
    f_MAOC_AGC = 0
    if MAOC > 0:
        f_MAOC_AGC = param_dict['k_m2a'] * scalar_wd * MAOC

    # MIC -> atmosphere
    f_MIC_atm = 0
    if MIC > 0 and LMWC > 0:
        f_MIC_atm = f_LMWC_MIC * ( 1.0 - ( param_dict['cue_ref'] - param_dict['cue_t'] * (T_soil - param_dict['tae_ref']) ) )

    ### Update state variables
    # Equation 1
    dPOC = C_input * param_dict['p_i'] + f_AGC_break * param_dict['p_a'] - f_POC_AGC - f_POC_LMWC

    # Equation 7
    dLMWC = C_input * (1.0 - param_dict['p_i']) - f_LMWC_leach + f_POC_LMWC - f_LMWC_MAOC - f_LMWC_MIC + f_MIC_dead * (1.0 - param_dict['p_b']) + f_MAOC_LMWC
    
    # Equation 17
    dAGC = f_MAOC_AGC + f_POC_AGC - f_AGC_break

    # Equation 20
    dMIC = f_LMWC_MIC - f_MIC_dead - f_MIC_atm

    # Equation 19
    dMAOC = f_LMWC_MAOC - f_MAOC_LMWC + f_MIC_dead * param_dict['p_b'] - f_MAOC_AGC + f_AGC_break * (1.0 - param_dict['p_a'])
  
    POC = POC + dPOC
    LMWC = LMWC + dLMWC
    AGC = AGC + dAGC
    MIC = MIC + dMIC
    MAOC = MAOC + dMAOC
    POC = max(POC, 0)
    LMWC = max(LMWC, 0)
    AGC = max(AGC, 0)
    MIC = max(MIC, 0)
    MAOC = max(MAOC, 0)
    return POC, LMWC, AGC, MIC, MAOC


def get_steady_state_c_pools(param_dict, const_dict,
                             spin_up_steps=365*100,
                             c_input_test=0.5,
                             T_soil=10, SM=0.15,
                             depth=0.2, pH=7.0, BD=1000, claysilt=80):
    time_list = np.arange(0, spin_up_steps+1, 1)
    f_POC = 0.15
    f_MAOC = 0.70
    f_AGC = 0.10
    f_MIC = 0.04
    f_LMWC = 0.01
    SOC_init = 1
    POC_init = SOC_init * f_POC
    MAOC_init = SOC_init * f_MAOC
    AGC_init = SOC_init * f_AGC
    MIC_init = SOC_init * f_MIC
    LMWC_init = SOC_init * f_LMWC
    
    POC_list = []
    LMWC_list = []
    AGC_list = []
    MIC_list = []
    MAOC_list = []
    SOC_list = []
    for t in time_list:
        if t <= 0:
            POC = POC_init
            LMWC = LMWC_init
            AGC = AGC_init
            MIC = MIC_init
            MAOC = MAOC_init
        else:
            POC, LMWC, AGC, MIC, MAOC = model_forward(
                POC=POC, LMWC=LMWC, AGC=AGC, MIC=MIC, MAOC=MAOC,
                param_dict=param_dict, const_dict=const_dict,
                T_soil=T_soil,
                SM=SM,
                C_input=c_input_test,
                depth=depth, pH=pH, BD=BD, claysilt=claysilt
            )
        SOC = POC + LMWC + AGC + MIC + MAOC
        POC_list.append(POC)
        MAOC_list.append(MAOC)
        AGC_list.append(AGC)
        MIC_list.append(MIC)
        LMWC_list.append(LMWC)
        SOC_list.append(SOC)
    SOC_ss = SOC_list[-1]
    POC_ss = POC_list[-1]
    MAOC_ss = MAOC_list[-1]
    AGC_ss = AGC_list[-1]
    MIC_ss = MIC_list[-1]
    LMWC_ss = LMWC_list[-1]
    return SOC_ss, POC_ss, MAOC_ss, AGC_ss, MIC_ss, LMWC_ss


def model_spin_up__cali_cinput(param_dict, const_dict,
                               SOC_t0_obs,
                               spin_up_steps=365*100,
                               c_input_test=0.5,
                               T_soil=10, SM=0.15,
                               tolerance=0.0001, max_iter=100):
    c_input_ss = c_input_test
    for i in range(max_iter):
        SOC_ss_mod, POC_ss_mod, MAOC_ss_mod, AGC_ss_mod, MIC_ss_mod, LMWC_ss_mod = get_steady_state_c_pools(param_dict=param_dict, const_dict=const_dict,
                                                                                                            spin_up_steps=365*100,
                                                                                                            c_input_test=c_input_ss,
                                                                                                            T_soil=10, SM=0.15)
        ratio = SOC_t0_obs / SOC_ss_mod
        error = abs(SOC_t0_obs - SOC_ss_mod)
        # print(f"Iter {i+1}: C input={c_input_ss:.3f} -> Sim_SOC = {SOC_ss_mod:.1f} (Target = {SOC_t0_obs})")
        if error < SOC_t0_obs * tolerance:
            print(">>> Success!")
            return c_input_ss, SOC_ss_mod, POC_ss_mod, MAOC_ss_mod, AGC_ss_mod, MIC_ss_mod, LMWC_ss_mod
        c_input_ss = c_input_ss * ratio
    print("Warning: finding c input failed, maybe because of the restrict of Qmax")
    return c_input_ss, SOC_ss_mod, POC_ss_mod, MAOC_ss_mod, AGC_ss_mod, MIC_ss_mod, LMWC_ss_mod


def model_simulation(POC, LMWC, AGC, MIC, MAOC,
                     param_dict, const_dict,
                     time_list, c_input_list, T_soil_list, SM_list,
                     depth=0.2, pH=7.0, BD=1000, claysilt=80):
    POC_list = []
    LMWC_list = []
    AGC_list = []
    MIC_list = []
    MAOC_list = []
    SOC_list = []
    for i in range(len(time_list)):
        c_input = c_input_list[i]
        T_soil = T_soil_list[i]
        SM = SM_list[i]
        POC, LMWC, AGC, MIC, MAOC = model_forward(
            POC=POC, LMWC=LMWC, AGC=AGC, MIC=MIC, MAOC=MAOC,
            param_dict=param_dict, const_dict=const_dict,
            T_soil=T_soil,
            SM=SM,
            C_input=c_input,
            depth=depth, pH=pH, BD=BD, claysilt=claysilt
        )
        SOC = POC + LMWC + AGC + MIC + MAOC
        POC_list.append(POC)
        MAOC_list.append(MAOC)
        AGC_list.append(AGC)
        MIC_list.append(MIC)
        LMWC_list.append(LMWC)
        SOC_list.append(SOC)
    return SOC_list, POC_list, MAOC_list, AGC_list, MIC_list, LMWC_list
