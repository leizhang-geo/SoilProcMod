### Author: Lei Zhang
### The python code for implementing the RothC model

# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class RothCModel():
    """RothC soil model.
    Details of the model can be referred to:
    https://repository.rothamsted.ac.uk/item/8746v/rothc-26-3-a-model-for-the-turnover-of-carbon-in-soil
    https://fao-gsp.github.io/GSOCseq/the-rothc-model.html#model-description
    """
    def __init__(self):
        self.model_name_ = 'RothC'
        self.SOC_list_spin_up_ = []
        self.C_input_list_spin_up_ = []
        self.SOC_list_forward_ = []
        self.C_input_list_forward_ = []

    def fT_RothC(self, T):
        factor_a = 47.9 / (1 + np.exp(106.06 / (T + 18.27)))
        return factor_a

    def fW_RothC(self, P, E, soil_depth=30, clay=23.4, pE=1.0, bare=False):
        M = P - E * pE
        Acc_TSMD = np.zeros(len(M))
        Acc_TSMD[0] = (0 if M[0] > 0 else M[0])
        for i in range(1, len(M)):
            B = 1 if bare[i] == False else 1.8
            Max_TSMD = -(20 + 1.3 * clay - 0.01 * (clay ** 2)) * (soil_depth / 23) * (1 / B)
            if Acc_TSMD[i-1] + M[i] < 0:
                Acc_TSMD[i] = (Acc_TSMD[i-1] + M[i])
            else:
                Acc_TSMD[i] = 0
            if Acc_TSMD[i] <= Max_TSMD:
                Acc_TSMD[i] = Max_TSMD
        factor_b = [1 if x > 0.444 * Max_TSMD else (0.2 + 0.8 * ((Max_TSMD - x)/(Max_TSMD - 0.444 * Max_TSMD))) for x in Acc_TSMD]
        factor_b = np.clip(factor_b, 0.2, np.inf)
        return factor_b

    def forward(self, time_steps,
                c_inputs, DPM, RPM, BIO, HUM, IOM,
                temp, preci, evp, clay, soil_depth, bare, vege_cover=None, fPR=None, DR=1.44,
                k_DPM=10, k_RPM=0.3, k_BIO=0.66, k_HUM=0.02, k_IOM=0,
                use_mic_adj_decay=False, k_mic_half=0.409, k_mic_adj_max=3,
                plot=False):
        """
        Parameters
        ----------
        time_steps : an array-like list of time steps.
        
        c_inputs : an array-like carbon input values at all time steps, or a scalar which means a constant carbon input.
        
        DPM, RPM, BIO, HUM, IOM: the initial amount of carbon for the five pools, including the four active compartments (Decomposable Plant Material, DPM; 
            Resistant Plant Material, RPM; Microbial Biomass, BIO; and Humified Organic Matter, HUM), and an inert organic matter (IOM) compartment.
        
        temp : an array-like temperature (degrees Celsius) by time, monthly data are normally used.
        preci : an array-like rainfall (mm) by time, monthly data are normally used.
        evp : an array-like open pan evaporation (mm) by time, monthly data are normally used.
        clay : clay content of the soil (as a percentage).
        soil_depth : depth of soil layer sampled (cm) 
        bare : a boolean value represents the soil is bare or vegetated (scalar or array).
        vege_cover : a value of vegetation cover to describe the vegetation cover (vege: 0.6, bare: 1.0), the default value is 1.0 if it is not set by user.
        fPR : a decomposition factor regulated by user, the default value is 1.0 if it is not set by user.
        DR : a scalar representing the ratio of decomposable plant material to resistant plant material (DPM/RPM), default=1.44
        
        k_DPM, k_RPM, k_BIO, k_HUM, k_IOM: the values of the decomposition rates for the different pools, 
            the default values are: k_DPM=10, k_RPM=0.3, k_BIO=0.66, k_HUM=0.02, k_IOM=0
        
        use_mic_adj_decay : a boolen value for determining whether adjust the decomposation rate by considering 
            the microbal biomass (BIO pool size). A true value will allow the model use reverse Michaelis-Menten (MM) 
            equation to incorporate the negative feedback between the C input and C stock. Default='False'.
        
        k_mic_half : the MM constant represents the micriobal biomass at which the reaction rate is at half-maximum decomposation rate modifying effect.
            default=0.409 (reference: Allison, S.D. et al., 2010. Soil-carbon response to warming dependent on microbial physiology. Nature Geoscience 3, 336–340.
            https://doi.org/10.1038/ngeo846)
        
        k_mic_adj_max : the maximum decomposation rate modifying factor.
        
        plot : a boolean value to control whether plot a figure to show the trend of SOC in different pools.
        
        More details of modelling procedures can be referred to:
        https://fao-gsp.github.io/GSOCseq/modeling-approach-for-the-gsocseq.html#general-modeling-procedures
        """
        if not isinstance(bare, list):
            bare = np.array([bare for _ in range(len(time_steps))])
        if not isinstance(c_inputs, list):
            c_inputs = np.array([c_inputs for _ in range(len(time_steps))])
        
        temp = np.array(temp)
        preci = np.array(preci)
        evp = np.array(evp)
        c_inputs = np.array(c_inputs)
        bare = np.array(bare)

        # temperature effects
        fT = self.fT_RothC(T=temp)

        # moisture effects
        fW = self.fW_RothC(P=preci, E=evp, soil_depth=soil_depth, clay=23.4, pE=1.0, bare=bare)

        # vegetation cover effects (vege: 0.6, bare: 1.0)
        if vege_cover is None:
            fC = np.array([1.0] * len(time_steps))
        else:
            fC = np.array(vege_cover)
        if fPR is None:
            fPR = 1.0

        a, b, c = fT, fW, fC

        # forward the model and calculate carbon in each pool step by step
        t_length = len(time_steps)
        DPM_list = np.zeros(t_length)
        RPM_list = np.zeros(t_length)
        BIO_list = np.zeros(t_length)
        HUM_list = np.zeros(t_length)
        IOM_list = np.zeros(t_length)
        soc_list = np.zeros(t_length)
        CO2_list = np.zeros(t_length)
        DPM_list[0] = DPM
        RPM_list[0] = RPM
        BIO_list[0] = BIO
        HUM_list[0] = HUM
        IOM_list[0] = IOM
        soc_list[0] = np.sum([DPM, RPM, BIO, HUM, IOM])
        t_step_len = (np.max(time_steps) - np.min(time_steps)) / len(time_steps)
        for i in range(1, t_length):
            t = time_steps[i]
            c_input = c_inputs[i]
            
            k_mic_adj = 1.0
            if use_mic_adj_decay and BIO_list[i-1] > 0 and soc_list[i-1] - IOM_list[i-1] > 0:
                k_mic_adj = k_mic_adj_max * BIO_list[i-1] / (k_mic_half + BIO_list[i-1])
            else:
                k_mic_adj = 1.0
            
            DPM_input = c_input * (DR / (1 + DR))
            RPM_input = c_input - DPM_input
            DPM_decomp = DPM_list[i-1] * (1 * a[i] * b[i] * c[i] * k_DPM * k_mic_adj * fPR) * t_step_len
            RPM_decomp = RPM_list[i-1] * (1 * a[i] * b[i] * c[i] * k_RPM * k_mic_adj * fPR) * t_step_len
            DPM_list[i] = max(0.0, DPM_list[i-1] + DPM_input - DPM_decomp)
            RPM_list[i] = max(0.0, RPM_list[i-1] + RPM_input - RPM_decomp)

            ratio_CO2_to_BIO_HUM = 1.67 * (1.85 + 1.60 * np.exp(-0.0786 * clay))
            BIO_HUM_input = (DPM_decomp + RPM_decomp) * (1 / (1 + ratio_CO2_to_BIO_HUM))
            BIO_input = BIO_HUM_input * 0.46
            BIO_decomp = BIO_list[i-1] * (1 * a[i] * b[i] * c[i] * k_BIO * k_mic_adj * fPR) * t_step_len
            HUM_input = BIO_HUM_input * 0.54
            HUM_decomp = HUM_list[i-1] * (1 * a[i] * b[i] * c[i] * k_HUM * k_mic_adj * fPR) * t_step_len
            BIO_list[i] = max(0.0, BIO_list[i-1] + BIO_input - BIO_decomp)
            HUM_list[i] = max(0.0, HUM_list[i-1] + HUM_input - HUM_decomp)

            BIO_HUM_input_2 = (BIO_decomp + HUM_decomp) * (1 / (1 + ratio_CO2_to_BIO_HUM))
            BIO_input_2 = BIO_HUM_input_2 * 0.46
            BIO_decomp_2 = BIO_list[i] * (1 * a[i] * b[i] * c[i] * k_BIO * k_mic_adj * fPR) * t_step_len
            HUM_input_2 = BIO_HUM_input_2 * 0.54
            HUM_decomp_2 = HUM_list[i] * (1 * a[i] * b[i] * c[i] * k_HUM * k_mic_adj * fPR) * t_step_len
            BIO_list[i] = max(0.0, BIO_list[i] + BIO_input_2 - BIO_decomp_2)
            HUM_list[i] = max(0.0, HUM_list[i] + HUM_input_2 - HUM_decomp_2)

            IOM_list[i] = IOM_list[i-1]

            soc_list[i] = np.sum([DPM_list[i], RPM_list[i], BIO_list[i], HUM_list[i], IOM_list[i]])

            CO2_list[i] += (DPM_decomp + RPM_decomp) - BIO_HUM_input
            CO2_list[i] += (BIO_decomp + HUM_decomp) - BIO_HUM_input_2
        
        self.SOC_list_forward_ = soc_list
        
        if plot:
            line_width = 1.5
            plt.figure(figsize=(5.2, 3.7), dpi=120)
            plt.plot(time_steps, soc_list, label='SOC', linewidth=line_width)
            plt.plot(time_steps, DPM_list, label='DPM', linewidth=line_width)
            plt.plot(time_steps, RPM_list, label='RPM', linewidth=line_width)
            plt.plot(time_steps, BIO_list, label='BIO', linewidth=line_width)
            plt.plot(time_steps, HUM_list, label='HUM', linewidth=line_width)
            plt.plot(time_steps, IOM_list, label='IOM', linewidth=line_width)
            plt.legend()
            plt.xlabel('Time (year)', fontsize=12.5)
            plt.ylabel('SOC stock (g C m$^{-2}$)', fontsize=12.5)
            sns.despine()

        return soc_list, DPM_list, RPM_list, BIO_list, HUM_list, IOM_list, CO2_list
    
    def spin_up(self, time_steps,
                c_input, DPM, RPM, BIO, HUM, IOM, SOC_eq,
                temp, preci, evp, clay, soil_depth, bare, vege_cover=None, fPR=None, DR=1.44,
                k_DPM=10, k_RPM=0.3, k_BIO=0.66, k_HUM=0.02, k_IOM=0,
                use_mic_adj_decay=False, k_mic_half=0.409, k_mic_adj_max=3,
                show_info=False, plot=False):
        """
        When spin up (initialize) the model, the initial value of DPM, RPM, BIO and HUM pools are all set to be zero.
        The length of the spin up simulation period can usually vary between 100s to 1000s years (FAO, 2019).
        SOC_eq is the SOC stock value at the steady state, and IOM can be estimated using a function of SOC_eq by Falloon et al. (1998).
        c_input can be set as an arbitary positive value, and the c_input_eq (the estimated C input at the steady state) will be calculated
        by: c_input * ((SOC_eq - IOM) / (SOC_sim - IOM)), where SOC_sim is the simulated soil carbon after the many years run.
        
        More details of modelling procedures can be referred to:
        https://fao-gsp.github.io/GSOCseq/modeling-approach-for-the-gsocseq.html#general-framework
        """
        res = self.forward(time_steps=time_steps,
                           c_inputs=c_input, DPM=DPM, RPM=RPM, BIO=BIO, HUM=HUM, IOM=IOM,
                           temp=temp, preci=preci, evp=evp, clay=clay, soil_depth=soil_depth, bare=bare, vege_cover=vege_cover, fPR=fPR, DR=DR,
                           k_DPM=k_DPM, k_RPM=k_RPM, k_BIO=k_BIO, k_HUM=k_HUM, k_IOM=k_IOM,
                           use_mic_adj_decay=use_mic_adj_decay, k_mic_half=k_mic_half, k_mic_adj_max=k_mic_adj_max,
                           plot=plot)
        soc_list, DPM_list, RPM_list, BIO_list, HUM_list, IOM_list, CO2_list = res[0], res[1], res[2], res[3], res[4], res[5], res[6]
        SOC_sim = soc_list[-1]
        
        self.SOC_list_spin_up_ = soc_list * (SOC_eq / SOC_sim)
        self.C_input_list_spin_up_ = c_input * ((SOC_eq - IOM) / (self.SOC_list_spin_up_ - IOM))
        
        c_input_eq = c_input * ((SOC_eq - IOM) / (SOC_sim - IOM))
        RPM = ((0.184 * SOC_eq + 0.1555) * (clay + 1.275)**(-0.1158)) * 0.9902 + 0.4788
        BIO = ((0.014 * SOC_eq + 0.0075) * (clay + 8.8473)**(0.0567)) * 1.09038 + 0.04055
        HUM = ((0.7148 * SOC_eq + 0.5069) * (clay + 0.3421)**(0.0184)) * 0.9878 - 0.3818
        DPM = SOC_eq - IOM - RPM - HUM - BIO
        SOC = np.sum([DPM, RPM, BIO, HUM, IOM])
        
        if show_info:
            print('Cinput_eq = {:.3f}  SOC = {:.3f}  DPM = {:.3f}  RPM = {:.3f}  BIO = {:.3f}  HUM = {:.3f}  IOM = {:.3f}'.format(
                c_input_eq, SOC, DPM, RPM, BIO, HUM, IOM))

        return c_input_eq, SOC, DPM, RPM, BIO, HUM, IOM
