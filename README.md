# SoilProcMod
This repository contains the code for the soil carbon process-oriented models. Currently we include RothC and Millennial models in this repository.

## Requirement
- Python3
- numpy

## Usage instructions

### Model structure and equations

All model parameters can be set in `param_dict` in the model source code. in `model_forward` function, all equations included in the model can be found.

### Spin up the model

- Prepare the monthly or daily climate forcing data as the model inputs.

- Run `model_forward` function recursively (usually it needs to run the model for 1,000 or 10,000 years) to reach the steady state. Two strategies can be considered for spinning up the model:
    
    - If we use a fixed carbon input, we can spin up the model to get the SOC stock (also C stock in each pool) by using the function `get_steady_state_c_pools`.

    - If we want the model to match a certain SOC stock value when the system reaches the steady state, we can spin up the model to get an estimated carbon input value by using the funtion `model_spin_up__cali_cinput`.

### Model forward

- Prepare the monthly or daily climate forcing data and carbon input data from the starting year (the first year we have soil sample data in a study area) to the ending year (the last year we have soil sample data in a study area).

- Run `model_forward` function from starting year to ending year using the forcing data and the estimated soil C stock value for each C pool derived from the spin up stage.

### Model parameters calibration and model validation

If we have soil sample data across multiple years in an area, we can use these data to calibrate the model parameters. The target is to make the simulated SOC to be closed to the observed SOC as much as possible. This calibration procedure can be done manually with expert knowledge on process-baed models; or use optimization algrithms to automatically solve it but we need to select several key parameters to be optimized and also need to set a pausible range (minimal and maximal values) for each selected parameter as a constraint when using optimization algrithms.

## License

The code and data shared in this study by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://leizhang-geo.github.io">Lei Zhang</a> are licensed under <a href="http://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"></a></p>

## Contact

For questions and supports please contact the author: Lei Zhang 张磊 (lei.zhang@lbl.gov | lei.zhang.geo@outlook.com)

Lei Zhang's [Homepage](https://leizhang-geo.github.io/)