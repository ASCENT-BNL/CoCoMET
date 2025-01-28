# CoCoMET
<img src="./docs/images/cocomet_logo.png" alt="Logo" width="200" height="200"/>

Convective cloud Model Evaluation Toolkit.

**Current Features**:

1. **WRF**:  
   1. tobac tracking of variables  
      1. Reflectivity  
      1. Brightness temperature  
      1. Updraft velocity  
      1. Precipitation rate  
      1. Any WRF variables in the dataset (case sensitive)
   1. MOAAP tracking of MCSs and cloud shields
   1. TAMS tracking of MCSs and cloud shields  
1. **RAMS**:  
   1. tobac tracking of variables  
      1. Reflectivity  
      1. Brightness temperature  
      1. Updraft velocity  
      1. Precipitation rate  
      1. Any WRF variables in the dataset (case sensitive)
   1. MOAAP tracking of MCSs and cloud shields  
   1. TAMS tracking of MCSs and cloud shields  
1. **MesoNH**:  
   1. tobac tracking of variables:  
      1. Reflectivity  
      1. Brightness temperature  
      1. Updraft velcoity  
      1. Any MesoNH variables in the dataset (case sensitive)
   1. MOAAP tracking of MCSs and cloud shields
1. **NEXRAD**:  
   1. Automatically grid radars  
   1. tobac tracking of variables:  
      1. Reflectivity  
1. **Multi-NEXRAD**:  
   1. Automatically grid multiple radars  
   1. tobac tracking of variables:  
      1. Reflectivity  
1. **Standardized Radar Grids (CoMET-UDAF Section S1.1.):**  
   1. tobac tracking of variables:  
      1. Reflectivitiy  
1. **GOES**  
   1. tobac tracking of variables:  
      1. Brightness temperature  
1. **Analysis**:  
   1. Calculates areas at given height  
   1. Calculates volume  
   1. Calculates echo top heights  
   1. Identifies mergers and splitters  
   1. Extracts ARM Products:  
      1. Links ARM VAP output to tracks  
      1. Links INTERPSONDE to tracks  
         1. Calculates convective initiation properties from INTERPSONDE data (CAPE, CIN, etc.)


**Planned Features**:
1. Post-processing functions
1. Add ARM Radars
1. Add RWP
1. Add IMERG Satellite Data
1. Calculate ECAPE


## User Workflow

<img src="./docs/images/comet_user_workflow.png" alt="User workflow"/>

## Internal Workflow

<img src="./docs/images/comet_internal_workflow.png" alt="Internal Workflow"/>

## Acknowledgments
This project was supported by the U.S. Department of Energy (DOE) Early Career Research Program, Atmospheric System Research (ASR) program, 
and the Office of Workforce Development for Teachers and Scientists (WDTS) under the Science Undergraduate Laboratory Internships Program (SULI).

If you are using this software for a publication, please cite: ####