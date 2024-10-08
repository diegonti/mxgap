# MXene Learning Models

In the `models/` folder, all the trained models are saved as a pickle object, which can be easily imported and used. These models have been trained with 4356 terminated MXenes, from the data obtained in a previous high-throughput computational screening (see [EEM article](https://doi.org/10.1002/eem2.12774)).

The `MODELS_LIST.txt` is a text file containing all models and options in a well organized manner (see below). While in `NORM_INFO.txt` are present the normalization constants for each feature used. This files are not meant to be changed.

Six different Machine Learning models were trained with the MXene data: Gradient Boosting (GB), Random Forest (RF), Support Vector Machine (SV), Multi-Layer Perceptron (MLP), Logistic Regression (LR), and Kernel Ridge Regression (KRR). The GB, RF, SV, and MLP models were trained for both classification (C) of MXenes into metallic or semiconductor and Regression (R) to predict the bandgap. LR was exclusively used for Classification, as it cannot handle regression tasks, while KRR was used only for bandgap prediction.

For the Regressors, three different types are given, the ones trained with the full set of MXenes (full), trained with only semiconductor MXenes (_onlygap) and trained by predicting VBM and CBM (_edges).

The features these model need are elemental, structural and electronic, meaning it needs DOS information. A version of the models trained without DOS information is also given (_notDOS).

**The best model (and the default option in the program) is the GBC + RFR_only gap combination.**


### Table of possible model tags:

<div align="center">

| Classifiers	  |   Regressors (full) |   Regressors (only gap) |   Regressors (edges) |
|-----------------|---------------------|-------------------------|--------------------- |
| GBC             |    GBR              |   GBR_onlygap           |   GBR_edges          |
| RFC             |    RFR              |   RFR_onlygap           |   RFR_edges          | 
| SVC             |    SVR              |   SVR_onlygap           |   SVR_edges          | 
| MLPC            |    MLPR             |   MLPR_onlygap          |   MLPR_edges         | 
| LR              |    KRR              |   KRR_onlygap           |   KRR_edges          | 
| GBC_notDOS      |    GBR_notDOS       |   GBR_onlygap_notDOS    |   GBR_edges_notDOS   | 
| RFC_notDOS      |    RFR_notDOS       |   RFR_onlygap_notDOS    |	  RFR_edges_notDOS   | 
| SVC_notDOS      |    SVR_notDOS       |   SVR_onlygap_notDOS    |   SVR_edges_notDOS   | 
| MLPC_notDOS     |    MLPR_notDOS      |   MLPR_onlygap_notDOS   |   MLPR_edges_notDOS  | 
| LR_notDOS       |    KRR_notDOS       |   KRR_onlygap_notDOS    |   KRR_edges_notDOS   | 

</div>