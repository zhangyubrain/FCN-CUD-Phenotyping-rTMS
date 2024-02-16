# CUD classification and prediction of the obtained CUD-discriminative pattern
Kanhao Zhao, Gregory A. Fonzo, Hua Xie, Desmond J. Oathes, Corey J. Keller, Nancy B. Carlisle, Amit Etkin, Eduardo A. Garza-Villarreal, Yu Zhang. [Discriminative functional connectivity signature of cocaine use disorder links to rTMS treatment response](https://www.nature.com/articles/s44220-024-00209-1). Nature Mental Health, 2024.

<div align=center>
<img width="1000" alt="1669910392114" src="https://github.com/zhangyubrain/FCN-CUD-Phenotyping-rTMS/blob/main/img/1695421844211.png">
</div>

## dataset
[The SUDMEX-CONN dataset](https://zenodo.org/record/5123331). <br />
[The SUDMEX-TMS dataset](https://zenodo.org/record/7126853). <br />
[The UCLA-CNP dataset](https://openneuro.org/datasets/ds000030/versions/1.0.0). <br />
[The NYU dataset](http://fcon_1000.projects.nitrc.org/indi/retro/nyuCocaine.html). <br />

## Implementation
### Classification of the CUD in the discovery cohort and independent cohort
classification.py
### Prediction of the rTMS treatment, using the CUD-discriminative weights
rTMS_prediction.py
### Reproduction instructions
Once the original .nii files are obtained from the website. The files were preprocessed using fMRIPrep. Then the time signals were randomly segmented into 150 time points three times and these FCs were used as augmented data. Then we regressed out the FD values of each FC. The FCs between the discovery cohort and independent cohort were harmonized with ComBat, controlling the age, and sex information.
