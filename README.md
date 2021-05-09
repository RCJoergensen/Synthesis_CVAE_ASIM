# Synthesis_CVAE_ASIM
This reposetory contains the documentation and code for the CVAE created for the ASIM dataset, in order to detect anomalous data and attempt to group the raw dataset. Documentation of the CVAE can be found in the report; ASIM_CVAE_SynthesisProject.pdf here on the main page.

The jupyter notebook for the final CVAE presented in the repport is found in Synthesis_ASIM_CVAE.ipynb here on the main page.
The additional scripts used in the notebook can be found in /Synthesis_CVAE_ASIM/Scripts/TrainingTheModel/

All scripts used for scoring the CVAE can be found in /Synthesis_CVAE_ASIM/Scripts/TrainingTheModel/ScoringTheCVAE/
The main script used for this is Main_ScoringTheCVAE.py with the remaining being functions called in the document.

In the Literature folder is a  link to the report presenting the original CVAE used as a baseline in this project, which is referenced several times throughout the report, but not otherwise available online.

In the TrainedModels folder all the trained models presented in the report can be found. The naming convention is the same as is being used in Table 1, p. 14, in ASIM_CVAE_SynthesisProject.pdf, where an explanation can also be found.

GroundTruth_Table.xlsx is the groundtruth dataset created for the scoring of the CVAE. However, the raw datafiles are not included, as the ASIM dataset is not publicly available.

Author: RCJ, s164044, DTU Space
Last Edited: 09/05/2021
