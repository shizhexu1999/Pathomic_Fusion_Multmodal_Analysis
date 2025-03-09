"""!@mainpage 19 Pathomic Fusion: an interpretable attention-based framework
that integrates genomics and imaging to predict cancer outcomes (SX263)

@section intro_sec Introduction
This repository serves for the submission of the project 19: Pathomic Fusion.
In this repository, we provide the code, reproduced results, and evaluation
plots to complement our analysis and discussion in the report.
The executive summary and project report are located in the report folder of
this repository, or you can download it from Google Drive.
To improve the efficiency and accuracy of cancer prediction,
there is a pressing demand to develop a new model that extracts and combines
information/features across different modalities in an integrative manner.
This project aims to summarise, reproduce and improve the work in the paper
published in 2020, "Pathomic Fusion: An Integrated Framework for
Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis”,
in which the novel fusion approach combines histopathology images,
cell graph and genomic features for survival outcome prediction
and grade classification of glioma and clear cell renal cell carcinoma.
The multimodal interpretability can identify new integrative biomarkers
with diagnostic, prognostic, and therapeutic relevance.
In addition to reproducing all the results from the original paper,
including visualizations and local and global explanations,
we develop and implement new architectures such as CellViT segmentation
and multilinear/quadrilinear fusion with CONCH.

@author Shizhe Xu
@date 29 June 2024
"""
