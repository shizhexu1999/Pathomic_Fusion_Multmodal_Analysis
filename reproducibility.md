**********************************************
## **Reproducibility of "19 Pathomic Fusion: an interpretable attention-based framework that integrates genomics and imaging to predict cancer outcomes (SX263)"**
**********************************************

## Summary
We give more detailed guidance here on reproducing the results of TCGA-GBMLGG and TCGA-KIRC in our report, and provide specific commands for data splitting, training, testing, and both local and global feature explanations.

## Processed Datasets
In our `README.md` and report, we have briefly introduced processed data of GBMLGG and KIRC, which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/16dPY5ekOK3Zu67Cctm4yY868XlCCqegn?usp=sharing).
You can find details of the data structure in the [pathomic_fusion_reproducibility](https://github.com/mahmoodlab/PathomicFusion/blob/master/data/TCGA_GBMLGG/README.md) from the original paper.

## Pretrained Models and Checkpoints
All pretrained models and predictions can be downloaded from [Google Drive](https://drive.google.com/drive/folders/16dPY5ekOK3Zu67Cctm4yY868XlCCqegn?usp=sharing). Please unzip the file before use.

## Run Evaluation Notebooks Directly for Instant Results
The `checkpoints` folder in this repository contains the training and testing results for each model of TCGA-GBMLGG and TCGA-KIRC at every epoch (15-fold). You can directly read the results by running evaluation notebooks as follows. Note that notebooks also contain the results from the original paper.

- **Evaluation-GBMLGG_paper.ipynb**: It uses the original data splits to reproduce the results of survival analysis and grade classification for TCGA-GBMLGG, and evaluation plots in our report are generated from this notebook.
- **Evaluation-GBMLGG.ipynb**: It uses our own data splits to produce the results of survival analysis for TCGA-GBMLGG, and the pattern in the c-index corroborates the conclusion from the original data splits. It also contains the improved results from multilinear/quadrilinear fusion.
- **Evaluation-GBMLGG_paper_conch.ipynb**: It uses our own data splits that store CONCH features to produce the results of survival analysis for TCGA-GBMLGG.
- **Evaluation-KIRC_paper.ipynb**: It uses the original data splits to reproduce the results of survival analysis for TCGA-KIRC, and evaluation plots in our report are generated from this notebook.

## Make Data Splits
Key commands for making data splits.

**Splits for survival analysis of TCGA-GBMLGG**
```bash
$ python src/make_splits/make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st
$ python src/make_splits/make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st --use_rnaseq 1
$ python src/make_splits/make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15_paper --gpu_ids 0
$ python src/make_splits/make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15_paper --use_rnaseq 1 --gpu_ids 0
```

**Splits for grade classification of TCGA-GBMLGG**
```bash
$ python src/make_splits/make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --use_vgg_features 0 --roi_dir all_st
$ python src/make_splits/make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 0 --roi_dir all_st --use_rnaseq 1
$ python src/make_splits/make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15_paper --gpu_ids 0 --act_type LSM --label_dim 3
$ python src/make_splits/make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15_paper --use_rnaseq 1 --gpu_ids 0 --act_type LSM --label_dim 3
```

**Splits for testing CNN of TCGA-GBMLGG**
```bash
$ python src/make_splits/make_splits_CNN.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15_paper --gpu_ids 0
$ python src/make_splits/make_splits_CNN.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15_paper --act_type LSM --label_dim 3 --gpu_ids 0
```

**Splits for extracting CONCH features of TCGA-GBMLGG**
```bash
$ python src/make_splits/make_splits_conch.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st
$ python src/make_splits/make_splits_conch.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st --use_rnaseq 1
$ python src/make_splits/make_splits_conch.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15_conch --gpu_ids 0
$ python src/make_splits/make_splits_conch.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15_conch  --use_rnaseq 1 --gpu_ids 0
```

**Splits for Integrated Gradients**
```bash
$ python src/make_splits/make_splits_whole_data.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st --use_rnaseq 1 --histomolecular_type idhwt_ATC
```

**Splits for multilinear fusion**
```bash
$ python src/make_splits/make_splits_multilinear_fusion.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15_rnaseq --use_rnaseq 1 --gpu_ids 0 --use_conch_features 1
```

## Train and Test for GBMLGG (Survival Analysis)
Key commands for training and testing each model for survival analysis of GBMLGG.

**CNN**
```bash
$ python src/main.py --exp_name surv_15_paper --task surv --mode path --model_name path --niter 0 --niter_decay 50 --batch_size 8 --lr 0.0005 --reg_type none --lambda_reg 0 --gpu_ids 0
$ python src/test_15_fold.py --exp_name surv_15_paper --task surv --mode path --model_name path --niter 0 --niter_decay 50 --batch_size 8 --lr 0.0005 --reg_type none --lambda_reg 0 --gpu_ids 0 --use_vgg_features 1
```

**GCN**
```bash
$ python src/main.py --exp_name surv_15_paper --task surv --mode graph --model_name graph --niter 0 --niter_decay 50 --lr 0.002 --init_type max --reg_type none --lambda_reg 0 --use_vgg_features 1 --gpu_ids 0
```

**SNN**
```bash
$ python src/main.py --exp_name surv_15_paper --task surv --mode omic --model_name omic --niter 0 --niter_decay 50 --batch_size 64 --reg_type all --init_type max --lr 0.002 --weight_decay 5e-4 --gpu_ids 0 --use_rnaseq 1 --input_size_omic 320 --verbose 1
```

**CNN+CNN**
```bash
$ python src/main.py --exp_name surv_15_paper --task surv --mode pathpath --model_name pathpath_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --label_dim 1 --reg_type none
```

**GCN+GCN**
```bash
$ python src/main.py --exp_name surv_15_paper --task surv --mode graphgraph --model_name graphgraph_fusion --niter 10 --niter_decay 10 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --label_dim 1 --reg_type none
```

**SNN+SNN**
```bash
$ python src/main.py --exp_name surv_15_paper --task surv --mode omicomic --model_name omicomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --gpu_ids 0 --label_dim 1 --reg_type all --use_rnaseq 1 --input_size_omic 320
```

**CNN+SNN**
```bash
$ python src/main.py --exp_name surv_15_paper --task surv --mode pathomic --model_name pathomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --omic_gate 0 --use_rnaseq 1 --input_size_omic 320
```

**GCN+SNN**
```bash
$ python src/main.py --exp_name surv_15_paper --task surv --mode graphomic --model_name graphomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --omic_gate 0 --use_rnaseq 1 --input_size_omic 320
```

**CNN+GCN+SNN**
```bash
$ python src/main.py --exp_name surv_15_paper --task surv --mode pathgraphomic --model_name pathgraphomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion_A --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --omic_gate 0 --grph_scale 2 --use_rnaseq 1 --input_size_omic 320
```

## Train and Test for GBMLGG (Grade Classification)
Key commands for training and testing each model for grade classification of GBMLGG.

**CNN**
```bash
$ python src/main.py --exp_name grad_15_paper --task grad --mode path --model_name path --niter 0 --niter_decay 50 --batch_size 8 --lr 0.0005 --reg_type none --lambda_reg 0 --act LSM --label_dim 3 --gpu_ids 0
$ python src/test_15_fold.py --exp_name grad_15_paper --task grad --mode path --model_name path --niter 0 --niter_decay 50 --batch_size 8 --lr 0.0005 --reg_type none --lambda_reg 0 --act LSM --label_dim 3 --gpu_ids 0 --use_vgg_features 1
```

**GCN**
```bash
$ python src/main.py --exp_name grad_15_paper --task grad --mode graph --model_name graph --niter 0 --niter_decay 50 --lr 0.002 --init_type max --reg_type none --lambda_reg 0 -use_vgg_features 1 --act LSM --label_dim 3 --gpu_ids 0
```

**SNN**
```bash
$ python src/main.py --exp_name grad_15_paper --task grad --mode omic --model_name omic --niter 0 --niter_decay 50 --batch_size 64 --reg_type all --init_type max --lr 0.002 --weight_decay 5e-4 --act LSM --label_dim 3 --gpu_ids 0
```

**CNN+CNN**
```bash
$ python src/main.py --exp_name grad_15_paper --task grad --mode pathpath --model_name pathpath_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --act LSM --label_dim 3 --reg_type none
```

**GCN+GCN**
```bash
$ python src/main.py --exp_name grad_15_paper --task grad --mode graphgraph --model_name graphgraph_fusion --niter 10 --niter_decay 10 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --act LSM --label_dim 3 --reg_type none
```

**SNN+SNN**
```bash
$ python src/main.py --exp_name grad_15_paper --task grad --mode omicomic --model_name omicomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --gpu_ids 0 --act LSM --label_dim 3 --reg_type all
```

**CNN+SNN**
```bash
$ python src/main.py --exp_name grad_15_paper --task grad --mode pathomic --model_name pathomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --path_gate 0 --act LSM --label_dim 3
```

**GCN+SNN**
```bash
$ python src/main.py --exp_name grad_15_paper --task grad --mode graphomic --model_name graphomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --grph_gate 0 --act LSM --label_dim 3
```

**CNN+GCN+SNN**
```bash
$ python train_cv.py --exp_name grad_15 --task grad --mode pathgraphomic --model_name pathgraphomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion_B --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --path_gate 0 --act LSM --label_dim 3
```

## Train and Test for GBMLGG (Survival Analysis)
Key commands for training and testing each model for survival analysis of CCRCC.

**CNN**
```bash
$ python src/main_KIRC.py --checkpoints_dir ./checkpoints/TCGA_KIRC/ --dataroot data/TCGA_KIRC/ --exp_name surv_15_paper --task surv --mode path --model_name path --niter 0 --niter_decay 30 --batch_size 8 --lr 0.0005 --reg_type none --lambda_reg 0
```

**GCN**
```bash
$ python src/main_KIRC.py --checkpoints_dir ./checkpoints/TCGA_KIRC/ --dataroot data/TCGA_KIRC/ --exp_name surv_15_paper --mode graph --init_type max --lambda_reg 0.0003 --model_name graph --niter_decay 25 --reg_type none --use_vgg_features 1
```

**SNN**
```bash
$ python src/main_KIRC.py --checkpoints_dir ./checkpoints/TCGA_KIRC/ --dataroot data/TCGA_KIRC/ --exp_name surv_15_paper --mode omic --init_type max --lambda_reg 0.0003 --model_name omic --niter_decay 30 --reg_type all --weight_decay 0.0005 --input_size_omic 362 --batch_size 32
```

**CNN+CNN**
```bash
$ python src/main_KIRC.py --checkpoints_dir ./checkpoints/TCGA_KIRC/ --dataroot data/TCGA_KIRC/ --exp_name surv_15_paper --mode pathpath --model_name pathpath_fusion --lr 0.002 --niter 10 --niter_decay 20 --reg_type none --use_vgg_features 1
```

**GCN+GCN**
```bash
$ python src/main_KIRC.py --checkpoints_dir ./checkpoints/TCGA_KIRC/ --dataroot data/TCGA_KIRC/ --exp_name surv_15_paper --mode graphgraph --model_name graphgraph_fusion --niter 10 --niter_decay 10 --lr 0.0001 --reg_type none --use_vgg_features 1
```

**SNN+SNN**
```bash
$ python src/main_KIRC.py --checkpoints_dir ./checkpoints/TCGA_KIRC/ --dataroot data/TCGA_KIRC/ --exp_name surv_15_paper --mode omicomic --model_name omicomic_fusion --lr 0.0001 --niter 10 --niter_decay 20 --input_size_omic 362
```

**CNN+SNN**
```bash
$ python src/main_KIRC.py --checkpoints_dir ./checkpoints/TCGA_KIRC/ --dataroot data/TCGA_KIRC/ --exp_name surv_15_paper --mode pathomic --model_name pathomic_fusion --niter 10 --niter_decay 20 --omic_gate 0 --use_rnaseq 1 --use_vgg_features 1 --input_size_omic 362 --lr 0.0001 --beta1 0.5
```

**GCN+SNN**
```bash
$ python src/main_KIRC.py --checkpoints_dir ./checkpoints/TCGA_KIRC/ --dataroot data/TCGA_KIRC/ --exp_name surv_15_paper --mode graphomic --model_name graphomic_fusion --niter 10 --niter_decay 20 --omic_gate 0 --use_rnaseq 1 --use_vgg_features 1 --input_size_omic 362 --lr 0.0001 --beta1 0.5
```

**CNN+GCN+SNN**
```bash
$ python src/main_KIRC.py --checkpoints_dir ./checkpoints/TCGA_KIRC/ --dataroot data/TCGA_KIRC/ --exp_name surv_15_paper --mode pathgraphomic --model_name pathgraphomic_fusion --niter 10 --niter_decay 20 --omic_gate 0 --use_rnaseq 1 --use_vgg_features 1 --input_size_omic 362 --grph_scale 2 --lr 0.0001 --beta1 0.5 --fusion_type pofusion_A --grph_scale 2
```

## Multilinear Fusion

**CNN+GCN+SNN+CONCH**

```bash
$ python src/main_multilinear_fusion.py --exp_name surv_15_rnaseq --task surv --mode multilinear --model_name multilinear_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type multilinearfusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --use_conch_features 1 --gpu_ids 0 --omic_gate 0 --grph_scale 2 --use_rnaseq 1 --input_size_omic 320
```

## Grad-CAM
```bash
$ python src/Grad-CAM_self_written_without_classifier.py
$ python src/Grad-CAM_self_written_resnet50.py
$ python src/Grad-CAM_with_packages.py --output_dir Grad_CAM_output_with_packages_vgg19
```

## Integrated Gradients

**IG local explanation for SNN**
```bash
$ python src/IG/IG_localisation.py --exp_name surv_15_paper --task surv --mode omic --model_name omic --niter 0 --niter_decay 5 --batch_size 2000 --reg_type all --init_type max --lr 0.002 --weight_decay 5e-4 --gpu_ids 0 --use_rnaseq 1 --input_size_omic 320 --verbose 1
```

**IG local explanation for GCN**
```bash
$ python src/IG/IG_graph_final.py --mode graph --exp_name surv_15_paper
```

**IG global explanation without molecular subtyping**
```bash
$ python src/IG/IG_whole_data.py --exp_name surv_15_paper --task surv --mode omic --model_name omic --niter 0 --niter_decay 30 --batch_size 769 --reg_type all --init_type max --lr 0.002 --weight_decay 5e-4 --gpu_ids 0 --use_rnaseq 1 --input_size_omic 320 --verbose 1
```

**IG global explanation with molecular subtyping**
```bash
$ python src/IG/IG_histomolecular.py --exp_name surv_15_paper --task surv --mode omic --model_name omic --niter 0 --niter_decay 30 --batch_size 769 --reg_type all --init_type max --lr 0.002 --weight_decay 5e-4 --gpu_ids 0 --use_rnaseq 1 --input_size_omic 320 --verbose 1
```

**IG global explanation with CNN+SNN**
```bash
$ python src/IG/IG_bilinear.py --exp_name surv_15_paper —task surv --mode pathomic --model_name pathomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --omic_gate 0 --use_rnaseq 1 --input_size_omic 320 --batch_size 1000 --histomolecular_type idhmut_ATC
```

**IG global explanation with CNN+GCN+SNN**
```bash
$ python src/IG/IG_trilinear.py --exp_name surv_15_paper --task surv --mode pathgraphomic --model_name pathgraphomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion_A --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --omic_gate 0 --grph_scale 2 --use_rnaseq 1 --input_size_omic 320
```

## KANs for Genomic Modality

```bash
$ python src/KAN/main_KAN.py --exp_name surv_15_KAN --task surv --mode omic --model_name omic --niter 0 --niter_decay 1 --batch_size 159 --reg_type all --init_type max --lr 0.002 --weight_decay 5e-4 --gpu_ids 0 --use_rnaseq 1 --input_size_omic 320 --verbose 1
```
