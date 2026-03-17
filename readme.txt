參考論文:
此專案是實現以下論文，以實現ABUS image的分割與分類:
Y. Zhou, H. Chen, Y. Li, Q. Liu, X. Xu, S. Wang, P. T. Yap, D. Shen, “Multi-task learning for segmentation and classification of tumors in 3D automated breast ultrasound images,” Medical Image Analysis, vol. 70, pp. 101918, 2021. https://doi.org/10.1016/j.media.2020.101918 

This model is a 3D convolutional neural network for two tasks: tumor segmentation and tumor classification. 
The overall structure is an encoder–decoder network.The encoder extracts features from the input 3D volume, and the decoder restores the spatial information to produce the segmentation result.
At the same time, the network also includes a classification branch.
So, this model can output both:
the segmentation map and the classification result

This means the model learns segmentation and classification together in one framework.

=================================
訓練指令範例:
python train.py --batch_size 1 --lambda_cls 0.5 --cls_on_orig --no_norm --larger_cls --out_dir ./20260314_checkpoints --epochs 150 --focal_gamma 0.0

測試指令範例:
python test.py --ckpt .\20260314_checkpoints\best_joint.pt --data_root .\data --normalize zscore --norm batchnorm --amp --out_dir .\20260314_test --save_pred --larger_cls --no_norm

推論指令範例:
python infer.py --ckpt .\20260305_checkpoints\best_joint.pt --image .\data\Test\DATA\DATA_130.nrrd --gt .\data\Test\MASK\MASK_130.nrrd --normalize zscore --norm batchnorm --out_dir .\infer_outputs --out_prefix 130 --erode_iters 2 --no_norm --larger_cls

=================================
(1) models/cmsvnet_iter.py
裡面包含：
MultiScaleClassifier
MultiScaleClassifierLarger
SingleScaleClassifierLarger
CMSVNet
IterConfig
forward_iterative_with_losses（或 iterative forward）
compute_joint_loss（含 focal + dice）

也可以把這些和 vnet 放同一檔，但拆開比較好維護。

=================================
(3) data/dataset_abus.py
 Dataset + DataLoader


=================================
(4) train.py
放完整訓練流程（main）：
讀 args
建 model / optimizer / scaler
epoch loop：train / val
存 best checkpoint

=================================
(5) utils/metrics.py
包括：
dice 計算
cls acc
AUC（如果你要）
confusion matrix（可選）
