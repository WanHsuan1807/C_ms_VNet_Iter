訓練指令範例:
python train.py --amp --detach_probmap --loss_on_all_iters --out_dir ./20260227_checkpoints
python train.py --batch_size 1 --lambda_cls 0.5 --cls_on_orig --no_norm --larger_cls --out_dir ./20260228_checkpoints
python train.py --batch_size 1 --lambda_cls 0.5 --cls_on_orig --no_norm --larger_cls --out_dir ./20260314_checkpoints --epochs 150 --focal_gamma 0.0

測試指令範例:
python test.py --ckpt .\checkpoints\best_joint.pt --data_root .\data --normalize zscore --norm batchnorm --amp --out_dir .\20260227_test --save_pred
python test.py --ckpt .\20260314_checkpoints\best_joint.pt --data_root .\data --normalize zscore --norm batchnorm --amp --out_dir .\20260314_test --save_pred --larger_cls --no_norm

推論指令範例:
python infer.py --ckpt .\20260227_checkpoints\best_joint.pt --image .\data\Test\DATA\DATA_130.nrrd --gt .\data\Test\MASK\MASK_130.nrrd --normalize zscore --norm batchnorm --out_dir .\infer_outputs --out_prefix 130 --erode_iters 2
python infer.py --ckpt .\20260305_checkpoints\best_joint.pt --image .\data\Test\DATA\DATA_130.nrrd --gt .\data\Test\MASK\MASK_130.nrrd --normalize zscore --norm batchnorm --out_dir .\infer_outputs --out_prefix 130 --erode_iters 2 --no_norm --larger_cls

=================================
(1) models/vnet.py
放你原本的 VNet（我幫你加的 return_encoder_features=True 那版）。

=================================
(2) models/cmsvnet_iter.py
放：
MultiScaleClassifier
CMSVNet
IterConfig
forward_iterative_with_losses（或 iterative forward）
compute_joint_loss（含 focal + dice）

也可以把這些和 vnet 放同一檔，但拆開比較好維護。

=================================
(3) data/dataset_abus.py
放你的 Dataset + DataLoader：
__getitem__() 回 (x, seg_gt, y_cls)
x shape [1,D,H,W]
seg_gt shape [D,H,W] (0/1)
y_cls shape [] 或 [1] (0/1)

=================================
(4) train.py
放完整訓練流程（main）：
讀 args
建 model / optimizer / scaler
epoch loop：train / val
存 best checkpoint

=================================
(5) utils/metrics.py
放：
dice 計算
cls acc
AUC（如果你要）
confusion matrix（可選）

=================================
(6) utils/config.py 或 configs/*.yaml
放超參數（lr、batch、n_iter、lambda_cls…）避免 train.py 變超亂。

python train.py --epochs 10 --batch_size 2 --log_every 10 --lambda_cls 1.0 --only_cls --freeze_backbone --no_cls_weight --debug_cls_grad --no_norm --larger_cls