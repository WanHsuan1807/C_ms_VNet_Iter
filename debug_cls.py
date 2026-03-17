import torch
from data.dataset_abus_2 import ABUSLocalConfig, ABUSLocalSegClsDataset, count_cls_labels
from models.vnet import VNet
from models.cmsvnet_iter import CMSVNet, IterConfig, forward_iterative_with_losses
from torch.utils.data import DataLoader

if __name__ == '__main__':
    cfg = ABUSLocalConfig(root='./data', split='train', normalize='zscore', bbox_margin=(8,16,16), roi_shape=(64,64,64))
    ds = ABUSLocalSegClsDataset(cfg)
    print('len ds', len(ds))
    counts = count_cls_labels(ds)
    print('counts', counts)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    
    vnet = VNet(n_channels=1, n_classes=2, n_filters=8, normalization='batchnorm', has_dropout=False)
    stage_channels=[4*8,8*8,16*8]
    model=CMSVNet(vnet, stage_channels, cls_hidden=64)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=model.to(device)

    data = next(iter(loader))
    x,seg_gt,y_cls=data
    x=x.to(device).float(); seg_gt=seg_gt.to(device); y_cls=y_cls.to(device)
    zcfg=IterConfig(n_iter=2, lambda_cls=0.7, focal_gamma=0.0, detach_probmap=True, loss_on_all_iters=True)
    nm=counts['malignant']; nn_=counts['benign']
    print('pre-forward')
    loss_joint, loss_dict, seg_last, cls_last = forward_iterative_with_losses(model,x,seg_gt,y_cls,zcfg,nm=nm,nn_=nn_)
    print('loss_joint', loss_joint.item(), 'loss_dict', {k:v.item() for k,v in loss_dict.items()})
    print('cls_last', cls_last[:2])
    
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss_joint.backward()
    for name,param in model.named_parameters():
        if 'classifier' in name:
            print(name, 'requires_grad', param.requires_grad, 'grad norm', None if param.grad is None else param.grad.norm().item())
    print('done')
