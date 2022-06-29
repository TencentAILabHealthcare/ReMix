import argparse, os
import numpy as np
import mmcv
import torch
from tqdm import tqdm
from mmcv.parallel import MMDataParallel
from openselfsup.datasets import build_dataloader, build_dataset
from openselfsup.models import build_model

def nondist_forward_collect(func, data_loader, length):
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(**data)
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_all[k] = np.concatenate(
            [batch[k].numpy() for batch in results], axis=0)
        assert results_all[k].shape[0] == length

    return results_all

def extract(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='extract', **x)
    results = nondist_forward_collect(func, data_loader,
                                            len(data_loader.dataset))
    return results


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract dataset features using a pretrained model')
    parser.add_argument('--pretrained', type=str, required=True, help='path to extracted model weights')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    args = parser.parse_args()

    config_file = args.config
    cfg = mmcv.Config.fromfile(config_file)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = args.pretrained
    model = build_model(cfg.model)

    model = MMDataParallel(model, device_ids=[0])
    output_pth = f'data/Unitopatho/features'
    wsi_list_root = 'data/Unitopath/meta/wsi_list'
    for split in ['train', 'test']:
        list_files = os.listdir(f'{wsi_list_root}/{split}/')
        out = f'{output_pth}/{split}_npy'
        if not os.path.exists(out):
            os.makedirs(out)
        for list_file in tqdm(list_files):
            if not os.path.exists(f"{out}/{list_file.replace('.txt','.npy')}"):
                cfg.data.extract['data_source']['list_file'] = f'{wsi_list_root}/{split}/{list_file}'
                dataset = build_dataset(cfg.data.extract)
                data_loader = build_dataloader(
                        dataset,
                        imgs_per_gpu=cfg.data.imgs_per_gpu,
                        workers_per_gpu=cfg.data.workers_per_gpu,
                        dist=False,
                        shuffle=False)
                result_dict = extract(model, data_loader)
                features = result_dict['backbone']