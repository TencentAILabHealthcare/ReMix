import argparse
import glob
import os
import threading
import urllib.request
import zipfile
from os.path import join

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import img_as_ubyte, io
from skimage.color import rgb2hsv
from skimage.util import img_as_ubyte
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    print('downloading dataset')
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def unzip_data(zip_path, data_path):
    os.makedirs(data_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)


def convert_camelyon16():
    dataset_pth = 'datasets/Camelyon16'
    for token in ['0-normal', '1-tumor']:
        os.makedirs(f'{dataset_pth}/{token}-npy', exist_ok=True)
        pths = glob.glob(f'{dataset_pth}/{token}/*.csv')
        for pth in tqdm(pths, desc=f'converting Camelyon16 csv to npy, {token}'):
            df = pd.read_csv(pth)
            feats = df.to_numpy()
            np.save(pth.replace(token, f'{token}-npy').replace('.csv', '.npy'), feats)


def split_camelyon16():
    bags_csv = 'datasets/Camelyon16/Camelyon16.csv'
    bags_path = pd.read_csv(bags_csv)
    os.makedirs('datasets/Camelyon16/remix_processed', exist_ok=True)
    train_list_txt = open('datasets/Camelyon16/remix_processed/train_list.txt', 'w')
    test_list_txt = open('datasets/Camelyon16/remix_processed/test_list.txt', 'w')
    train_labels, test_labels = [], []
    for i in tqdm(range(len(bags_path)), desc='splitting Camelyon16'):
        feats_npy_pth = bags_path.iloc[i].iloc[0]
        feats_npy_pth = feats_npy_pth.replace('0-normal', '0-normal-npy')
        feats_npy_pth = feats_npy_pth.replace('1-tumor', '1-tumor-npy')
        feats_npy_pth = feats_npy_pth.replace('.csv', '.npy')
        if 'test' not in feats_npy_pth:
            train_list_txt.write(f'{feats_npy_pth},{bags_path.iloc[i].iloc[1]}\n')
            train_labels.append(bags_path.iloc[i].iloc[1])
        else:
            test_list_txt.write(f'{feats_npy_pth},{bags_path.iloc[i].iloc[1]}\n')
            test_labels.append(bags_path.iloc[i].iloc[1])
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    np.save('datasets/Camelyon16/remix_processed/test_bag_labels.npy', test_labels)
    np.save('datasets/Camelyon16/remix_processed/train_bag_labels.npy', train_labels)
    print('done')


def thres_saturation(img, t=15):
    # typical t = 15
    _img = np.array(img)
    _img = rgb2hsv(_img)
    h, w, c = _img.shape
    sat_img = _img[:, :, 1]
    sat_img = img_as_ubyte(sat_img)
    ave_sat = np.sum(sat_img) / (h * w)
    return ave_sat >= t


def slide_to_patch_jpeg(out_base, img_slides, patch_size=224):
    for s in tqdm(range(len(img_slides))):
        img_slide = img_slides[s]
        img_name = img_slide.split(os.path.sep)[-1].split('.png')[0].replace(' ', '_')
        img_class = img_slide.split(os.path.sep)[-2]
        bag_path = join(out_base, img_class, img_name)
        os.makedirs(bag_path, exist_ok=True)
        img = Image.open(img_slide).convert('RGB')
        w, h = img.size
        step_j = h // 8
        step_i = w // 8
        for i in range(8):
            for j in range(8):
                roi = img.crop((int(j * step_j), int(i * step_i), int((j + 1) * step_j), int((i + 1) * step_i)))
                if thres_saturation(roi, 30):
                    resized = cv2.resize(np.array(roi), (patch_size, patch_size))
                    patch_name = "{}_{}".format(i, j)
                    io.imsave(join(bag_path, patch_name + ".jpg"), img_as_ubyte(resized))


def split_unitopatho(download_dir):
    raw_image_pth = 'datasets/Unitopatho/raw'
    print('gathering all patches....')
    all_patches = glob.glob(f'{raw_image_pth}/*/*/*.jpg')
    print('gathered', len(all_patches), 'samples.')
    ############# For OpenSelfSup pre-training ###########################
    os.makedirs('datasets/Unitopatho/meta', exist_ok=True)
    train_txt = open('datasets/Unitopatho/meta/train.txt', 'w')
    test_txt = open('datasets/Unitopatho/meta/test.txt', 'w')
    # These files are included in the downloaded files.
    train_wsi_txt = open(f'{download_dir}/train_wsi.txt', 'r').readlines()
    test_wsi_txt = open(f'{download_dir}/test_wsi.txt', 'r').readlines()
    train_wsi_txt = [l.strip() for l in train_wsi_txt]
    test_wsi_txt = [l.strip() for l in test_wsi_txt]
    
    top_label_mapping = {'HP': 0, 'NORM': 1, 'TA.HG': 2, 'TA.LG': 3, 'TVA.HG': 4, 'TVA.LG': 5}
    for pth in tqdm(all_patches, desc='creating Unitopatho meta files'):
        pth_parsed = pth.split('/')
        top_label = top_label_mapping[pth_parsed[-3]]
        wsi_name = pth_parsed[-2].split('.ndpi')[0].replace('_', ' ')
        if wsi_name in train_wsi_txt:
            train_txt.write(f"{'/'.join(pth_parsed[-3:])}\n")
        elif wsi_name in test_wsi_txt:
            test_txt.write(f"{'/'.join(pth_parsed[-3:])}\n")
        else:
            print('Error:', wsi_name)
    ######################################################################

    ################# Extract bag list ###################################
    train_bag_pth = 'datasets/Unitopatho/meta/wsi_list/train'
    test_bag_pth = 'datasets/Unitopatho/meta/wsi_list/test'
    os.makedirs(train_bag_pth, exist_ok=True)
    os.makedirs(test_bag_pth, exist_ok=True)
    for i, pth in tqdm(enumerate(all_patches), total=len(all_patches),
                       desc='creating Unitopatho bag list'):
        pth_parsed = pth.split('/')
        top_label = top_label_mapping[pth_parsed[-3]]
        wsi_name = pth_parsed[-2].split('.ndpi')[0].replace('_', ' ')
        if wsi_name in train_wsi_txt:
            if i == 0:  # force rewrite
                patch_list = open(f'{train_bag_pth}/{wsi_name}_{top_label}.txt', 'w')
            else:
                patch_list = open(f'{train_bag_pth}/{wsi_name}_{top_label}.txt', 'a')
        elif wsi_name in test_wsi_txt:
            if i == 0:  # force rewrite
                patch_list = open(f'{test_bag_pth}/{wsi_name}_{top_label}.txt', 'w')
            else:
                patch_list = open(f'{test_bag_pth}/{wsi_name}_{top_label}.txt', 'a')
        else:
            print('Error', wsi_name)
            exit()
        file_pth = '/'.join(pth_parsed[-3:])
        patch_list.write(f'{file_pth}\n')
    ######################################################################

    ################# Gather final list ##################################
    os.makedirs('datasets/Unitopatho/remix_processed', exist_ok=True)
    train_list_txt = open('datasets/Unitopatho/remix_processed/train_list.txt', 'w')
    test_list_txt = open('datasets/Unitopatho/remix_processed/test_list.txt', 'w')
    train_files = glob.glob(f'{train_bag_pth}/*.txt')
    test_files = glob.glob(f'{test_bag_pth}/*.txt')
    train_labels, test_labels = [], []
    for f_name in train_files:
        f_name = f_name.split('/')[-1].replace('txt', 'npy')
        label = f_name.split('_')[-1].split('.')[0]
        train_labels.append(int(label))
        train_list_txt.write(f"datasets/Unitopatho/features/train_npy/{f_name},{label}\n")
        
    for f_name in test_files:
        f_name = f_name.split('/')[-1].replace('txt', 'npy')
        label = f_name.split('_')[-1].split('.')[0]
        test_labels.append(int(label))
        test_list_txt.write(f"datasets/Unitopatho/features/test_npy/{f_name},{label}\n")
    
    train_labels, test_labels = np.array(train_labels), np.array(test_labels)
    np.save('datasets/Unitopatho/remix_processed/train_bag_labels.npy', train_labels)
    np.save('datasets/Unitopatho/remix_processed/test_bag_labels.npy', test_labels)
    ######################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('--dataset', type=str, default='Camelyon16', choices=['Camelyon16', 'Unitopatho'])
    parser.add_argument('--task', type=str, nargs="+",
                        default=['download', 'convert', 'split'],
                        choices=['download', 'convert', 'split', 'crop'])
    parser.add_argument('--num_threads', type=int, default=64, help='Number of threads for parallel processing')
    parser.add_argument('--patch_size', type=int, default=224, help='Patch size')
    args = parser.parse_args()
    
    if args.dataset == 'Camelyon16':
        if args.task[0] == 'download':
            print('downloading Camelyon16 datasets (pre-computed features)')
            download_url('https://uwmadison.box.com/shared/static/l9ou15iwup73ivdjq0bc61wcg5ae8dwe.zip', 'c16-dataset.zip')
            unzip_data('c16-dataset.zip', 'datasets/Camelyon16')
            os.remove('c16-dataset.zip')
            args.task.pop(0)
            
        if args.task[0] == 'convert':
            convert_camelyon16()
            args.task.pop(0)
            
        if args.task[0] == 'split':
            split_camelyon16()
            args.task.pop(0)
            
    elif args.dataset == 'Unitopatho':
        # Please change the download_dir to your own directory.
        download_dir = '/mnt/hanbochen/datasets/colon_polyp/unitopatho/unitopath-public'
        
        if args.task[0] == 'crop':
            print('Cropping patches, this could take a while for big dataset, please be patient')
            # path to target folder
            out_base = 'datasets/Unitopatho/raw'
            os.makedirs(out_base, exist_ok=True)
            # we use the resolution of 800
            data_dir = f'{download_dir}/800'
            all_slides = glob.glob(join(data_dir, '*/*.png'))
            each_thread = int(np.floor(len(all_slides) / args.num_threads))
            threads = []
            for i in range(args.num_threads):
                if i < (args.num_threads - 1):
                    t = threading.Thread(target=slide_to_patch_jpeg,
                                         args=(out_base, all_slides[each_thread * i: each_thread * (i + 1)], args.patch_size))
                else:
                    t = threading.Thread(target=slide_to_patch_jpeg,
                                         args=(out_base, all_slides[each_thread * i:], args.patch_size))
                threads.append(t)

            for thread in threads:
                thread.start()
            # wait until all threads finish
            for thread in threads:
                thread.join()
            args.task.pop(0)
            
        if args.task[0] == 'split':
            split_unitopatho(download_dir)
            args.task.pop(0)
