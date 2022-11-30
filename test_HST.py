import argparse
import os
import torch
import torch.nn.utils as utils
import cv2
import numpy as np
from collections import OrderedDict
from model import HST
import util_calculate_psnr_ssim as util

parser = argparse.ArgumentParser(description='Test HST')

parser.add_argument('--ckpt', type=str, default='', help='path to load checkpoint')
parser.add_argument('--scale', type=int, default=4, help='SR scale, 4 is used in the competition')
parser.add_argument('--window_size', type=int, default=8, help='window size, 8 is default')
parser.add_argument('--comp_level', type=int, default=40, help='compression level, support 10, 20, 30, 40')
parser.add_argument('--use_ensemble', action='store_true')

args = parser.parse_args()


weight = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

model = HST(img_size=64)

model.load_state_dict(weight)
model = model.cuda()

test_paths = ['Set5_comp'+str(args.comp_level), 'Set14_comp'+str(args.comp_level), 'BSD100_comp'+str(args.comp_level), 'urban100_comp'+str(args.comp_level), 'manga109_comp'+str(args.comp_level)]

output_paths = ['Set5_out', 'Set14_out', 'BSD100_out', 'urban100_out', 'manga109_out']

gts = ['Set5', 'Set14', 'BSD100', 'urban100', 'manga109']


def test(model, img):
    _, _, h_old, w_old = img.size()
    padding = args.scale * args.window_size
    h_pad = (h_old // padding + 1) * padding - h_old
    w_pad = (w_old // padding + 1) * padding - w_old
    img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
    img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
    
    img = model(img)
    img = img[..., :h_old * 4, :w_old * 4]
    return img

for i in range(len(gts)):

    output_path = output_paths[i]
    test_path = test_paths[i]
    gt = gts[i]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    f = open(os.path.join(output_path, 'log.txt'),'w')

    model.eval()
    count = 0
    with torch.no_grad():

        p = 0
        s = 0
        py = 0
        sy = 0

        for img_n in sorted(os.listdir(test_path)):
            count += 1
            lr = cv2.imread(os.path.join(test_path, img_n))
            hr_n = img_n.split(".")[0] + '.png'
            hr = cv2.imread(os.path.join(gt, hr_n))
            lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
            
            img = np.ascontiguousarray(lr.transpose((2, 0, 1)))
            img = torch.from_numpy(img).float()
            img /= 255.
            img = img.unsqueeze(0).cuda()
            E = test(model, img)
            if args.use_ensemble:
                E1 = test(model, img.flip(-1)).flip(-1)
                E2 = test(model, img.flip(-2)).flip(-2)
                E3 = test(model, img.flip(-1, -2)).flip(-1, -2)
                L_t = img.transpose(-2, -1)
                E4 = test(model, L_t).transpose(-2, -1)
                E5 = test(model, L_t.flip(-1)).flip(-1).transpose(-2, -1)
                E6 = test(model, L_t.flip(-2)).flip(-2).transpose(-2, -1)
                E7 = test(model, L_t.flip(-1, -2)).flip(-1, -2).transpose(-2, -1)

                E = (E.clamp_(0, 1) + E1.clamp_(0, 1) + E2.clamp_(0, 1) + E3.clamp_(0, 1) + E4.clamp_(0, 1) + E5.clamp_(0, 1) + E6.clamp_(0, 1) + E7.clamp_(0, 1)) / 8.0

            img = E
            sr = img.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)

            sr = sr * 255.
            sr = np.clip(sr.round(), 0, 255).astype(np.uint8)

            sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

            psnr = util.calculate_psnr(sr.copy(), hr.copy(), crop_border=4, test_y_channel=False)
            psnr_y = util.calculate_psnr(sr.copy(), hr.copy(), crop_border=4, test_y_channel=True)

            ssim = util.calculate_ssim(sr.copy(), hr.copy(), crop_border=4, test_y_channel=False)
            ssim_y = util.calculate_ssim(sr.copy(), hr.copy(), crop_border=4, test_y_channel=True)

            p += psnr
            s += ssim
            py += psnr_y
            sy += ssim_y
            f.write('{}: PSNR, {}. PSNR_Y, {}. SSIM, {}. SSIM_Y, {}.\n'.format(img_n, psnr, psnr_y, ssim, ssim_y))

            cv2.imwrite(os.path.join(output_path, hr_n), sr)


        p /= count
        s /= count
        py /= count
        sy /= count
        print(p, py, s, sy)
        f.write('avg PSNR: {}, PSNR_Y: {}, SSIM: {}, SSIM_Y: {}.'.format(p, py, s, sy))
    
    f.close()
