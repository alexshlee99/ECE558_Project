import os
import random
import numpy as np
import argparse
import mat73
from tqdm import tqdm
import torch
import torch.utils.data as data

from dataset import RetinaSimDataset
from modl import MoDL_PAI
from utils import abs_normalizer, normalizer

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
import time
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=R'checkpoint/rand_5_10/1e-3/best_psnr.pth', help='path to checkpoint')

parser.add_argument('--sysmat_dir_path', type=str, default=R"C:\Users\Alex\Desktop\Courses\ECE 558\Final Project\sys_mat", help='path to system matrices')
parser.add_argument('--sim_dir_path', type=str, default=R"C:\Users\Alex\Desktop\Courses\ECE 558\Final Project\data", help='path to dataset')

parser.add_argument('--num_layers', type=int, default=5, help='number of layers')
parser.add_argument('--num_iters', type=int, default=10, help='number of iterations')

parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--gpu', type=int, default=0, help='the number of the device')
args = parser.parse_args()


def setup_seed(seed):
    """
    Manually set seed & keep deterministic for consistency. 
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def load_full_A(mat_dir, device): 
    data_dict = mat73.loadmat(os.path.join(mat_dir, "A.mat"))
    A = torch.from_numpy(data_dict['A']).to(device)
    return A

def forward_op(x):
    # return u @ (s @ (v.T @ x))
    return A @ x

def adjoint_op(y):
    # return v @ (s @ (u.T @ y))
    return A.T @ y

def test(model, test_loader, device):
    with torch.no_grad():

        gtss = []
        preds = []
        psnrs = []
        ssims = []
        img_cnt = 0
        
        # Initialize accuracy metrics. 
        PSNR = PeakSignalNoiseRatio(data_range=(0,1), dim=(1,2,3), reduction='none').to(device)
        SSIM = StructuralSimilarityIndexMeasure(data_range=(0,1), reduction='none').to(device)
        
        model.eval()
        for _, (sds, masks, gts) in enumerate(tqdm(test_loader)):
            ### Batch size. 
            bs, _, _ = sds.shape

            ### Send to device. 
            sds = sds.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)
            gts = gts.to(device, dtype=torch.float).unsqueeze(1)

            # ### Compute x0 (with mask).
            sds_masked = sds * masks
            y = sds_masked.reshape(bs, -1).T
            x0 = adjoint_op(y).T.reshape(bs, 1, 128, 128)

            # x0_npy = abs_normalizer(x0).detach().cpu().numpy()
            # plt.imshow(x0_npy[5].squeeze(), cmap='hot')
            # plt.tight_layout()
            # plt.show()

            ### Forward through network. 
            outputs = model(x0, forward_op, adjoint_op, masks)  
            
            ### Get reconstructed image. 
            # Normalize both ground truth & predicted images.
            # (We don't use sigmoid since the signals are generated from the actual pressure!)
            norm_gts = abs_normalizer(gts)
            norm_preds = abs_normalizer(outputs)
            
            gtss.append(norm_gts.detach().cpu().numpy())
            preds.append(norm_preds.detach().cpu().numpy())

            ### Calculate accuracies. 
            img_cnt += bs
            psnrs.append(PSNR(norm_preds, norm_gts).detach().cpu().numpy())
            ssims.append(SSIM(norm_preds, norm_gts).detach().cpu().numpy())

        # At the end, concatenate along the batch dimension (dim=0)
        all_gts = np.concatenate(gtss, axis=0)
        all_preds = np.concatenate(preds, axis=0)   # [N, 1, 128, 128]
        all_psnrs = np.concatenate(psnrs, axis=0)   # [N,]
        all_ssims = np.concatenate(ssims, axis=0)   # [N,]
    return all_gts, all_preds, all_psnrs, all_ssims

def main():
    """
    Main function for train/test. 
    """
    ### Deterministic. 
    setup_seed(0)
    
    ### Image transforms.  
    test_transforms = None    
    
    ### Load datasets. 
    test_dataset = RetinaSimDataset(args.sim_dir_path, phase='test', transform=test_transforms)
    test_loader = data.DataLoader(test_dataset, 
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.workers,
                                        pin_memory=True)
    
    ### Load model. 
    model = MoDL_PAI(args.num_layers, args.num_iters)  # num of channels = num of transducer elements
    model.to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Model is ready for inference
    
    ### Test.
    start = time.time()
    gts, preds, psnrs, ssims = test(model, test_loader, device)
    end = time.time()
    print(f'Mean PSNR: {psnrs.mean()}')
    print(psnrs)
    print('==================================')
    print(f'Mean SSIM: {ssims.mean()}')
    print(ssims)

    ### View.     
    index = 0
    plt.subplot(1, 2, 1)
    plt.imshow(gts[index].squeeze(), cmap='hot')
    plt.title('Ground Truth')

    plt.subplot(1, 2, 2)
    plt.imshow(preds[index].squeeze(), cmap='hot')
    plt.title(f'Model Predicted Output: PSNR = {psnrs[index]:.2f}, SSIM = {ssims[index]:.2f}')  
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    start = time.time()

    ### Set device. 
    device = torch.device('cuda:{}'.format(args.gpu))

    ### Load system matrices. 
    # u, s, v = load_mtx(args.sysmat_dir_path, device, k=8000)
    A = load_full_A(args.sysmat_dir_path, device)

    end = time.time()
    print(f'Time to load system matrix: {end - start}')

    # Run.  
    main()