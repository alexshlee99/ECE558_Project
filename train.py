import os
import wandb
import random
import numpy as np
import argparse
import mat73
from tqdm import tqdm
import torch
import torch.utils.data as data

from dataset import RetinaSimDataset
from modl import MoDL_PAI
from torch.nn import MSELoss
from utils import abs_normalizer, normalizer

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
import time


parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='checkpoint/', help='path to save checkpoints')
parser.add_argument('--dataset', default='Retina', type=str, help="dataset used for training")

parser.add_argument('--sysmat_dir_path', type=str, default=R"C:\Users\Alex\Desktop\Courses\ECE 558\Final Project\sys_mat", help='path to system matrices')
parser.add_argument('--sim_dir_path', type=str, default=R"C:\Users\Alex\Desktop\Courses\ECE 558\Final Project\data", help='path to dataset')
parser.add_argument('--map_dir_path', type=str, default=R"C:\Users\Alex\Desktop\Courses\ECE 558\Final Project\pix_map", help='path to dataset')

# parser.add_argument('--sysmat_dir_path', type=str, default=R"C:\Users\Alex\Desktop\PAI_matlab\NePAF\sys_mat\sim\retina\(500x500)_128ele_80bl", help='path to system matrices')
# parser.add_argument('--sim_dir_path', type=str, default=R"C:\Users\Alex\Desktop\PAI_matlab\NePAF\dataset\sim\retina\(500x500)_128ele_80bl", help='path to dataset')

parser.add_argument('--num_layers', type=int, default=5, help='number of layers')
parser.add_argument('--num_iters', type=int, default=10, help='number of iterations')

parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--gpu', type=int, default=0, help='the number of the device')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--learning_rate', default=0.0001, type=float, metavar='LR', help='initial learning rate (default: 0.0005)')
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

def load_mtx(mat_dir, device, k=8000): 
    """
    Function to load system matrices as Torch tensors for 0 angle. 
    """
    ### 0 degrees.
    deg0_path = os.path.join(mat_dir, "0", f"{k}")
    data_dict = mat73.loadmat(os.path.join(deg0_path, "U.mat"))
    u_0 = torch.from_numpy(data_dict['U_prime_single']).to(device)

    data_dict = mat73.loadmat(os.path.join(deg0_path, "S.mat"))
    s_0 = torch.from_numpy(data_dict['S_prime_single']).to(device)  

    data_dict = mat73.loadmat(os.path.join(deg0_path, "V.mat"))
    v_0 = torch.from_numpy(data_dict['V_prime_single']).to(device)
    return u_0, s_0, v_0

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


def train(model, train_loader, criterion, optimizer, scheduler, device):
    """
    Main code for training. 
    """
    # Initialize variables to store loss & accuracy metrics. 
    running_loss = 0.0

    # Train.
    model.train()  # change mode. 
    for _, (sds, masks, gts) in enumerate(tqdm(train_loader)):
        
        ### Batch size. 
        bs, _, _ = sds.shape

        ### Send to device. 
        sds = sds.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)
        gts = gts.to(device, dtype=torch.float).unsqueeze(1)

        ### Compute x0 (with mask).
        sds_masked = sds * masks
        y = sds_masked.reshape(bs, -1).T
        x0 = adjoint_op(y).T.reshape(bs, 1, 128, 128)
        # x0 = adjoint_op(sds_masked.view(-1,bs)).view(bs, 128, 128).unsqueeze(1)

        ### Forward through network. 
        outputs = model(x0, forward_op, adjoint_op, masks)  

        # import matplotlib.pyplot as plt
        # x0_npy = x0.detach().cpu().numpy()
        # plt.imshow(x0_npy[0].squeeze(), cmap='hot')
        # plt.tight_layout()
        # plt.show()

        ### Compute loss with acoustic forward. 
        loss = criterion(outputs, gts)
        loss = loss / bs
        
        ### Backpropagation. 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ### Calculate and store loss & train accuracy for current iteration. 
        running_loss += loss.item()
    
    # scheduler.step()
    avg_running_loss = running_loss / float(train_loader.dataset.__len__())
    return avg_running_loss

def val(model, val_loader, criterion, device):
    with torch.no_grad():
        running_loss = 0.0
        psnr = 0.0
        ssim = 0.0
        img_cnt = 0
        
        # Initialize accuracy metrics. 
        PSNR = PeakSignalNoiseRatio(data_range=(0,1), dim=(1,2,3), reduction='sum').to(device)
        SSIM = StructuralSimilarityIndexMeasure(data_range=(0,1), reduction='sum').to(device)
        
        model.eval()
        for _, (sds, masks, gts) in enumerate(tqdm(val_loader)):
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

            ### Forward through network. 
            outputs = model(x0, forward_op, adjoint_op, masks)  
            
            ### Compute loss with acoustic forward. 
            loss = criterion(outputs, gts)
            loss = loss / bs
            
            ### Get reconstructed image. 
            # Normalize both ground truth & predicted images.
            # (We don't use sigmoid since the signals are generated from the actual pressure!)
            norm_gt = abs_normalizer(gts)
            norm_pred = abs_normalizer(outputs)
            
            ### Calculate accuracies. 
            running_loss += loss.item()
            img_cnt += bs
            psnr += PSNR(norm_pred, norm_gt).item()
            ssim += SSIM(norm_pred, norm_gt).item()
        
        # Calculate test accuracy and loss. 
        avg_running_loss = running_loss / float(val_loader.dataset.__len__())
        avg_psnr = psnr / img_cnt
        avg_ssim = ssim / img_cnt
    return avg_running_loss, avg_psnr, avg_ssim

def main():
    """
    Main function for train/test. 
    """
    ### Deterministic. 
    setup_seed(0)
    
    ### Image transforms.  
    train_transforms = None  # convert into shape for resnet. 
    val_transforms = None    
    
    ### Load datasets. 
    train_dataset = RetinaSimDataset(args.sim_dir_path, phase='train', transform=train_transforms)
    val_dataset = RetinaSimDataset(args.sim_dir_path, phase='val', transform=val_transforms)
    
    train_loader = data.DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.workers,
                                        pin_memory=True)
    val_loader = data.DataLoader(val_dataset, 
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.workers,
                                        pin_memory=True)
    
    ### Load model. 
    model = MoDL_PAI(args.num_layers, args.num_iters)  # num of channels = num of transducer elements
    model.to(device)
    
    ### Set loss function, optimizer, and scheduler. 
    criterion = MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # beta: (0.9, 0.999), eps: 1e-8
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)  # test... 
    scheduler = None

    ### Train & Test. 
    best_val_psnr = 0.0
    best_val_ssim = 0.0
    for i in range(1, args.epochs + 1):
        # Forward. 
        train_loss = train(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_psnr, val_ssim = val(model, val_loader, criterion, device)
        
        # # Log in wandb. 
        wandb.log({"Train Loss": train_loss, 
                   "Validation Loss": val_loss, "Validation PSNR": val_psnr, "Validation SSIM": val_ssim})
        
        # Print test results. 
        print('Epoch: [{0}/{1}]\t'
                'Train Loss: {Train_Loss}\t'
                "|  "
                'Val Loss: {Val_Loss}\t'
                'Val PSNR: {Val_PSNR}\t'
                'Val SSIM: {Val_SSIM}\t'
                .format(i, args.epochs, 
                        Train_Loss = np.round(train_loss, 4), 
                        Val_Loss = np.round(val_loss, 4), 
                        Val_PSNR = np.round(val_psnr, 4), 
                        Val_SSIM = np.round(val_ssim, 4),
                        ))
        
        # Save checkpoints. 
        if np.round(val_psnr, 3) > best_val_psnr:  # Best Total Acc.
            checkpoint = {
                'epoch': i,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if not os.path.exists(args.outdir):  # If directory not exists, then create. 
                os.makedirs(args.outdir)
            torch.save(checkpoint, os.path.join(args.outdir, 'best_psnr.pth'))
            # Update. 
            best_val_psnr = val_psnr
        
        # Save checkpoints. 
        if np.round(val_ssim, 3) > best_val_ssim:  # Best Total Acc.
            checkpoint = {
                'epoch': i,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if not os.path.exists(args.outdir):  # If directory not exists, then create. 
                os.makedirs(args.outdir)
            torch.save(checkpoint, os.path.join(args.outdir, 'best_ssim.pth'))
            # Update. 
            best_val_ssim = val_ssim

    ### Print. 
    print(f"Best PSNR: {best_val_psnr}")
    print(f"Best SSIM: {best_val_ssim}")


if __name__ == '__main__':

    ### Initialize wandb. 
    wandb.init(
            # Set the wandb project where this run will be logged
            project="Courses",

            # Track hyperparameters and run metadata
            config={
            "learning_rate": args.learning_rate,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size
            }
        )

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