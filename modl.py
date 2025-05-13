import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
#  CNN Denoiser
# -----------------------------------------------------------------------------
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class cnn_denoiser(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        layers = []
        layers += conv_block(1, 64)

        for _ in range(n_layers-2):
            layers += conv_block(64, 64)

        layers += nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1),
            nn.BatchNorm2d(1)
        )

        self.nw = nn.Sequential(*layers)
    
    def forward(self, x):
        idt = x # (1, nrow, ncol)
        dw = self.nw(x) + idt # (1, nrow, ncol)
        return dw


# -----------------------------------------------------------------------------
#  Linear operator (P^T M P + λ I)
# -----------------------------------------------------------------------------
class myAtA(nn.Module):
    def __init__(self, forward_op, adjoint_op, mask, lam):
        """
        forward_op: callable x->[B,128,L]
        adjoint_op: callable y->[B,1,H,W]
        lam:        scalar λ (float or 0-D tensor)
        mask:       [128] float mask
        """
        super().__init__()
        self.P = forward_op
        self.Pt = adjoint_op
        self.lam = lam
        self.mask = mask

    def forward(self, x):
        # x: [B,H,W]

        # Reshape image into column vectors. 
        B,_,H,W = x.shape
        x_vec = x.reshape(B, -1).T

        # Convert into sensor data (forward). 
        y = self.P(x_vec)           # [M, B]

        # Mask unwanted sensors. 
        y_mask = y.T.reshape(B, 128, 500) * self.mask
        
        # Convert into reconstructed image (adjoint). 
        x_back = self.Pt(y_mask.reshape(B, -1).T)     # [N, B]
        return x_back.T.reshape(B, 1, H, W) + self.lam * x  # + λ I · x


# -----------------------------------------------------------------------------
#  Conjugate Gradient solver for real tensors
# -----------------------------------------------------------------------------
def myCG(AtA, rhs, max_iters=10, tol=1e-10):
    """
    Solves (AtA) x = rhs via CG, where AtA is a module.
    rhs, x have shape [B,1,H,W].
    """
    x   = torch.zeros_like(rhs)
    r   = rhs.clone()         # initial residual (AtA(0)=0)
    p   = r.clone()
    
    # Flatten spatial dims, keep batch
    def dot(a, b):
        return (a.mul(b)).view(a.size(0), -1).sum(dim=1, keepdim=True)  # [B,1]
    
    rTr = dot(r, r)
    
    for _ in range(max_iters):
        if torch.all(rTr < tol):
            break
        Ap    = AtA(p)
        pAp   = dot(p, Ap)
        alpha = rTr / (pAp + 1e-16)
        x     = x + alpha.view(-1, 1, 1, 1) * p
        r     = r - alpha.view(-1, 1, 1, 1) * Ap
        rTr_new = dot(r,r)
        beta    = rTr_new / (rTr + 1e-16)
        p       = r + beta.view(-1, 1, 1, 1) * p
        rTr     = rTr_new
    return x
    

# -----------------------------------------------------------------------------
#  Data‐consistency layer
# -----------------------------------------------------------------------------
class data_consistency(nn.Module):
    def __init__(self):
        super().__init__()
        # learnable λ
        self.lam = nn.Parameter(torch.tensor(0.05),
                                requires_grad=True)

    def forward(self, z_k, x0, P, Pt, mask):
        """
        z_k:   [B,1,H,W]  denoiser output
        x0:    [B,1,H,W]  backproj of masked data
        P, Pt: as above
        mask:  [128]      0/1 detector mask
        """
        # build right-hand side
        rhs = x0 + self.lam * z_k   # [B, 1, H, W] 
        # build linear operator with this λ & mask
        AtAop = myAtA(P, Pt, mask, self.lam)
        # solve for x
        return myCG(AtAop, rhs)


# -----------------------------------------------------------------------------
#  Full MoDL for PAI (single‐channel)
# -----------------------------------------------------------------------------
class MoDL_PAI(nn.Module):
    def __init__(self, n_layers, k_iters):
        """
        n_layers: number of conv layers in denoiser
        k_iters:  number of unrolled CG+denoise steps
        """
        super().__init__()
        self.k_iters = k_iters
        self.dw = cnn_denoiser(n_layers)
        self.dc = data_consistency()
    
    def forward(self, x0, P, Pt, mask):
        """
        x0:   [B,1,H,W]     initial backproject
        P:    callable      forward Op
        Pt:   callable      adjoint Op
        mask: [B,128,1]         detector mask
        """
        xk = x0.clone()
        for _ in range(self.k_iters):
            zk = self.dw(xk)                # denoiser
            xk = self.dc(zk, x0, P, Pt, mask)  # DC‐CG step
        return xk

