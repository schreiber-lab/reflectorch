import torch
import torch.nn as nn
import torch.nn.functional as F

from reflectorch.models.activations import activation_by_name

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
    

class FnoEncoder(nn.Module):
    """An embedding network based on the Fourier Neural Operator (FNO) architecture

    .. image:: ../documentation/fig_reflectometry_embedding_networks.png
        :width: 400px
        :align: center

    Args:
        ch_in (int): number of input channels
        dim_embedding (int): dimension of the output embedding
        modes (int): number of Fourier modes
        width_fno (int): number of channels of the intermediate representations
        n_fno_blocks (int): number of FNO blocks
        activation (str): the activation function
        fusion_self_attention (bool): if ``True`` a fusion layer is used after the FNO blocks to produce the final embedding
    """
    def __init__(
            self, 
            ch_in: int = 2, 
            dim_embedding: int = 128, 
            modes: int = 32, 
            width_fno: int = 64, 
            n_fno_blocks: int = 6, 
            activation: str = 'gelu',
            fusion_self_attention: bool = False,
            ):
        super().__init__()


        self.ch_in = ch_in
        self.dim_embedding = dim_embedding
        
        self.modes = modes
        self.width_fno = width_fno
        self.n_fno_blocks = n_fno_blocks
        self.activation = activation_by_name(activation)()
        self.fusion_self_attention = fusion_self_attention
        

        self.fc0 = nn.Linear(ch_in, width_fno) #(r(q), q)
        self.spectral_convs = nn.ModuleList([SpectralConv1d(in_channels=width_fno, out_channels=width_fno, modes=modes) for _ in range(n_fno_blocks)])
        self.w_convs = nn.ModuleList([nn.Conv1d(in_channels=width_fno, out_channels=width_fno, kernel_size=1) for _ in range(n_fno_blocks)])
        self.fc_out = nn.Linear(width_fno, dim_embedding)

        if fusion_self_attention:
            self.fusion = FusionSelfAttention(width_fno, 2*width_fno)
        
    def forward(self, x):
        """"""

        x = x.permute(0, 2, 1) #(B, D, S) -> (B, S, D)
        x = self.fc0(x)
        x = x.permute(0, 2, 1) #(B, S, D) -> (B, D, S) 

        for i in range(self.n_fno_blocks):
            x1 = self.spectral_convs[i](x)
            x2 = self.w_convs[i](x)
            
            x = x1 + x2
            x = self.activation(x)

        if self.fusion_self_attention:
            x = x.permute(0, 2, 1)
            x = self.fusion(x)  
        else:
            x = x.mean(dim=-1)

        x = self.fc_out(x)
        
        return x
    
class FusionSelfAttention(nn.Module):
    def __init__(self, 
                 embed_dim: int = 64, 
                 hidden_dim: int = 64,
                 activation=nn.Tanh,
                 ):
        super().__init__()
        self.fuser = nn.Sequential(nn.Linear(embed_dim, hidden_dim), 
                                   activation(),
                                   nn.Linear(hidden_dim, 1, bias=False))
        
    def forward(self, c):  # (batch_size x seq_len x embed_dim)
        a = self.fuser(c)
        alpha = torch.exp(a)
        alpha = alpha/alpha.sum(dim=1, keepdim=True)
        return (alpha*c).sum(dim=1)  # (batch_size x embed_dim)