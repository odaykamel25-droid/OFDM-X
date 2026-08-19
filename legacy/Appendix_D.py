"""
Appendix D

Suggested Autoencoder Architecture

"""
import torch
import torch.nn as nn
class OFDM_Autoencoder(nn.Module):
def __init__(self, N):
super(OFDM_Autoencoder, self).__init__()
self.encoder = nn.Sequential(
nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2),
nn.ReLU(),
nn.Conv1d(16, 8, kernel_size=3, padding=1),
nn.ReLU()
import torch
import torch.nn as nn
class OFDM_Autoencoder(nn.Module):
def __init__(self, N):
super(OFDM_Autoencoder, self).__init__()
self.encoder = nn.Sequential(
nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2),
nn.ReLU(),
nn.Conv1d(16, 8, kernel_size=3, padding=1),
