"""
Appendix E

Implementing architecture in Python

"""
# s_x is complex-valued OFDM symbol: shape (N,)
x_real = s_x.real
x_imag = s_x.imag
x_input = np.stack([x_real, x_imag], axis=0)  # shape (2, N)
x_input = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)# shape (1, 2, N ) 
def loss_function(s_hat, s_orig, lambda1=1.0, lambda2=1.0):
# Reconstruct complex signals
s_hat_complex = s_hat[:, 0, :] + 1j * s_hat[:, 1, :]
s_orig_complex = s_orig[:, 0, :] + 1j * s_orig[:, 1, :]
# Power calculations
power_avg = torch.mean(torch.abs(s_hat_complex)**2)
power_peak = torch.max(torch.abs(s_hat_complex)**2)# PAPR loss))
papr_loss = power_peak /power_avg
# Reconstruction loss (L1)
recon_loss = torch.mean(torch.abs(s_hat - s_orig
return lambda1 * papr_loss + lambda2 * recon_loss
loss.backward()
optimizer.step()
model = OFDM_Autoencoder(N=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(num_epochs):
optimizer.zero_grad()
output = model(x_input)
loss = loss_function(output, x_input)
