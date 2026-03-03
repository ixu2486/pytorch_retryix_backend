import torch
import pytorch_retryix_backend as rxb

rxb.init()
torch.set_default_device('retryix')
model=torch.nn.Linear(128,64).to('retryix')
opt=torch.optim.SGD(model.parameters(),lr=0.01)
data=torch.randn(32,128,device='retryix')
target=torch.randn(32,64,device='retryix')
for i in range(5):
    out=model(data)
    loss=((out-target)**2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print('step',i,'loss',loss.item())

print('training snippet done')
