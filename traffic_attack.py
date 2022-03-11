# module
from __future__ import print_function
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import time
import torch.nn as nn
from SSGE import Attack,resnet18
import torch.utils.data as data
import torchvision
from attack import uap_sgd
import random
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from resnet import resnet18
from resnet2 import RESNET18
from test import clipping_info,loss_cal

parser = argparse.ArgumentParser(description='privaCY')
parser.add_argument('--epsilon', type=float, default=0.3, metavar='EPS', help='L-infinity perturbation limit for PGD attack')
parser.add_argument('--batch-size', '-b', type=int, default=16, metavar='N', help='input batch size for training (default: 500)')
parser.add_argument('--epochs', type=int, default=125, metavar='N', help='number of epochs to train (default: 20)')
parser.add_argument('--no_train', type=int, default=0, metavar='N', help='no training algorithm')
parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='learning momentum')
parser.add_argument('--percentage', type=float, default=1, help='learning momentum')
parser.add_argument('--lambdas', type=float, default=0.0001, help='learning momentum')
parser.add_argument('--adv_model', default='./results/baseline_MNIST_classifier.pt', metavar='FILE', help='location of PGD trained classifier')
parser.add_argument('--layer', type=int, default=6, metavar='N', help='Layer Number')
parser.add_argument('--evaluate', type=int, default=1, help='set to 1 to evaluate our trained adversary model in adv_model2/set to 0 to train a model with our method +PGD/else trains with our adversary only')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
print(args)


use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")



## normalize layer
class Normalize_layer(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        
        return input.sub(self.mean).div(self.std)

mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]


data_transforms = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor(),
    
])

train_data_path = "./data/GTS/Train"
train_data = torchvision.datasets.ImageFolder(root = train_data_path, transform = data_transforms)

# Divide data into training and validation (0.8 and 0.2)
ratio = 0.85
n_train_examples = int(len(train_data) * ratio)
n_val_examples = len(train_data) - n_train_examples

train_data, val_data = data.random_split(train_data, [n_train_examples, n_val_examples])

print(f"Number of training samples = {len(train_data)}")
print(f"Number of validation samples = {len(val_data)}")

train_loader = data.DataLoader(train_data, shuffle=False, batch_size = args.batch_size)
test_loader = data.DataLoader(val_data, shuffle=False, batch_size = args.batch_size)

attacker = Attack(dataloader=None,
                        attack_method='pgd', epsilon=args.epsilon)

def lasso_var(var,var1):
   
    return (var1.mean() -var).abs().sum()
# Train baseline classifier on clean data
def train_baseline(classifier, adv_classifier, recordf, record,record7,record6,record5,record4,record3,record2,class_opt, device, epoch,lambdas):
    classifier.train()
    for batch_idx, (data, target) in enumerate(train_loader):
       
        if batch_idx == 166:
            break
      
        
        data, target = data.to(device), target.to(device)
        '''output = adv_classifier (data)
        pred = output.argmax(dim=1, keepdim=True)
        
        target = pred.view(-1)'''
        class_opt.zero_grad() # Update the classifier
        loss = F.cross_entropy(classifier(data), target)
        
        loss_term = 0
        cc = 0
        for name, param in classifier.named_modules():
            if isinstance(param, nn.Linear) or isinstance(param, nn.Conv2d) :
                cc += 1
                if cc < args.layer:
                    loss_term += lambdas * (lasso_var(param.weight.view(-1)[record[cc]][param.weight.view(-1)[record[cc]]>=0],param.weight[param.weight >=0])  + lasso_var(param.weight.view(-1)[record[cc]][param.weight.view(-1)[record[cc]]<0],param.weight[param.weight < 0]))
                    loss_term += lambdas * (loss_cal(param.weight.data.view(-1)[record7[cc]],record7[cc],7))
                    loss_term += lambdas * (loss_cal(param.weight.data.view(-1)[record6[cc]],record6[cc],6))
                    loss_term += lambdas * (loss_cal(param.weight.data.view(-1)[record5[cc]],record5[cc],5))
                    loss_term += lambdas * (loss_cal(param.weight.data.view(-1)[record4[cc]],record4[cc],4))
                    loss_term += lambdas * (loss_cal(param.weight.data.view(-1)[record3[cc]],record3[cc],3))
                    loss_term += lambdas * (loss_cal(param.weight.data.view(-1)[record2[cc]],record2[cc],2))
                    done = 1
        #print(loss_term)
        loss += loss_term
        loss.backward()

        if epoch < 111:
            cc = 0
            for name, param in classifier.named_modules():
                if isinstance(param, nn.Linear) or isinstance(param, nn.Conv2d) :
                    cc += 1
                    if cc < args.layer:
                        param.weight.grad.data.view(-1) [recordf[cc]] = 0

        class_opt.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss




# Tests classifier on clean data or attacker output
def test(classifier, attacker1, device, epoch):
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
           
            data, target = data.to(device), target.to(device)
 
            output = classifier(data)
           
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
      

    test_loss /= len(test_loader.dataset)

  
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct

def functional(classifier, model, attacker1, device, epoch):
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
 
            output = classifier(data)
           
            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            output1 = model(data)
           
            
            pred1 = output1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred1= pred1.view(target.size())
            test_loss += F.cross_entropy(output, pred1, reduction='sum').item()  # sum up batch loss
            correct += pred.eq(pred1.view_as(pred)).sum().item()
      

    test_loss /= len(test_loader.dataset)

  
    print('Test set: Average loss: {:.4f}, Functional Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct

## attacking the classifier with black-box adversary generated from model.
def adv_test(classifier, model,attacker, device, epoch):
    classifier.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
            
            data, target = data.to(device), target.to(device)
            data = attacker.attack_method(
                model, data, target)
            output = classifier(data)
            
            test_loss += F.cross_entropy(output, target.cuda(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
         


    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))









import torch.nn.functional as F



#quantization function
class _Quantize(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, step):         
        ctx.step = step.item()
        output = torch.round(input/ctx.step)
        return output
                
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.step
        return grad_input, None
                
quantize1 = _Quantize.apply

class quantized_conv(nn.Conv2d):
    def __init__(self,nchin,nchout,kernel_size,stride,padding=0,bias=False):
        super().__init__(in_channels=nchin,out_channels=nchout, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)    
        
    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max()/((2**self.N_bits-1))
       
        QW = quantize1(self.weight, step)
        
        return F.conv2d(input, QW*step, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
    

        

class bilinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features)
        
    
        
        
    def forward(self, input):
       
        self.N_bits = 7
        step = self.weight.abs().max()/((2**self.N_bits-1))
        
        QW = quantize1(self.weight, step)
       
        
        return F.linear(input, QW*step, self.bias)
import copy

class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 =  nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                 nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=43):
        super(ResNet1, self).__init__()
        self.in_planes = 64

        self.conv1 =  nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(4608, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        return out


def resnet18():
    return ResNet1(BasicBlock1, [2, 2, 2, 2])


# defining CNN model
model1 = resnet18()
adv_classifier = torch.nn.Sequential(
                    Normalize_layer(mean,std),
                    model1
                    )
adv_classifier.load_state_dict(torch.load(args.adv_model))  



  

adv_classifier = adv_classifier.cuda()
print("hi")
print("Test accuracy of the model" )
corr = test(adv_classifier, attacker, device, epoch=0)




net_f = resnet18()
 
classifier2 = torch.nn.Sequential(
                    Normalize_layer(mean,std),
                    net_f
                    )
classifier2 =  classifier2.cuda()
class_adv = torch.optim.Adam(classifier2.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(class_adv, milestones=[30,60,90], gamma=0.1)
summary(classifier2, (3, 112, 112))
cc= 0 

count =0
for name, module in classifier2.named_modules():

    if isinstance(module,  nn.BatchNorm2d):
        
        count+=1
        
        module.weight.data.uniform_(0.01, 0.5)
        module.bias.data[:] = 0
         
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
       
        count+=1 
        
        module.weight.data.uniform_(-0.05, 0.05)
   

        print(cc,count)
    
        cc+=1

if args.no_train ==1:
    for name, module in classifier2.named_modules():

        if isinstance(module,  nn.BatchNorm2d):
        
            count+=1
        
            module.weight.data[:] = 0
            module.bias.data[:] = 0
         
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
       
            count+=1
            module.weight.data[:] = 0
   

        print(cc,count)
    
        cc+=1

recordr = {} ## all bits
recordf = {} ## MSB + 7
record = {} ## only MSB
recordm = {} ## MSB + any number
record7 = {} ## MSB + 6
record6 = {} ## MSB + 5
record5 = {} ## MSB + 4
record4 = {} ## MSB + 3
record3 = {} ## MSB + 2
record2 = {} ## MSB + 1

'''#perc = torch.tensor([0.5,0.055,0.056,0.055,0.067,0.077,0.078]) # layer-wise percentage
perc = torch.tensor([0.067,0.033,0.033,0.033,0.17,0.17,0.17])
#perc = torch.tensor([0.25,0.05,0.05,0.05,0.1,0.15,0.15])'''

'''
new:
90: torch.tensor([0.58,0.033,0.056,0.044,0.056,0.067,0.078])
80: torch.tensor([0.3125,0.0625,0.0625,0.0625,0.0875,0.1,0.125])
60: torch.tensor([0.133,0.033,0.033,0.05,0.067,0.12,0.2])
'''
perc = torch.tensor([0.58,0.033,0.056,0.044,0.056,0.067,0.078])

cc = 0
for name, module in adv_classifier.named_modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        cc+=1
        if cc < args.layer:
            tot = module.weight.data.view(-1).size()[0]
            p_tot = int(args.percentage*tot)
            step_f= int(p_tot*perc[0])
            step_7= int(p_tot*perc[1]) + step_f
            step_6 = int(p_tot*perc[2]) + step_7
            step_5 = int(p_tot*perc[3]) + step_6
            step_4 = int(p_tot*perc[4]) + step_5
            step_3 = int(p_tot*perc[5]) + step_4
            step_2 = int(p_tot*perc[6]) + step_3
        
            
            recordr[cc] = torch.Tensor(random.sample(range(0,tot), p_tot)).long()
            recordf[cc] = recordr[cc][0:step_f]
            recordm[cc] = recordr[cc][step_f :]
            record7[cc] = recordr[cc][step_f:step_7]
            record6[cc] = recordr[cc][step_7:step_6]
            record5[cc] = recordr[cc][step_6:step_5]
            record4[cc] = recordr[cc][step_5:step_4]
            record3[cc] = recordr[cc][step_4:step_3]
            record2[cc] = recordr[cc][step_3:step_2]
            record[cc] = recordr[cc][step_2:]
            print(recordf[cc].size()[0]/tot,recordf[cc].size()[0]/tot,record7[cc].size()[0]/tot,record6[cc].size()[0]/tot,record5[cc].size()[0]/tot,
            record4[cc].size()[0]/tot,record3[cc].size()[0]/tot,record2[cc].size()[0]/tot,record[cc].size()[0]/tot)


cc= 0 
for name, module in classifier2.named_modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        cc +=1
        print(cc)
        m=0 
        for name1, module1 in adv_classifier.named_modules():
            if isinstance(module1, nn.Linear) or isinstance(module1, nn.Conv2d):
                m+=1
                if cc==m:
                    if cc < args.layer:
                        module.weight.data.view(-1)[recordm[cc]].uniform_(0.001, 0.1)
                        module.weight.data.view(-1)[recordm[cc]] = module.weight.data.view(-1)[recordm[cc]] * module.weight.data.view(-1)[recordm[cc]].sign() * module1.weight.data.view(-1)[recordm[cc]].clone().sign()
                        module.weight.data.view(-1)[recordf[cc]] = module1.weight.data.view(-1)[recordf[cc]]
                        module.weight.data.view(-1)[record7[cc]] = clipping_info (module.weight.data.view(-1)[record7[cc]],module1.weight.data.view(-1)[record7[cc]],record7[cc],7)
                        module.weight.data.view(-1)[record6[cc]] = clipping_info (module.weight.data.view(-1)[record6[cc]],module1.weight.data.view(-1)[record6[cc]],record6[cc],6)
                        module.weight.data.view(-1)[record5[cc]] = clipping_info (module.weight.data.view(-1)[record5[cc]],module1.weight.data.view(-1)[record5[cc]],record5[cc],5)
                        module.weight.data.view(-1)[record4[cc]] = clipping_info (module.weight.data.view(-1)[record4[cc]],module1.weight.data.view(-1)[record4[cc]],record4[cc],4)
                        module.weight.data.view(-1)[record3[cc]] = clipping_info (module.weight.data.view(-1)[record3[cc]],module1.weight.data.view(-1)[record3[cc]],record3[cc],3)
                        module.weight.data.view(-1)[record2[cc]] = clipping_info (module.weight.data.view(-1)[record2[cc]],module1.weight.data.view(-1)[record2[cc]],record2[cc],2)

total = 0
for name, module in classifier2.named_modules():
    if isinstance(module, nn.Conv2d):
        ss = module.weight.data.size()
        total += ss[0]*ss[1]*ss[2]*ss[3]
        print(total)

for name, module in classifier2.named_modules():
    if isinstance(module, nn.Linear):
        ss = module.weight.data.size()
        total += ss[0]*ss[1]
        print(ss[0]*ss[1])
print(total)
corrr = test(classifier2, None, device, epoch=0)
best_acc = 0
t0 = time.time()
print("Attacking the Classifier with white-box PGD" )
adv_test(adv_classifier,adv_classifier,attacker, device, 0)


_ = functional(classifier2, adv_classifier,attacker, device, epoch=0)
best_acc = 0
t0 = time.time()
print("Attacking the Classifier with hammer leak" )
adv_test(adv_classifier,classifier2,attacker, device, 0)



count =0

losses = np.zeros([args.epochs])
if args.evaluate==0:
    print('Training both  baseline classifier classifiers')
    # Classification model setup
 
    scheduler.step()

    for epoch in range(1, args.epochs + 1):
    
        losses[epoch-1] =  train_baseline(classifier2, adv_classifier,recordf,record,record7,record6,record5,record4,record3,record2,class_adv, device, epoch,args.lambdas)
        classifier2.eval()
        if epoch == 109:
            args.lambdas = 0
        cc= 0 
        for name, module in classifier2.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                cc +=1
             
                m=0 
                for name1, module1 in adv_classifier.named_modules():
                    if isinstance(module1, nn.Linear) or isinstance(module1, nn.Conv2d):
                        m+=1
                        if cc==m:
                            print((module.weight.data.view(-1).sign() - module1.weight.data.view(-1).sign()).abs().sum())
        if (epoch+1)%5 == 0 and epoch < 111:
            cc= 0 
            for name, module in classifier2.named_modules():
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    cc +=1
                    m=0 
                    for name1, module1 in adv_classifier.named_modules():
                        if isinstance(module1, nn.Linear) or isinstance(module1, nn.Conv2d):
                            m+=1
                            if cc==m:
                                if cc<args.layer:
                                    print(cc)
                                    module.weight.data.view(-1)[record[cc]] = module.weight.data.view(-1)[record[cc]].abs() * module1.weight.data.view(-1)[record[cc]].sign()
                                    module.weight.data.view(-1)[record7[cc]] = clipping_info (module.weight.data.view(-1)[record7[cc]],module1.weight.data.view(-1)[record7[cc]],record7[cc],7)
                                    module.weight.data.view(-1)[record6[cc]] = clipping_info (module.weight.data.view(-1)[record6[cc]],module1.weight.data.view(-1)[record6[cc]],record6[cc],6)
                                    module.weight.data.view(-1)[record5[cc]] = clipping_info (module.weight.data.view(-1)[record5[cc]],module1.weight.data.view(-1)[record5[cc]],record5[cc],5)
                                    module.weight.data.view(-1)[record4[cc]] = clipping_info (module.weight.data.view(-1)[record4[cc]],module1.weight.data.view(-1)[record4[cc]],record4[cc],4)
                                    module.weight.data.view(-1)[record3[cc]] = clipping_info (module.weight.data.view(-1)[record3[cc]],module1.weight.data.view(-1)[record3[cc]],record3[cc],3)
                                    module.weight.data.view(-1)[record2[cc]] = clipping_info (module.weight.data.view(-1)[record2[cc]],module1.weight.data.view(-1)[record2[cc]],record2[cc],2)

                                    #module.weight.data.view(-1)[recordf[cc]] = module1.weight.data.view(-1)[recordf[cc]]
                                    print((module.weight.data.view(-1).sign() - module1.weight.data.view(-1).sign()).abs().sum())
                                    
                                
        accs = test(classifier2, None, device, epoch)
        if epoch == 111:
            classifier2 = torch.load('nmt.pt')
        if best_acc < accs:
            best_acc = accs
            torch.save(classifier2, 'nmt.pt')

       
classifier2 = torch.load('nmt.pt')
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss term")
plt.savefig("figure.png")
accs = test(classifier2, None, device, epoch)
_ = functional(classifier2, adv_classifier,attacker, device, epoch=0)
cc= 0 
for name, module in classifier2.named_modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        cc +=1
        print(cc)
        m=0 
        for name1, module1 in adv_classifier.named_modules():
            if isinstance(module1, nn.Linear) or isinstance(module1, nn.Conv2d):
                m+=1
                if cc==m:
                    print((module.weight.data.view(-1).sign() - module1.weight.data.view(-1).sign()).abs().sum())
       




t0 = time.time()
print("Attacking PGD trained Classifier with Black-box PGD" )
adv_test(adv_classifier,classifier2,attacker, device, 0)
torch.cuda.current_stream().synchronize()
t1= time.time()
print(" Black-PGD Attack Time:",'{} seconds'.format(t1 - t0))


