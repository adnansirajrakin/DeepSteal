import torch


x = torch.randn(10)*10
y=  torch.randn(10)*10
x= x.cuda()
y= y.cuda()
def clipping_info(training,target,indexes,bits=7):
    for c in range(2):
        #print(c)    
        if training != []:

            if c ==0:
                x = training[target>=0].abs()
                y = target [target>=0]
            else:
                x = training[target<0].abs()
                y = target [target<0].abs()
            if x.numel() != 0:
                intervals=1

                for k in range(bits-1):
                    intervals = intervals/2

            #print(intervals)

                inter_values = torch.zeros([int(1/intervals)]).cuda()
   
                for i in range(1,int(1/intervals)):
                    inter_values [i] = inter_values[i-1] + intervals

                inter_range_x = inter_values * x.max()
                inter_range_y = inter_values * y.max()
    
            #print(x)
            #print(y)

                for j in range(int(1/intervals)):
        
                    if j == 0:
                    #print(x[y<=inter_range_y[j+1]])
                        x[y<=inter_range_y[j+1]] = torch.clamp(x[y<=inter_range_y[j+1]], min=x.min(), max=inter_range_x[j+1])

                    elif j == int(1/intervals)-1:
                    #print(x[y>inter_range_y[j]])
                        x[y>inter_range_y[j]] = torch.clamp(x[y>inter_range_y[j]], min = inter_range_x[j], max = x.max())

                    else:          
                    #print(x[y>inter_range_y[j]][x[y>inter_range_y[j]]<inter_range_y[j+1]])
                        x[y>inter_range_y[j]][x[y>inter_range_y[j]]<inter_range_y[j+1]] = torch.clamp(x[y>inter_range_y[j]][x[y>inter_range_y[j]]<inter_range_y[j+1]], min= inter_range_x[j], max=inter_range_x[j+1])
                if c ==0:
                    training[target>=0]  = x
                else:
                    training[target<0]  = x * (-1)
        #print(training)
        #print(target)
        #print(inter_range_x,inter_range_y)
    return training

    


def loss_cal(training,indexes,bits=7):
    loss = 0
    for c in range(2):
        #print(c)    
        if training != []:

            if c ==0:
                x = training[training>=0].abs()
               
            else:
                x = training[training<0].abs()
                
            if x.numel() != 0:
                intervals=1

                for k in range(bits-1):
                    intervals = intervals/2

            #print(intervals)

                inter_values = torch.zeros([int(1/intervals)]).cuda()
   
                for i in range(1,int(1/intervals)):
                    inter_values [i] = inter_values[i-1] + intervals

                inter_range_x = inter_values * x.max()
                inter_range_y = inter_values * x.max()

                #print(inter_range_x)    
            #print(x)
            #print(y)

                for j in range(int(1/intervals)):
        
                    if j == 0:
                    #print(x[y<=inter_range_y[j+1]])
                        means = (x.min() + inter_range_x[j+1])/2
                        loss += (x[x<=inter_range_y[j+1]] - means).abs().sum()

                    elif j == int(1/intervals)-1:
                    #print(x[y>inter_range_y[j]])
                        means = (inter_range_x[j] + x.max())/2
                        loss += (x[x>inter_range_y[j]] - means).abs().sum()

                    else:          
                    #print(x[y>inter_range_y[j]][x[y>inter_range_y[j]]<inter_range_y[j+1]])
                        means = (inter_range_x[j] + inter_range_x[j+1])/2
                        loss += (x[x>inter_range_y[j]][x[x>inter_range_y[j]]<inter_range_y[j+1]] - means).abs().sum()
                 
    return loss

print(x,y)
indexes = torch.tensor([0,1,2,3,4,5,6,7,8,9])

loss = loss_cal(x,indexes,bits=2)
print(loss)
x[indexes] = clipping_info(x[indexes],y[indexes],indexes,2) 
print(x)

