import sys
import os
import matplotlib.pyplot as plt

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

## methods for creating plots:

def plot_model_layer(models, direction, layer, con = 'first'):
    if con == 'first':
        conv_number = 0
    else:
        conv_number = 3

    path = 'D:\Git\dip_svd\SingularValues\SVD_'
    plt.figure()
    fig, ax = plt.subplots()
    for i in range(4):   
        if direction=='down':
            S = get_singular_value_vector(models[i].down[layer].conv[conv_number])
        else:
            S = get_singular_value_vector(models[i].up[layer].conv[conv_number+1])
        S = abs(S)
        plt.plot(S)
    ax.set_xlim(0, 128)
    ax.set_ylim(-0.01, 1)    
    ax.set_title('singular values of '+direction+' layer '+str(layer)+", "+con+' convolution')
    ax.set_xlabel("index")
    ax.set_ylabel("singular value magnitude")
    plt.legend(["after DIP", "after pretraining", "after EDIP", "after DIP-SVD"])
    plt.savefig(path+direction+str(layer)+'_'+con+'_conv''.png')

def get_singular_value_vector(conv):
    S = conv.vector_S
    S = S[0,:,0,0]
    S = S.to('cpu')
    S = S.detach().numpy()
    return S
    
def plot_model_sv(model):
    path = 'D:\Git\dip_svd\SingularValues\SVD_'

    down = model.down
    counter = 0
    for block in down:
        plt.figure()
        plt.subplot()
        S = get_singular_value_vector(block.conv[0])
        plt.plot(S)
        plt.subplot()
        S = get_singular_value_vector(block.conv[3])
        plt.plot(S)
        plt.savefig(path+'down'+str(counter)+'.png')


        counter = counter + 1
    up = model.up
    counter = 0
    for block in up:
        plt.figure()
        plt.subplot()
        S = get_singular_value_vector(block.conv[1])
        plt.plot(S)
        plt.subplot()
        S = get_singular_value_vector(block.conv[4])
        plt.plot(S)
        plt.savefig(path+'up'+str(counter)+'.png')


        counter = counter + 1

