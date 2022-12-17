import torch
import numpy as np

def inverse_zigzag_block(input, B, vmax, hmax, C):
    h = 0
    v = 0
    vmin = 0
    hmin = 0
    output = np.zeros((B, vmax, hmax, C), dtype=int)
    i = 0
    up = 1 if hmax % 2 else 0
    while ((v < vmax) and (h < hmax)): 
        if ((h + v) % 2) == up:                 
            if (v == vmin):
                output[:, v, h, :] = input[:, i, :]        
                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        
                i = i + 1
            elif ((h == hmax -1 ) and (v < vmax)):   
                output[:, v, h, :] = input[:, i, :]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    
                output[:, v, h, :] = input[:, i, :] 
                v = v - 1
                h = h + 1
                i = i + 1

        else:                                    
            if ((v == vmax -1) and (h <= hmax -1)):       
                output[:, v, h, :] = input[:, i, :] 
                h = h + 1
                i = i + 1
        
            elif (h == hmin):                  
                output[:, v, h, :] = input[:, i, :] 
                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
                                
            elif((v < vmax -1) and (h > hmin)):     
                output[:, v, h, :] = input[:, i, :] 
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax-1) and (h == hmax-1)):          
            output[:, v, h, :] = input[:, i, :] 
            break
    return output


def zigzag_block(input):
    h = 0
    v = 0
    vmin = 0
    hmin = 0
    vmax = input.shape[1]
    hmax = input.shape[2]
    B = input.shape[0]
    C = input.shape[3]
    i = 0
    output = np.zeros((B, vmax * hmax, C), dtype=int)
    up = 1 if hmax % 2 else 0

    while ((v < vmax) and (h < hmax)):
        if ((h + v) % 2) == up:                 
            if (v == vmin):
                output[:, i, :] = input[:, v, h, :]         
                if (h == hmax): 
                    v = v + 1   
                else:
                    h = h + 1   
                i = i + 1
            elif ((h == hmax -1 ) and (v < vmax)):   
                output[:, i, :] = input[:, v, h, :]  
                v = v + 1
                i = i + 1
            elif ((v > vmin) and (h < hmax -1 )):    
                output[:, i, :] = input[:, v, h, :]  
                v = v - 1
                h = h + 1
                i = i + 1
        else:                                    
            if ((v == vmax -1) and (h <= hmax -1)):       
                output[:, i, :] = input[:, v, h, :]  
                h = h + 1
                i = i + 1
            elif (h == hmin):                  
                output[:, i, :] = input[:, v, h, :]  
                if (v == vmax -1):
                    h = h + 1   
                else:
                    v = v + 1   
                i = i + 1
            elif ((v < vmax -1) and (h > hmin)):     
                output[:, i, :] = input[:, v, h, :]  
                v = v + 1
                h = h - 1
                i = i + 1
        if ((v == vmax-1) and (h == hmax-1)):          
            output[:, i, :] = input[:, v, h, :]  
            break
    return output

def zigzag(input):
    h = 0
    v = 0
    vmin = 0
    hmin = 0
    vmax = input.shape[2]
    hmax = input.shape[3]
    B = input.shape[0]
    C = input.shape[1]
    i = 0
    output = np.zeros((B, C, vmax * hmax), dtype=int)
    up = 1 if hmax % 2 else 0

    while ((v < vmax) and (h < hmax)):
        if ((h + v) % 2) == up:                 
            if (v == vmin):
                output[:, :, i] = input[:, :, v, h]         
                if (h == hmax): 
                    v = v + 1   
                else:
                    h = h + 1   
                i = i + 1
            elif ((h == hmax -1 ) and (v < vmax)):   
                output[:, :, i] = input[:, :, v, h]  
                v = v + 1
                i = i + 1
            elif ((v > vmin) and (h < hmax -1 )):    
                output[:, :, i] = input[:, :, v, h]  
                v = v - 1
                h = h + 1
                i = i + 1
        else:                                    
            if ((v == vmax -1) and (h <= hmax -1)):       
                output[:, :, i] = input[:, :, v, h]  
                h = h + 1
                i = i + 1
            elif (h == hmin):                  
                output[:, :, i] = input[:, :, v, h]  
                if (v == vmax -1):
                    h = h + 1   
                else:
                    v = v + 1   
                i = i + 1
            elif ((v < vmax -1) and (h > hmin)):     
                output[:, :, i] = input[:, :, v, h]  
                v = v + 1
                h = h - 1
                i = i + 1
        if ((v == vmax-1) and (h == hmax-1)):          
            output[:, :, i] = input[:, :, v, h]  
            break
    return output

def inverse_zigzag(input, B, C, vmax, hmax):
    h = 0
    v = 0
    vmin = 0
    hmin = 0
    output = np.zeros((B, C, vmax, hmax), dtype=int)
    i = 0
    up = 1 if hmax % 2 else 0
    while ((v < vmax) and (h < hmax)): 
        if ((h + v) % 2) == up:                 
            if (v == vmin):
                output[:, :, v, h] = input[:, :, i]        
                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        
                i = i + 1
            elif ((h == hmax -1 ) and (v < vmax)):   
                output[:, :, v, h] = input[:, :, i]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    
                output[:, :, v, h] = input[:, :, i] 
                v = v - 1
                h = h + 1
                i = i + 1

        else:                                    
            if ((v == vmax -1) and (h <= hmax -1)):       
                output[:, :, v, h] = input[:, :, i] 
                h = h + 1
                i = i + 1
        
            elif (h == hmin):                  
                output[:, :, v, h] = input[:, :, i] 
                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
                                
            elif((v < vmax -1) and (h > hmin)):     
                output[:, :, v, h] = input[:, :, i] 
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax-1) and (h == hmax-1)):          
            output[:, :, v, h] = input[:, :, i] 
            break
    return output
