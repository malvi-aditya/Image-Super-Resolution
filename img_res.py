#Import necessary packages
import matplotlib.pyplot as plt, cv2, numpy as np, math, os
from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.optimizers import SGD, Adam
from skimage.measure import compare_ssim as ssim
 

#Function for Peak Signal-to-Noise Ratio (PSNR)
def psnr(target, ref):
    
    # assume RGB/BGR images
    
    target_data = target.astype(float)
    ref_data = ref.astype(float)
    
    # difference between images
    diff = ref_data - target_data
    diff.flatten('C')
    
    # root mean square error
    rmse = math.sqrt(np.mean(diff ** 2.))
    
    # Formula for PSNR
    return 20 * math.log10(255. / rmse)

# Function for Mean Squared error
def mse(target, ref):
    
    err = np.sum((target.astype('float') - ref.astype('float')) ** 2)
    # target.shape[0] * target.shape[1] == total no. of pixel in image
    err /= float(target.shape[0] * target.shape[1])
    
    return err


#Function that combines all three image quality metrics
def compare_images(target, ref):
    
    values = []
    values.append(psnr(target, ref))
    values.append(mse(target, ref))
    values.append(ssim(target, ref, multichannel = True)) # To handle RGB/BGR images multichannel==True
    
    return values
    
    
# Prepare degrade images by introducing quality distortions via resizing
    
def prepare_images(path, factor):
    
    #loop through the files in the directory
    for file in os.listdir(path):
        
        # open the file
        img = cv2.imread(path + '/' + file)
        
        #find old and new image dimensions- height, width, channels
        h, w, c = img.shape
        new_height = h//factor
        new_width = w//factor
        
        #resize the image - down
        img = cv2.resize(img , (new_width, new_height), interpolation = cv2.INTER_LINEAR)
        
        #resize the image - up
        img = cv2.resize(img , (w, h), interpolation = cv2.INTER_LINEAR)
        
        # save the image
        print('Saving {}'.format(file))
        cv2.imwrite('images/{}'.format(file), img)
        
#Prepare the images
prepare_images('source/', 2)

# testing the generated images using the image quality metrics

for file in os.listdir('images/'):
    
    #open target and reference images
    target = cv2.imread('images/{}'.format(file))
    ref = cv2.imread('source/{}'.format(file))
    
    #Calculate the scores
    scores = compare_images(target, ref)
    
    #Print scores
    print('{}\nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(file, scores[0], scores[1], scores[2]))
    
        
        
    