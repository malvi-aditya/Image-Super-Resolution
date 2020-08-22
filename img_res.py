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
#prepare_images('source/', 2)

# testing the generated images using the image quality metrics

for file in os.listdir('images/'):
    
    #open target and reference images
    target = cv2.imread('images/{}'.format(file))
    ref = cv2.imread('source/{}'.format(file))
    
    #Calculate the scores
    scores = compare_images(target, ref)
    
    #Print scores
    print('{}\nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(file, scores[0], scores[1], scores[2]))
    

    
# Define the model
    
def model():
    
    SRCNN = Sequential()
    
    # add layers
    SRCNN.add(Conv2D(filters=128, kernel_size=(9,9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size=(5,5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    # Optimizer
    adam = Adam(lr=0.0003)
    
    # Compile the model
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    
    return SRCNN

    
#define necessary image classification functions

def modcrop(img, scale):
    
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale) #modulus operator
    
    #this is ensuring that our dimensions of image are going to be divisible by a certain scale
    
    img = img[ 0:sz[0] , 1:sz[1] ]
    return img

def shave(image, border):
    img = image[border: -border, border: -border]
    return img



#define main prediction function

def predict(image_path):
    
    #load the SRCNN model with weights
    srcnn = model()
    srcnn.load_weights('training_weights.h5')
        
    #load the degraded and reference images
    path, file = os.path.split(image_path)
    degraded = cv2.imread(image_path)
    ref = cv2.imread('source/{}'.format(file))
    
    #preprocess the images
    ref = modcrop(ref, 3)
    degraded = modcrop(degraded, 3)
    
    #convert the image to YCrCb - (as srcnn's were trained on Y channel)
    temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)
    
    #create image slice and normalize ( model input shape takes 3 dimensional inputs of  1 channel)
    Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255
    
    # perform super-resolution with srcnn
    pre = srcnn.predict(Y, batch_size = 1)
    
    # post-process output
    pre *= 255
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    
    # copy Y channel back to image and convert to BGR
    temp = shave(temp, 6)
    temp[:, :, 0] = pre[0, :, :, 0]
    output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)
    
    # remove border from reference and degraged image
    ref = shave(ref.astype(np.uint8), 6)
    degraded = shave(degraded.astype(np.uint8), 6)
    
    # image quality calculations
    scores = []
    scores.append(compare_images(degraded, ref))
    scores.append(compare_images(output, ref))
    
    # return images and scores
    return ref, degraded, output, scores

#run on one image 

ref, degraded, output, scores = predict('images/flowers.bmp')

print()
print(100*"*")
print()
print("Test on Flowers Image: ")
print()

# print all scores for all images
print('Degraded Image: \nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(scores[0][0], scores[0][1], scores[0][2]))
print('Reconstructed Image: \nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(scores[1][0], scores[1][1], scores[1][2]))


# display images as subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 8))
#imshow supports RGB format and cv2 supports BGR so conversion needed
axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original')
axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
axs[1].set_title('Degraded')
axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
axs[2].set_title('SRCNN')

# remove the x and y ticks
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

#show the plot
plt.show()    


#run on all the images

print()
print(100*"*")
print()
print("Running and saving all images: ")
print()
    
for file in os.listdir('images'):
    
    # perform super-resolution
    
    ref, degraded, output, scores = predict('images/{}'.format(file))
    
    # display images as subplots
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Degraded')
    axs[1].set(xlabel = 'PSNR: {}\nMSE: {} \nSSIM: {}'.format(scores[0][0], scores[0][1], scores[0][2]))
    axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axs[2].set_title('SRCNN')
    axs[2].set(xlabel = 'PSNR: {} \nMSE: {} \nSSIM: {}'.format(scores[1][0], scores[1][1], scores[1][2]))

    # remove the x and y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
      
    print('Saving {}'.format(file))
    fig.savefig('output/{}.png'.format(os.path.splitext(file)[0])) 
    plt.close()
        
    