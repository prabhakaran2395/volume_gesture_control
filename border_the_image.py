import cv2 as cv
import pandas as pd
import numpy as np

img = cv.imread(r'D:\computer_vision\cv_projects\hand_tracking\resized_nearest.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_out = cv.copyMakeBorder(
                 img, 
                 20, 
                 20, 
                 20, 
                 20, 
                 cv.BORDER_CONSTANT, 
                 value=(0,0,0)
              )
img_reshaped = img_out.reshape(img_out.shape[0]*img_out.shape[1],3)
df = pd.DataFrame(img_reshaped, columns = ['c1', 'c2', 'c3'])
df.to_csv('border_image_output.csv', index=False)
