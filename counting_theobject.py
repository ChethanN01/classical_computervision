#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly


# In[7]:


import cvlib as cv
from cvlib.object_detection import draw_bbox


# In[15]:


img= cv2.imread(r'D:\download\chessSampleData\cars2.jpg')
img1= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


bbox, label, count = cv.detect_common_objects(img)

output = draw_bbox(img, bbox, label, count)



output=cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(output)
plt.show()


# In[17]:


print(len(label))


# In[19]:


print(label)


# In[ ]:




