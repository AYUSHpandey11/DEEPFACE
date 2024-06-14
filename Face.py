#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


from deepface import DeepFace 


# In[7]:


img = cv2.imread('abcd.jpeg')


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


predictions = DeepFace.analyze(img)


# In[10]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[13]:


predictions


# In[14]:


predictions['dominant_emotion']


# In[37]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[22]:


img = cv2.imread('Sad.jpeg')


# In[23]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[24]:


predictions = DeepFace.analyze(img)


# In[25]:


predictions


# In[26]:


img = cv2.imread('Disgust.png')


# In[27]:


plt.imshow(img)


# In[28]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[29]:


predictions = DeepFace.analyze(img)


# In[32]:


predictions


# In[38]:


img = cv2.imread('Surprise.jpeg')


# In[39]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[40]:


predictions = DeepFace.analyze(img)

