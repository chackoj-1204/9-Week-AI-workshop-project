import pickle
import cv2
import numpy as np
#from sklearn.preprocessing import StandardScaler
#
# images = []
#
# vidcap = cv2.VideoCapture("President Biden gives major voting rights speech in Philadelphia — 7 13 21.mp4")
# def getFrame(sec):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
#     hasFrames,image = vidcap.read()
#     if hasFrames:
#         images.append(image)
#     return hasFrames
# sec = 0
# frameRate = 0.5 #//it will capture image in each 0.5 second
# count=1
# success = getFrame(sec)
# while success:
#     count = count + 1
#     sec = sec + frameRate
#     sec = round(sec, 2)
#     success = getFrame(sec)
#
# all_frames = np.array(images)

# print(images)
# pickle.dump( all_frames, open( "bidenpics.p", "wb" ) )
# pickle.dump( frames[:2200,:,280:1000,:] , open( "bidenpicssquare.p", "wb" ) )

# frames = pickle.load( open( "bidenpicssquare.p", "rb" ) )
# images = []
# for i in frames:
#     images.append((cv2.resize(i, dsize=(256, 256))-127.5)/127.5)
# images = np.array(images)
# print(images.shape)
# pickle.dump( images , open( "bidenpicsfinal.p", "wb" ) )


# frames = pickle.load( open( "bidenpicsfinal.p", "rb" ) )
# cv2.imshow('sample image', frames[420])
# cv2.waitKey(0)  # waits until a key is pressed
# cv2.destroyAllWindows()
#
# images = []
#
# vidcap = cv2.VideoCapture("mydata.mp4")
# def getFrame(sec):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
#     hasFrames,image = vidcap.read()
#     if hasFrames:
#         images.append(image)
#     return hasFrames
# sec = 0
# frameRate = 0.5 #//it will capture image in each 0.5 second
# count=1
# success = getFrame(sec)
# while success:
#     count = count + 1
#     sec = sec + frameRate
#     sec = round(sec, 2)
#     success = getFrame(sec)
#
# all_frames = np.array(images)
#
# images = []
# for i in all_frames:
#     images.append((cv2.resize(i, dsize=(256, 256))-127.5)/127.5)
# images = np.array(images)
# print(images.shape)
# pickle.dump( images , open( "mypicsfinal.p", "wb" ) )

frames = pickle.load( open( "mypicsfinal.p", "rb" ) )
cv2.imshow('sample image', frames[100])
cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()