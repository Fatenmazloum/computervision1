import numpy as np
import cv2 as cv
import imutils #resizing, rotating, and cropping images.

#The goal is not just to find matching features, but to align (rotate, translate, scale) one image to make it match the other, even if one of them has been rotated, translated, or scaled. 
def alignimage(image,template,maxfeatures=500,keeppercent=0.2,debug=False):
    imagegray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    templategray=cv.cvtColor(template,cv.COLOR_BGR2GRAY)
    #converting image to gray helps on detection of features
    orb=cv.ORB_create(maxfeatures)#
    #cv.ORB_create(), it's initializing the ORB feature detector(instance of ORB feauture detector)
    #After creating the ORB detector, you need to call detectAndCompute() to find the keypoints (features) and their corresponding descriptors in an image. 
    (kpA,descA)=orb.detectAndCompute(imagegray,None)
    #kpA array of keypoints each(size,(x,y),angle,..), descA=array[[1,0,1,0,1,],[1,0,0,1,]....first row is first keypoint and etc]
    (kpB,descB)=orb.detectAndCompute(templategray,None)
    #calculation of keypoints  and descriptors for each image
    #then matching using  brute force with hammingdistance method
    method=cv.DescriptorMatcher_BRUTEFORCE_HAMMING
    matcher=cv.DescriptorMatcher_create(method)
    #create instance from Descriptormatcher and give it parameter which is method iam going to use
    matches=matcher.match(descA,descB)
    #matches: [(1,3,12),(2,4,15)..]means keypoint of image A at index 1 and keypoint of image B at index 3 and distance between desc = 12
    #.match() method will compare the descriptors of image A and image B and return a list of matches. The matches list stores the results of comparing the descriptors, where each element is a cv.DMatch object that contains: Index of the keypoint in image A(inquiryidx) and B (trainidx)and hamming distance between them.
    #The matches are returned unsorted by default. Typically, you want to sort the matches based on their distance (the smaller the distance, the better the match). This will sort the list from smallest to largest distance, so the best matches (smallest distance) come first.
    matches=sorted(matches,key=lambda x:x.distance)
    keep=int(len(matches)*keeppercent)#i need 20% from list only 20 important features means first 20 features
    matches=matches[:keep]#convert to int (for index)
    ptsA=np.zeros([len(matches),2],dtype="float")# 2D array 
    ptsB=np.zeros([len(matches),2],dtype="float")
    if debug:
        matchedvis=cv.drawMatches(image,kpA,template,kpB,matches,None)
        matchedvis=imutils.resize(matchedvis,width=1000)
        cv.imshow("fin",matchedvis)
        cv.waitKey(0)
    #fill arrays
    for(i,m) in enumerate(matches):
        ptsA[i]=kpA[m.queryIdx].pt#matching kepoint for image A
        ptsB[i]=kpB[m.trainIdx].pt
        #ptsA and B stores the (x, y) coordinates of matching keypoints 
        #.pt attribute gives you the coordinates of that keypoint in the form of a tuple (x, y), where x is the horizontal coordinate (column) and y is the vertical coordinate (row).
        #queryIdx=0 means the first match comes from keypoint 0 in image A.trainIdx=5 means that this keypoint in image A (index 0) matched with keypoint 5 in image B.
        #now homogrophy matrix
    (H,mask)=cv.findHomography(ptsA,ptsB,method=cv.RANSAC)#help make alignment of image
    #the function cv2.findHomography() with the RANSAC method is an important step in aligning two images by finding a transformation matrix that maps keypoints in image A to corresponding keypoints in image B(or vice).
    #This matrix includes all the information needed to perform transformations like translation, rotation, scaling, and even perspective distortion between the two images.
    (h,w)=template.shape[:2]
    aligned=cv.warpPerspective(image,H,(w,h))
    return aligned

image=cv.imread(r"C:\Users\asus\Desktop\Computer Vision\Lab2\image.jpg")
temp=cv.imread(r"C:\Users\asus\Desktop\Computer Vision\Lab2\main.png")
aligned=alignimage(image,temp,debug=True)
cv.imshow("Aligned Image", aligned)
cv.imshow("original Image", temp)
cv.waitKey(0)
cv.destroyAllWindows()


""""
    Step-by-Step Breakdown:
1. Feature Detection:

ORB (Oriented FAST and Rotated BRIEF) starts by detecting keypoints (features) in both images. 

2. Feature Orientation:

Each keypoint in the image has an associated orientation. This is the rotation of the feature (the direction in which it's oriented).

ORB automatically calculates this orientation for each keypoint when it detects them. This is important because it helps the algorithm handle rotations of the image. Without this step, the keypoint could be considered different in two images just because one image has a rotated version of the feature.

3.Descriptor Generation:

Once the keypoints are detected and oriented, ORB generates a descriptor (a "fingerprint") for each keypoint. These descriptors are unique to the keypoint and capture information about the surrounding region, such as intensity patterns.

The descriptor is what allows the feature to be matched to a similar feature in another image.


4.Descriptor Comparison:

After the descriptors are created for both images, the next step is to compare them.

ORB uses Hamming distance to compare the binary descriptors of the keypoints. The Hamming distance is the number of bit differences between two binary descriptors. The smaller the Hamming distance, the more similar the descriptors are.

You use a Brute Force Matcher (cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)) to compare the descriptors. The matcher finds the best matches based on Hamming distance.

5. Homography Calculation:

Once you have matched the descriptors (keypoints), you can compute the homography matrix (if enough good matches are found).

The homography matrix allows you to transform one image to align with the other (in terms of rotation, translation, and scaling).

6.Final Step – Image Alignment:

Using the homography matrix, you can then align the two images. If the images are the same (but one is rotated), they should match after applying the homography transformation.
    """

    


