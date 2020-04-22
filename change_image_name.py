import cv2, os

path = "Images/"

imgs = sorted(os.listdir(path))

for i in range(len(imgs)):
    img = cv2.imread((path+imgs[i]))
    cv2.imwrite((path+imgs[i][:-4]+'_'+str(i+1)+'.jpg'), img)