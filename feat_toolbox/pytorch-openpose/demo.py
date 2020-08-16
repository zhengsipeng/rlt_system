import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import sys
from src import model
from src import util
from src.body import Body
from src.hand import Hand


# the 21 keypoints hand index
# please see examples in README.md

# the 18 keypoints body index
# please see examples in README.md

def detect_keypoint(test_image, is_vis):
    body_estimation = Body('model/body_pose_model.pth')
    hand_estimation = Hand('model/hand_pose_model.pth')

    oriImg = cv2.imread(test_image)  # B,G,R order
    
    # detect body 
    # subset: n*20 array, n is the human_number in the index, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
    # candidate: m*4, m is the keypoint number in the image, [x, y, confidence, id]
    candidate, subset = body_estimation(oriImg)  # candidate: output the keypoints([25, 4]),  x, y, score, keypoint_index
    
    canvas = copy.deepcopy(oriImg)
    canvas, bodypoints = util.draw_bodypose(canvas, candidate, subset)
    
    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)
    all_hand_peaks = []
    hand_personid_isleft = []
    for x, y, w, is_left, person_id in hands_list:
        # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # if is_left:
            # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
            # plt.show()
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        # else:
        #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
        #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
        #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        #     print(peaks)
        all_hand_peaks.append(peaks)
        hand_personid_isleft.append([person_id, is_left])

    # all_hand_peaks: [p, 21, 2] p is the hand number in the image
    # hand_personid_isleft: [p, 2]  is_isleft, person_id
    all_hand_peaks = np.asarray(all_hand_peaks)
    hand_personid_isleft = np.asarray(hand_personid_isleft)

    canvas = util.draw_handpose(canvas, all_hand_peaks)
    if is_vis:
        plt.imshow(canvas[:, :, [2, 1, 0]])
        plt.axis('off')
        plt.show()

    return bodypoints, all_hand_peaks, hand_personid_isleft


if __name__ == '__main__':
    im = sys.argv[1]  # enter the im_path
    is_vis = False
    bodypoints, all_hand_peaks, hand_personid_isleft = detect_keypoint(im, is_vis)
