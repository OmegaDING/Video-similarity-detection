# @Description: 比较两个视频是否非常相似
#               非常相似定义：缩放、亮度、帧率、格式变换等造成的视频差异(旋转的效果不佳)

import cv2
import numpy as np


def pHash(img):
    """
    get image pHash value
    """

    # 缩放图片为32x32灰度图片
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)

    # 创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h,w), np.float32)
    vis0[:h,:w] = img

    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1 = vis1[:8, :8]

    # 把二维list变成一维list
    img_list = vis1.flatten().tolist()

    # 计算均值, 得到哈希值
    avg = sum(img_list) * 1. / 64
    avg_list = [0 if i < avg else 1 for i in img_list]
    # print(avg_list)
    return avg_list

def load_video(video, url):
    """
    如果没有读取，则根据视频Url下载视频
    """

    if not video:
        if url:
            return cv2.VideoCapture(url)
        else:
            raise Exception("missing video binary and url")
    else:
        return video

def hanming_dist(s1, s2):
    """
    求汉明距离
    """
    # print(s1, s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])

def compare(video1: cv2.VideoCapture = None, video2: cv2.VideoCapture =None, 
            video_url1: str = None, video_url2: str = None) -> bool:
    """
    输入比较的两个视频流/链接
    返回是否相似，帧相似度>=0.85直接返回True，否则返回False
    """

    # 下载
    video1 = load_video(video1, video_url1)
    video2 = load_video(video2, video_url2)

    # 获取较短视频的帧数
    min_frame_count = min(video1.get(cv2.CAP_PROP_FRAME_COUNT), 
                            video2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 获取视频FPS
    fps1 = video1.get(cv2.CAP_PROP_FPS)
    fps2 = video1.get(cv2.CAP_PROP_FPS)

    similar = 0
    frame_cnt = int(min_frame_count / fps1)
    # print(frame_cnt, min_frame_count, fps1, fps2)

    # total_dist = 0

    # 截帧
    for _ in range(frame_cnt):
        for _ in range(int(fps1)): # 按视频一间隔1s
            retval1 = video1.grab()
            retval2 = video2.grab()
            # print(retval1, retval2)
            
        if not retval1 or not retval2:
            grab_failure_cnt += 1
            if grab_failure_cnt >= 10:
                raise Exception('Grab failed too much >= {} times, could be endless loop.'.format(10))
        else:
            grab_failure_cnt = 0
        
        flag1, frame1 = video1.retrieve()
        flag2, frame2 = video2.retrieve()
        
        # print(flag1, flag2)
        # 提phash特征
        if flag1 & flag2:
            phash1 = pHash(frame1)
            phash2 = pHash(frame2)
        
            # 比较汉明距离
            if hanming_dist(phash1, phash2) < 25:
                similar += 1
            
            # total_dist += hanming_dist(phash1, phash2)
            # print(hanming_dist(phash1, phash2))

    # print("similar:", similar/min_frame_count, total_dist / min_frame_count)
    return similar / frame_cnt >= 0.85

if __name__ == '__main__':

    video1 = cv2.VideoCapture("a.mp4")
    video2 = cv2.VideoCapture("b.mp4")

    print(compare(video1, video2))

