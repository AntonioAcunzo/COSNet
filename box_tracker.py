from __future__ import print_function

import numpy as np
import os
from glob import glob
import cv2
import matplotlib.pylab as plt
import time
from scipy.spatial import distance
from matplotlib import cm


class Track:
    def __init__(self, cur_box, start_frame, track_id):
        self.id = track_id
        self.boxes = [cur_box]
        self.start_frame = start_frame
        self.is_active = True
        self.track_length = 1
        self.color = np.random.random(3,)*255
        print('new track: ' + str(track_id) + " at frame " + str(start_frame))

    def add_box(self, cur_box):
        self.track_length += 1
        self.boxes.append(cur_box)

    def kill(self):
        self.is_active = False


class Tracker:
    def __init__(self, iou_th=0.1):
        self.tracks = []
        self.iou_mat = []
        self.iou_th = iou_th
        self.active_tracks_ids = []
        self.cur_frame = 0
        self.prev_boxes = []
        self.last_frame = None


    @staticmethod
    def iou(bboxes1, bboxes2):
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea.astype('float') / (boxAArea + np.transpose(boxBArea) - interArea)
        return iou


    @staticmethod
    def compute_optical_flow(prev_frame, next_frame, resize_factor=1, return_polar=False):
        if resize_factor != 1:
            prev_frame = cv2.resize(prev_frame, (0, 0), fx=resize_factor, fy=resize_factor)
            next_frame = cv2.resize(next_frame, (0, 0), fx=resize_factor, fy=resize_factor)
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.resize(flow, (0, 0), fx=1/resize_factor, fy=1/resize_factor)/resize_factor
        if return_polar:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            return flow, mag, ang
        else:
            return flow


    @staticmethod
    def shift_box_flow(flow, bb):
        flow_x = np.mean(flow[bb[1]:bb[3], bb[0]:bb[2], 0])
        flow_y = np.mean(flow[bb[1]:bb[3], bb[0]:bb[2], 1])
        bb = np.array(bb + [flow_x, flow_y, flow_x, flow_y]).astype(np.int32)
        return bb


    def track_boxes(self, new_boxes, new_frame=None):
        if new_frame is not None and self.last_frame is not None:
            flow = self.compute_optical_flow(self.last_frame, new_frame, resize_factor=0.2)
            self.prev_boxes = np.array([self.shift_box_flow(flow, x) for x in self.prev_boxes])
        self.last_frame = new_frame

        if len(new_boxes) == 0:
            for t in self.active_tracks_ids:
                self.tracks[t].kill()
            self.active_tracks_ids = []
            self.prev_boxes = []
            self.cur_frame += 1
            return

        if len(self.active_tracks_ids) == 0:
            self.active_tracks_ids = [x + len(self.tracks) for x in range(len(new_boxes))]
            for b in new_boxes:
                self.tracks.append(Track(cur_box=b, start_frame=self.cur_frame, track_id=len(self.tracks)))

            self.prev_boxes = new_boxes
        else:
            # Compute IoU matrix
            self.iou_mat = self.iou(self.prev_boxes, new_boxes)

            # Apply threshold to IoU matrix
            self.iou_mat[self.iou_mat < self.iou_th] = 0.0

            # Match new boxes with alive tracks
            still_alive_tracks = np.zeros((len(self.active_tracks_ids),))
            used_boxes = np.zeros((len(new_boxes),))
            while np.max(self.iou_mat) > 0:
                track_id, box_id = np.unravel_index(np.argmax(self.iou_mat), self.iou_mat.shape)
                self.iou_mat[track_id, :] = 0.0
                self.iou_mat[:, box_id] = 0.0
                self.tracks[self.active_tracks_ids[track_id]].add_box(new_boxes[box_id])
                still_alive_tracks[track_id] = 1
                used_boxes[box_id] = 1

            # Kill dead tracks
            killed_tracks = np.where(still_alive_tracks == 0)[0]
            if len(killed_tracks) > 0:
                killed_tracks_ids = [self.active_tracks_ids[x] for x in killed_tracks]
                for kt in killed_tracks_ids:
                    self.tracks[kt].kill()
                self.active_tracks_ids = [x for x in self.active_tracks_ids if x not in killed_tracks_ids]

            # Add new tracks
            new_tracks = np.where(used_boxes == 0)[0]
            if len(new_tracks) > 0:
                for nt in new_tracks:
                    new_track_id = len(self.tracks)
                    self.tracks.append(Track(cur_box=new_boxes[nt], start_frame=self.cur_frame, track_id=new_track_id))
                    self.active_tracks_ids.append(new_track_id)

            self.prev_boxes = np.array([self.tracks[x].boxes[-1] for x in self.active_tracks_ids])

        self.cur_frame += 1


    def get_boxes_in_frame(self, frame_id):
        frame_tracks = [x for x in self.tracks if x.start_frame <= frame_id < x.start_frame + x.track_length]
        all_boxes = []
        for t in frame_tracks:
            frame_offset = frame_id - t.start_frame
            box = t.boxes[frame_offset]
            track_id = t.id
            all_boxes.append((track_id, box))
        return all_boxes

'''
def main():
    cap = cv2.VideoCapture('./data/ARENA-N1-01_02_ENV_RGB_3.mp4')
    #f = open('./data/ARENA-N1-01_02_ENV_RGB_3/gt/gt.txt', 'r')
    f = open('./data/boxes_blackswan.txt.txt', 'r')
    gt = [x.strip() for x in f.readlines()]
    gt = [x.split(',') for x in gt][:-1]
    gt = np.array(gt).astype(np.int32)
    frame_ids = gt[:, 0] - 1
    #print("frames id : " , frame_ids)
    boxes = gt[:, 2:6]
    boxes[:, 2:4] += boxes[:, :2]


    tracker = Tracker()
    ret = 1
    index = 0
    use_optical_flow = True

    while ret:
        ret, frame = cap.read()
        print(ret)
        if index==0: print(frame)
        if not ret:
            break
        cur_gts = np.where(frame_ids == index)[0]
        cur_boxes = boxes[cur_gts, :]

        t = time.time()
        if use_optical_flow:
            tracker.track_boxes(cur_boxes, frame)
        else:
            tracker.track_boxes(cur_boxes)
        tt = time.time() - t
        #print(1/tt)

        for t in tracker.active_tracks_ids:
            all_boxes = tracker.tracks[t].boxes
            box = all_boxes[-1]
            track_color = tracker.tracks[t].color
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), track_color, 2)
            cv2.putText(frame, str(t), (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            centers = [((b[0]+b[2])//2, b[3]) for b in all_boxes]
            for c in centers:
                cv2.circle(frame, c, 2, color=track_color, thickness=2)
        cv2.imshow('image', frame)
        cv2.waitKey(1)
        index += 1
    return Tracker

'''

def main(img_sequences_name,path_original_img,path_boxes_txt,string_data):
    print("avvio tracker")
    for i in img_sequences_name:
        path_boxes_txt = os.path.join(path_boxes_txt, i)
        #print("path boxes txt : ", path_boxes_txt)
        path_boxes_txt = os.path.join(path_boxes_txt, "Txt")
        #print("path boxes txt : ", path_boxes_txt)
        path_boxes_txt = os.path.join(path_boxes_txt, "boxes_"+ string_data + '.txt')
        print("path boxes txt : ", path_boxes_txt)
        img_seq_name = i
        path_original_img = path_original_img + "/%5d.jpg"
        print(path_original_img)
        cap = cv2.VideoCapture(path_original_img)
        f_tracker = open(path_boxes_txt, 'r')
        gt = [x.strip() for x in f_tracker.readlines()]
        gt = [x.split(',') for x in gt][:-1]
        gt = np.array(gt).astype(np.int32)
        frame_ids = gt[:, 0] - 1
        boxes = gt[:, 2:6]
        boxes[:, 2:4] += boxes[:, :2]


        tracker = Tracker()
        ret = 1
        index = 0
        use_optical_flow = True


        while ret:
            ret, frame = cap.read()
            #print(frame_ids)
            if not ret:
                break
            cur_gts = np.where(frame_ids == index)[0]
            cur_boxes = boxes[cur_gts, :]

            t = time.time()
            if use_optical_flow:
                tracker.track_boxes(cur_boxes, frame)
            else:
                tracker.track_boxes(cur_boxes)
            tt = time.time() - t
            #print(1/tt)

            for t in tracker.active_tracks_ids:
                all_boxes = tracker.tracks[t].boxes
                box = all_boxes[-1]
                track_color = tracker.tracks[t].color
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), track_color, 2)
                cv2.putText(frame, str(t), (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                centers = [((b[0]+b[2])//2, b[3]) for b in all_boxes]
                for c in centers:
                    cv2.circle(frame, c, 2, color=track_color, thickness=2)
            #cv2.imshow('image', frame)
            #cv2.waitKey(10)
            index += 1

        # tracker ottenuto -----------
        # get good boxes
        #img_seq_name = "blackswan"
        f_good = open(path_boxes_txt + '/boxes_good_' + img_seq_name + '.txt', 'w')

        list_tracks = []
        print("Tracks : " + str(tracker.tracks.__len__()))
        for i in tracker.tracks:
            print("----------------------------")
            print("id : " + str(i.id))
            print("bbox : " + str(i.boxes))
            print("start frame : " + str(i.start_frame))
            print("track_length : " + str(i.track_length))
            if i.track_length >= 3:
                print("Da salvare")
                list_tracks.append(i)
                for z in range(0, i.track_length):
                    box = i.boxes[z]
                    f_good.write(str(i.id) + "," + str(z + i.start_frame) + "," + str(box[0]) + "," + str(box[1]) + "," + str(
                        box[2] - box[0]) + "," + str(box[3] - box[1]) + "\n")
                    print(str(i.id) + "," + str(z + i.start_frame) + "," + str(box[0]) + "," + str(box[1]) + "," + str(
                        box[2] - box[0]) + "," + str(box[3] - box[1]))
        print(list_tracks.__len__())
        for j in list_tracks:
            print("DA SALVARE ----------------------------")
            print("id : " + str(j.id))
            print("bbox : " + str(j.boxes))
            print("start frame : " + str(j.start_frame))
            print("track_length : " + str(j.track_length))






if __name__ == "__main__":
    print("Eseguo")
    tracker = main()







