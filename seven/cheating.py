from typing import Counter
from mss import mss
import torch
import cv2
import numpy as np
import time

MONITOR_WIDTH = 1920  # game res
MONITOR_HEIGHT = 1080  # game res
MONITOR_SCALE = 5  # how much the screen shot is downsized by eg. 5 would be one fifth of the monitor dimensions
region = (
    int(MONITOR_WIDTH / 2 - MONITOR_WIDTH / MONITOR_SCALE / 2),
    int(MONITOR_HEIGHT / 2 - MONITOR_HEIGHT / MONITOR_SCALE / 2),
    int(MONITOR_WIDTH / 2 + MONITOR_WIDTH / MONITOR_SCALE / 2),
    int(MONITOR_HEIGHT / 2 + MONITOR_HEIGHT / MONITOR_SCALE / 2),
)

model = torch.hub.load('D:\EXTERNAL\Valorant-AI-cheats-main\scripts\yolov5', 'custom', path=r'D:\EXTERNAL\Valorant-AI-cheats-main\scripts\test.pt', source='local')
model.conf = 0.5
model.maxdet = 10
model.apm = True

start_time = time.time()
x = 1
counter = 0

with mss() as stc:
    while True:
        screenshot = np.array(stc.grab(region))
        df = model(screenshot, size=736).pandas().xyxy[0]

        counter += 1
        if (time.time() - start_time) > x:
            fps = "fps:" + str(int(counter / (time.time() - start_time)))
            print(fps)
            counter = 0
            start_time = time.time()

        # Keep only the largest detection box
        df = df.assign(area=(df['xmax'] - df['xmin']) * (df['ymax'] - df['ymin']))
        df = df.sort_values('area', ascending=False).iloc[:1, :-1]

        for i in range(len(df)):
            xmin, ymin, xmax, ymax = map(int, df.iloc[i, :4])
            confidence = float(df.iloc[i, 4])

            cv2.rectangle(screenshot, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)

        cv2.imshow("frame", screenshot)
        if cv2.waitKey(1) == ord('l'):
            cv2.destroyAllWindows()
            break
