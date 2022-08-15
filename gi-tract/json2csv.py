from cv2 import sort
from matplotlib.pyplot import new_figure_manager
import pandas as pd
import json

fp = open('/home/ray/workspace/Fly_Pluche/kaggle/gi-tract/data.json',
          encoding='utf-8')
p = json.load(fp)
confimed = []
time = []
for i in p:
    time.append(i["日期"])
    confimed.append(i['累计疑似'])
new = confimed
new_framedata = pd.DataFrame([time, new])
new_framedata = pd.DataFrame(new_framedata.values.T,columns=['日期','累计疑似'])
new_framedata.to_csv('china_data.csv')
print('over')