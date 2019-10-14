from imageai.Detection import ObjectDetection
import os
import csv
import pandas as pd
import datetime
import calendar
import cv2
import time
from PIL import Image
import serial
import time
from sklearn import metrics

t=0
ser1=serial.Serial('/dev/cu.usbmodem14201',9600)
time.sleep(3)
ser1.write('t'.encode())

temp_en=(ser1.readline().strip())
temp=temp_en.decode('utf-8')
#print(temp)
if(temp > '20'):
    t = 1
t1 = t

'''camera = cv2.VideoCapture(0)
for i in range(1):
    time.sleep(1)
    return_value, image = camera.read()
    cv2.imwrite('image'+str(i)+'.jpg', image)
del(camera)'''



Person = 0
Car = 0
room = 0

def obj_detect(Person,Car):
    Person = 0
    Car = 0
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "IMAGE1.jpg"), output_image_path=os.path.join(execution_path , "Out_Image.jpg"))
    print("Objects detected in the image along with their percentages: ")

    for eachObject in detections:
        print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
        if eachObject["name"] == "person":
            Person = 1
        if eachObject["name"] == "car":
            Car = 1
    return Person,Car
Per,C=obj_detect(Person,Car)
if (Per == 1):
    print("Person Detected")
if (C == 1):
    print("Car Detected")
#print(C)
print("---------------------------------------")


  
def findDay(date): 
  born = datetime.datetime.strptime(date, '%d %m %Y').weekday() 
  return (calendar.day_name[born]) 
  
now = datetime.datetime.now()
print ("Current date and time : ")
date=now.strftime("%d %m %Y")
day=findDay(date)
#print("Today: ",day)
#date1=now.strftime("%H:%M:%S")
date1=now.strftime("%H")
print("Hours: ",date1)

if (int(date1) >= 6 and int(date1) <12):
    zone = 1
   
if (int(date1) >=12 and int(date1) <18):
    zone = 2

if (int(date1) >=18 and int(date1) <24):
    zone = 3

if (int(date1) >=0 and int(date1) <6):
    zone = 4
print("Time Category: ",zone)
#print(zone)
print("-----------------------------------------")

if (day == "Monday" or "Tuesday" or "Wednesday" or "Thursday" or "Friday"):
    dayofweek = 1
elif (day == "Saturday" or "Sunday"):
    dayofweek = 2
print("Today: ",day)
print("Day Category: ",dayofweek)
#print(dayofweek)
print("-----------------------------------------")


'''with open("Performance.csv", 'a') as csv2:
    write = csv.writer(csv2)
    write.writerow(app)'''

print("-----------------FAN---------------------")

app = 0
appr1 = []
if(int(Per)==1 and int(t1)==1 and int(room) == 0):
    val = 10
else:
    val = 5

if(t1 == 1):
    print("Temperature is not optimal: ",temp)
#Person = 1

row = [int(dayofweek),int(zone),int(Per),int(room),int(t1),int(val)]

with open('FAN1.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)

csvFile.close()

df = pd.read_csv('FAN1.csv')
headers = df.columns.values
data = df.iloc[:,0:-1].values
target = df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data,target,test_size=0.1,random_state=0,shuffle=False)
#print('Training set: ')
#for x,y in zip(x_train,y_train):
    #print((x,y))
print('Test set: ')
for x,y in zip(x_test,y_test):
    print((x,y))
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)
print("Predictions: ",y_pred)

print("Accuracy Percentage for FAN Control: ",metrics.accuracy_score(y_test,y_pred)*100)

if(int(y_pred[-1]) == 10):
    print("Fan is turned ON")
    ser1.write('f'.encode())

if(int(y_pred[-1]) == int(val)):
    app = 1
else:
    app = 0
appr1 = [int(app)]

with open("Performance.csv", 'a') as csv2:
    wrt = csv.writer(csv2)
    wrt.writerow(appr1)

print("-------------------------------------------")
print("-----------------LIGHT---------------------")

app = 0
appr2 = []
if(int(zone)==3 and int(Per)==1 and int(room)==0):
    val1 = 10
else:
    val1 = 5
print("val1: ",val1)

row1 = [int(dayofweek),int(zone),int(Per),int(room),int(val1)]
with open('LIGHT1.csv', 'a') as csvFile:
    writer1 = csv.writer(csvFile)
    writer1.writerow(row1)

csvFile.close()

df1 = pd.read_csv('LIGHT1.csv')
headers1 = df1.columns.values
data1 = df1.iloc[:,0:-1].values
target1 = df1.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data1,target1,test_size=0.1,random_state=0,shuffle=False)
#print('Training set: ')
#for x,y in zip(x_train,y_train):
    #print((x,y))
print('Test set: ')
for x1,y1 in zip(X_test,Y_test):
    print((x1,y1))
from sklearn.naive_bayes import GaussianNB
gnb1 = GaussianNB()
gnb1.fit(X_train,Y_train)
Y_pred = gnb1.predict(X_test)
print("Predictions: ",Y_pred)

print("Accuracy Percentage for LIGHT Control: ",metrics.accuracy_score(Y_test,Y_pred)*100)

if(int(Y_pred[-1]) == 10):
    print("LIGHT is turned ON")
    ser1.write('l'.encode())

if(int(Y_pred[-1]) == int(val1)):
    app = 1
else:
    app = 0
appr2 = [int(app)]

with open("Performance.csv", 'a') as csv2:
    wrt = csv.writer(csv2)
    wrt.writerow(appr2)

print("--------------------------------------------")


try:  

    img  = Image.open('Out_Image.jpg')
    img.show()

except IOError: 

    pass


print("-----------------GARAGE---------------------")

app = 0
appr3 =[]

if(int(Per)==1 and int(C)==1 and int(room)==1):
    val2 = 10
else:
    val2 = 5

row2 = [int(dayofweek),int(zone),int(Per),int(C),int(room),int(val2)]
#row = ['1','1','1','1','1','10']

with open('GARAGE1.csv', 'a') as csvFile:
    writer2 = csv.writer(csvFile)
    writer2.writerow(row2)

csvFile.close()

df2 = pd.read_csv('GARAGE1.csv')
headers2 = df2.columns.values
data2 = df2.iloc[:,0:-1].values
target2 = df2.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train2, x_test2, y_train2, y_test2 = train_test_split(data2,target2,test_size=0.1,random_state=0,shuffle=False)
#print('Training set: ')
#for x,y in zip(x_train,y_train):
    #print((x,y))
print('Test set: ')
for x2,y2 in zip(x_test2,y_test2):
    print((x2,y2))
from sklearn.naive_bayes import GaussianNB
gnb2 = GaussianNB()
gnb2.fit(x_train2,y_train2)
y_pred2 = gnb2.predict(x_test2)
print("Predictions: ",y_pred2)

print("Accuracy Percentage for GARAGE DOOR Control: ",metrics.accuracy_score(y_test2,y_pred2)*100)

if(int(y_pred2[-1]) == 10):
    print("GARAGE DOOR is OPENED")
    ser1.write('g'.encode())

if(int(y_pred2[-1]) == int(val2)):
    app = 1
else:
    app = 0
appr3 = [int(app)]

with open("Performance.csv", 'a') as csv2:
    wrt = csv.writer(csv2)
    wrt.writerow(appr3)


print("-------------ACCURACY-------------------")

rows = []
sum = 0
with open("Performance.csv",'r') as csvPerf:
    csvread = csv.reader(csvPerf)
    fields = next(csvread)
    for row in csvread:
        rows.append(row)
    print("rows: %d"%(csvread.line_num))
for row in rows[:csvread.line_num]:
    for col in row:
        c = int(col,10)
        #print("%d"%c)
        sum = sum + c
per = (sum/csvread.line_num)*100
print("Accuracy Percentage of the system: ")
print("%d"%per)




print("-------------------END----------------------")

