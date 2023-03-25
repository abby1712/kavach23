import face_recognition
import cv2 
import os
from datetime import datetime
import csv

import numpy as np
import joblib

# Load model
isof = joblib.load('/Users/abby/Downloads/isolation_forest_model.joblib')

print(cv2.__version__)

attendance_file = 'attendance.csv'
header = ['Name', 'First Seen', 'Last Seen', 'Time Elapsed','Avg Time','Behavior']
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
time_since_last_seen=0
time_since_last_seen_min=0
Encodings=[]
Names=[]
n=0
image_dir='/Users/abby/Documents/saves/faces'

for root, dirs, files in os.walk(image_dir):
    print(files)
    for file in files:
        path = os.path.join(root, file)
        name = os.path.splitext(file)[0]
        print(name)
        Names.append(name)
        person = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(person)[0]
        Encodings.append(encoding)

cam = cv2.VideoCapture(0)
known_persons = []
while True:
    _, frame = cam.read()
    frameSmall = cv2.resize(frame, (0,0), fx=0.33, fy=0.33)

    frameRGB = cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB)
    facePositions = face_recognition.face_locations(frameRGB)
    allEncodings = face_recognition.face_encodings(frameRGB, facePositions)
    
    # Read the attendance data from the CSV file
    attendance_dict = {}
    with open(attendance_file, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            attendance_dict[row['Name']] = {'First Seen': row['First Seen'], 'Last Seen': row['Last Seen']}

    for (top, right, bottom, left), face_encoding in zip(facePositions, allEncodings):
        name = "Unknown Person"
        matches = face_recognition.compare_faces(Encodings, face_encoding)
        
        if True in matches:
            first_match_index = matches.index(True)
            name = Names[first_match_index]
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Check if person was seen in last minute
            if name in known_persons:
                last_seen_time = datetime.strptime(attendance_dict[name]['Last Seen'], '%Y-%m-%d %H:%M:%S')
                time_since_last_seen = (datetime.now() - last_seen_time).total_seconds()
                time_since_last_seen_min=(time_since_last_seen*(0.0166667))
                print("time_since_last_seen_min:",time_since_last_seen_min)
                if time_since_last_seen < 10:
                    continue
            else:
                attendance_dict[name] = {'First Seen': current_time, 'Last Seen': current_time}
                known_persons.append(name)
            
            # Update the attendance dictionary
            attendance_dict[name]['Last Seen'] = current_time
            
        top = top*3
        bottom = bottom*3
        right = right*3
        left = left*3
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
        cv2.putText(frame, name, (left, top-6), font, .75, (0,0,255), 2)
        cv2.putText(frame, "Person Identified as:", (left, top-25), font, .75, (0,0,255), 2)
    # Write the updated attendance data to the CSV file
    for name, times in attendance_dict.items():
            first_seen_time = datetime.strptime(times['First Seen'], '%Y-%m-%d %H:%M:%S')
            last_seen_time = datetime.strptime(times['Last Seen'], '%Y-%m-%d %H:%M:%S')
            time_elapsed = last_seen_time - first_seen_time
            times['Time Elapsed'] = str(time_elapsed)
            timeelapsedmin=(time_elapsed.total_seconds())*(0.0166667)
            n=n+1
            time_average = timeelapsedmin/n
            times['Avg_Time']=time_average


            
            new_point = np.array([[float(time_since_last_seen_min)]])

            # Predict if new data point is an anomaly or not
            is_anomaly = isof.predict(new_point) == -1

            if is_anomaly:
                times['Behavior'] = 'anomaly detected.'
            else:
                times['Behavior'] = 'No anomaly detected.'

    with open('attendance.csv', mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        for name, times in attendance_dict.items():
            writer.writerow({'Name': name, 'First Seen': times['First Seen'], 'Last Seen': times['Last Seen'], 'Time Elapsed': times['Time Elapsed'],'Avg Time': times['Last Seen'],'Behavior': times['Behavior']})
       
        cv2.imshow('Picture',frame)
        cv2.moveWindow('Picture',0,0)
    if cv2.waitKey(1)==ord('q'):
        break   
cam.release()

cv2.destroyAllWindows() 