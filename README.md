# Attendance_Tracker ProtoType
This repository contains the prototype implementation of a face recognition system, demonstrating the workflow from image acquisition to detection, feature extraction, training, and recognition using CNN.
# Face Recognition System
**What This Does?**  
This is a smart Python program that looks at a photo and tells us who's in it - just like the image is give and this gives the details of the person (from MogoDB Compass). Before that the person images should be trained(registered) then only it can identify the person.

**What It Can Do?**
Finds Faces Fast: Uses smart technology (MTCNN) to spot faces in any photo, even if the lighting isn't perfect or there are multiple people.

Creates a Unique Face "Fingerprint": Converts each face into a special number code (using FaceNet) that's like a digital fingerprint - unique to each person.

Matches People Instantly: Compares the face it found with everyone stored in your database and finds the best match.

**OUTPUT of this code:**

Student ID or Roll Number
Full Name
Parent's Name
Department/Class
Phone Number

How many photos you have of this person

Recognition Results:
{
  'rollNo': '******6749',
  'name': 'John Doe',
  'fatherName': 'Alex Doe',
  'department': 'CSD',
  'contact': '9876543210',
  'images_count': 40
}

**How the Magic Happens**
Step 1 - Spot the Face: The program scans your photo and goes "Found you!" when it detects a face, figuring out exactly where it is in the image.

Step 2 - Create the Face Code: It studies the face's unique features and creates a special 512-number code that captures what makes this face different from everyone else.

Step 3 - Play Detective: The program compares this new face code with all the ones it already knows, looking for the closest match.

Step 4 - Share the Results: Once it finds who is he, it pulls up all the stored information and displays it neatly.

**What we Need?**
Python 3.8 or newer (the programming language that runs everything)

A few helpful libraries that do the heavy lifting:

**Requirements:**
**pip install numpy opencv-contrib-python mtcnn keras-facenet pymongo**
Run it: Type **python face_recognition.py** and watch it work

**Get your answer** - it'll show us exactly who it found and their details

Perfect for schools, offices, or anywhere you need to quickly identify people from photos!
