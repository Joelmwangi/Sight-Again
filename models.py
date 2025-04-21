import os
import json
import face_recognition

PROFILE_DB = 'profiles.json'

def save_profile(name, image_filename):
    if os.path.exists(PROFILE_DB):
        with open(PROFILE_DB, 'r') as file:
            profiles = json.load(file)
    else:
        profiles = []

    profiles.append({'name': name, 'image': image_filename})
    with open(PROFILE_DB, 'w') as file:
        json.dump(profiles, file)

def load_profiles(upload_folder):
    encodings = []
    names = []
    if os.path.exists(PROFILE_DB):
        with open(PROFILE_DB, 'r') as file:
            profiles = json.load(file)
        for profile in profiles:
            path = os.path.join(upload_folder, profile['image'])
            if not os.path.exists(path):
                continue
            image = face_recognition.load_image_file(path)
            face_encoding = face_recognition.face_encodings(image)
            if face_encoding:
                encodings.append(face_encoding[0])
                names.append(profile['name'])
    return encodings, names
