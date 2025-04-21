import os
import cv2
import numpy as np
import face_recognition
import winsound
from flask import Flask, render_template, request, redirect, jsonify
from werkzeug.utils import secure_filename
# from models import Profile, load_profiles, save_profile
from models import load_profiles, save_profile
from forms import ProfileForm

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SOUND_FILE'] = os.path.join('static', 'sound', 'beep.wav')

@app.route('/train', methods=['GET', 'POST'])
def train():
    form = ProfileForm()
    if request.method == 'POST':
        file = request.files['image']
        name = request.form.get('name')
        if file and name:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            save_profile(name, filename)
            return redirect('/train')
    return render_template('train.html', form=form)

@app.route('/')
def index():
    known_face_encodings, known_face_names = load_profiles(app.config['UPLOAD_FOLDER'])

    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not video_capture.isOpened():
        return jsonify({'error': 'Webcam not accessible'})

    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if not ret or frame is None:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Does not Match"
                if True in matches:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        winsound.PlaySound(app.config['SOUND_FILE'], winsound.SND_ASYNC)
                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 250, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 250, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Looking for faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
