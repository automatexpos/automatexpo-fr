# api/match.py
import os
import tempfile
import base64
from flask import Flask, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

@app.route('/api/match', methods=['POST'])
def match():
    # Expect: 'input_image' (one file), and 'db_images' (one or more files)
    if 'input_image' not in request.files:
        return jsonify({'error': 'input_image is required'}), 400

    input_file = request.files['input_image']
    db_files = request.files.getlist('db_images')
    if not db_files:
        return jsonify({'error': 'db_images (one or more) are required'}), 400

    tmpdir = tempfile.mkdtemp()
    input_path = os.path.join(tmpdir, input_file.filename)
    input_file.save(input_path)

    best = None
    best_distance = None

    for f in db_files:
        try:
            fp = os.path.join(tmpdir, f.filename)
            f.save(fp)

            # DeepFace.verify returns a dict with keys like 'verified' and 'distance'
            res = DeepFace.verify(input_path, fp, enforce_detection=False)

            distance = res.get('distance', None)
            verified = res.get('verified', False)

            # Keep best (smallest) distance
            if verified:
                if best is None or (distance is not None and distance < best_distance):
                    best = fp
                    best_distance = distance
        except Exception as e:
            # non-blocking: log and continue
            print('error processing', f.filename, '->', e)

    if best:
        # read matched image bytes and return base64 to frontend
        with open(best, 'rb') as fh:
            data = fh.read()
        mime = 'image/jpeg'  # best-effort; Vercel/frontend can still display
        b64 = base64.b64encode(data).decode('utf-8')

        match_percentage = None
        if best_distance is not None:
            # convert small distance -> higher score (approx)
            match_percentage = max(0.0, 100.0 - best_distance * 100.0)

        return jsonify({
            'matched_image_data': b64,
            'matched_image_mime': mime,
            'match_percentage': match_percentage
        })

    return jsonify({'matched_image_data': None}), 200

# expose `app` (WSGI) for Vercel
