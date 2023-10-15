from flask import Flask, request, render_template, redirect, url_for
from utils import local_to_utc, zip_to_coordinates, send_confirmation_email, initialize_csv
from pathlib import Path
import csv
from config import SCHEDULE_CSV
from scheduler import reset_jobs
import logging
import argparse

logging.basicConfig(filename="./logs/app.log", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

app = Flask(__name__)
csv_path = SCHEDULE_CSV
initialize_csv(csv_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        email = request.form['email'].strip()
        zip_code = request.form['zip_code'].strip()
        notify_time = request.form['notify_time']
        time_zone = request.form['time_zone']
        min_aurora_level = request.form['min_aurora_level']

        # Validate input
        if not email or not zip_code or not notify_time or not time_zone:
            return render_template('index.html', error="Please fill out all fields.")
        # validate email
        if '@' not in email:
            return render_template('index.html', error="Please enter a valid email address.")
        # validate zip code
        try:
            int(zip_code)
            assert len(zip_code) == 5
        except ValueError:
            return render_template('index.html', error="Please enter a valid zip code.")

        longitude, latitude = zip_to_coordinates(zip_code)
        with csv_path.open('a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([email, notify_time, longitude, latitude, time_zone, min_aurora_level])

        # Reschedule jobs
        reset_jobs()

        send_confirmation_email()

        return redirect(url_for('index'))

    return render_template('index.html')

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--port", type=int, default=5000)
    args = args.parse_args()
    return args

if __name__ == '__main__':
    reset_jobs()
    # make port a system arg
    args = parse_args()
    app.run(debug=True, port=args.port)

