from flask import Flask, render_template, request, session, url_for, redirect, flash, jsonify, send_from_directory
from dbuitlity import create_connection
from passlib.hash import sha256_crypt
from functools import wraps
import torch
import os
import numpy as np
from datetime import datetime, date
from model import complete_task
import base64


app = Flask(__name__)
app.secret_key = 'hel'
UPLOADS_FOLDER = 'static/uploads'
app.config['UPLOADS_FOLDER'] = UPLOADS_FOLDER


def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash("You need to login first")
            return redirect(url_for('index'))
    return wrap


def not_login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if not 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash("You are already logged in")
            return redirect(url_for('dashboard'))
    return wrap


def admin_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session and session['data'][1] == 'admin@admin.com':
            return f(*args, **kwargs)
        else:
            flash("You are not authorised")
            return redirect(url_for('index'))
    return wrap


@app.route('/api', methods=['POST'])
def hom():
    data = request.get_json()
    imgStr = data['image']
    img = base64.b64decode(imgStr)
    '''filename = 'mobile-scan-{}.jpg'.format(
        datetime.now().strftime("%d:%m:%Y-%H:%M:%S"))'''
    filename = "10.jpg"
    apath = os.path.join(app.config['UPLOADS_FOLDER'], filename)
    with open(apath, "wb") as fh:
        fh.write(img)
        fh.close()
    prob = complete_task(apath, filename)[1]
    prob = round(float(prob), 2)
    print(prob)

    img_path = os.path.join('static', 'maps', filename)
    with open(img_path, "rb") as fh:
        responseImg = base64.b64encode(fh.read())
    # print(responseImg)
    return jsonify(filename=filename, image=responseImg, prob=prob)


@app.route('/')
@not_login_required
def index():
    flash('Our site uses cookies to store login info')
    return render_template('index.html')


@app.route('/login', methods=['POST', 'GET'])
@not_login_required
def login():
    c, _ = create_connection()
    c.execute('SELECT * FROM user WHERE email=? ',
              (request.form['email'],))
    data = c.fetchone()
    print(data)
    if data and sha256_crypt.verify(request.form['password'], data[4]):
        session['logged_in'] = True
        session['data'] = data
        flash('Welcome Back {} {}'.format(data[2], data[3]))
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid credentials')
        return redirect(url_for('index'))


@app.route('/register', methods=['POST', 'GET'])
@not_login_required
def register():
    c, conn = create_connection()
    c.execute('SELECT * FROM user WHERE email=?', (request.form['email'],))
    print(request.form)
    if len(c.fetchall()) != 0:
        flash('User already exists please login')
        return redirect(url_for('index'))
    else:
        if request.form['password1'] != request.form['password2']:
            flash('Both password have to be same')
            return redirect(url_for(register_form))
        password = sha256_crypt.hash(request.form['password1'])
        user_tuple = (request.form['email'].lower(),
                      request.form['first-name'].capitalize(), request.form['last-name'].capitalize(), password, request.form['age'], request.form['sex'])

        c.execute(
            'INSERT INTO user (email, fname, lname, password, age,sex) VALUES( ?, ?, ?, ?, ?, ?)', user_tuple)
        conn.commit()
        flash('New user created please login')
        return redirect(url_for('index'))


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    session.clear()
    flash("You have been logged out!")
    return redirect(url_for('index'))


@app.route('/register-form')
@not_login_required
def register_form():
    return render_template('register.html')


@app.route('/dashboard/')
@login_required
def dashboard():
    if session['data'][1] == 'admin@admin.com':
        flash('Loged in as admin')
    return render_template('dashboard.html')


@app.route('/task/', methods=['POST', 'GET'])
@login_required
def task():
    if request.method == 'POST':
        f = request.files['file']    
        '''f.filename = 'scan-{}-{}-{}.jpeg'.format(
            session['data'][2], session['data'][3], datetime.now().strftime("%d:%m:%Y-%H:%M:%S"))        '''
        filename = "10.jpg"
        apath = os.path.join(app.config['UPLOADS_FOLDER'], f.filename) 
        print(apath)
        f.save(apath)
        preds = complete_task(apath, f.filename)
        print(preds)       
        img_path = 'maps/{}'.format(f.filename)
        if preds[0] > preds[1]:
            result = "Normal"
        else:
            result = "Pneumonia"
        task_tuple = (session['data'][0], f.filename,
                      result, session['data'][5], session['data'][6], date.today(), str(preds[1]))
        c, conn = create_connection()
        c.execute(
            'INSERT INTO task (patientid, scanid, result, age, sex, date, prob) VALUES(?, ?, ?, ?, ?, ?,?)', task_tuple)
        conn.commit()
        return render_template('task.html', preds=preds, result=result, img_path=img_path)
    return redirect(url_for('index'))


@app.route('/about/')
def about():
    return render_template('about.html')


@app.route('/demo')
def demo():
    return send_from_directory(app.static_folder, 'demo.zip')


@app.route('/report/')
@login_required
def report():
    c, _ = create_connection()
    c.execute('SELECT * FROM task WHERE patientId=?', (session['data'][0],))
    tasks = c.fetchall()
    print(tasks)
    scans = len(tasks)
    return render_template('report.html', tasks=tasks, scans=scans)


@app.route('/admin/')
@admin_required
def admin():
    results = []
    c, _ = create_connection()
    c.execute('SELECT * FROM task WHERE sex=?', ('male',))
    no_male = len(c.fetchall())
    c.execute('SELECT * FROM task WHERE sex=?', ('female',))
    no_female = len(c.fetchall())
    gender_dict = {'label': 'Gender', '0': no_male,
                   '1': no_female, 'class': ['Male', 'Female'], 'chart': 'chart_Male_Female'}
    results.append(gender_dict)

    start_age = 0
    while start_age < 100:
        c.execute('SELECT * FROM task WHERE age BETWEEN ? AND ? AND result = ?;',
                  (start_age, start_age+19, 'Normal'))
        c1 = len(c.fetchall())

        c.execute('SELECT * FROM task WHERE age BETWEEN ? AND ? AND result = ?;',
                  (start_age, start_age+19, 'Pneumonia'))
        c2 = len(c.fetchall())

        dict = {'label': 'Age Group {} to {}'.format(start_age, start_age+19), '0': c1,
                '1': c2, 'class': ['Normal', 'Pneumonia'], 'chart': 'chart_{}_{}'.format(start_age, start_age+20)}
        results.append(dict)
        start_age += 20
    print(results)
    return render_template('admin.html', results=results)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1')
