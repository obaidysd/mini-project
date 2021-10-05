import sqlite3
from passlib.hash import sha256_crypt

admin_fname = 'Admin'
admin_lname = 'Admin'
admin_email = 'admin@admin.com'
admin_sex = 'None'
admin_age = 0
admin_password = 'Admin@123'
admin_password = sha256_crypt.hash(admin_password)

admin_tuple = (admin_email, admin_fname, admin_lname,
               admin_password, admin_age, admin_sex)


def create_connection():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    return c, conn


def create_table():
    c, _ = create_connection()
    user_query = '''CREATE TABLE user(
        id integer PRIMARY KEY AUTOINCREMENT,
        email text NOT NULL,
        fname text NOT NULL,
        lname text NOT NULL,
        password text NOT NULL,
        age integer NOT NULL,
        sex text NOT NULL
    )'''

    task_query = '''CREATE TABLE task(
        id integer PRIMARY KEY AUTOINCREMENT,
        patientId text NOT NULL,
        scanId text NOT NULL,
        result text NOT NULL,
        age integer NOT NULL,
        sex text NOT NULL,
        date text NOT NULL,
        prob text NOT NULL
    )'''
    c.execute(user_query)
    c.execute(task_query)


def create_admin():
    c, conn = create_connection()
    c.execute(
        'INSERT INTO user (email, fname, lname, password, age,sex) VALUES( ?, ?, ?, ?, ?, ?)', admin_tuple)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_table()
    create_admin()
