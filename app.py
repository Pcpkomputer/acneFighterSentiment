# Import Module Flask dan Yang Lain Untuk Keperluan HTTP Server
from flask import Flask, json, render_template,request,session,redirect,url_for,jsonify,send_file

# Import Module Koneksi Mysql Untuk Mengakses Database
from database.koneksi import mydb

# Import Module json untuk manipulasi data python menjadi json
import json

# Fungsi Utils Untuk Preprocessing Text dan Generate Model
from utils import preprocessingtext, stopwordremovaltext, casefoldingtext, create_model_bydataset

# Import Module Pandas
import pandas as pd

# Import Module Numpy
import numpy as np

# Module pdfkit untuk mengubah HTML menjadi PDF
import pdfkit as pdf

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Inisiasi Variabel main application flask
app = Flask(__name__)
app.secret_key="thisisasecretkey"



# Route GET dan POST untuk nedpoint "/" 
# Berisi fungsi untuk mengecek session apakah sudah login atau belum
# Kemudian terdapat fungsi ketika dilakukan request POST maka melakukan pencocokan antara username password input dengan yang berada pada database
# Ketika dilakukan request GET maka akan menampilkan html biasa

@app.route("/", methods=["POST","GET"])
def index():
    if 'login' in session:
        return redirect(url_for("dashboard"))


    if request.method=="POST":
        email = request.form["email"]
        password = request.form["password"]

        mydb.connect()
        cursor = mydb.cursor()

        cursor.execute("SELECT * FROM admin WHERE email=%s",(email,))
        row = cursor.fetchone()

        cursor.close()
        mydb.close()
        
        if row==None:
            return render_template("login.html",err=True,message="Akun tidak ditemukan")
        else:
            e = row[1]
            p = row[2]

            if e==email and p==password:
                session["login"] = True
                session["email"] = email
                session["password"] = password
                return redirect(url_for("dashboard"))
            else:
                return render_template("login.html",err=True,message="Login gagal")

    return render_template("login.html")


# Route GET untuk endpoint /dashboard
# Berfungsi untuk menampilkan halaman dashboard dan dilakukan logic query pada database yang diperlukan

@app.route("/dashboard")
def dashboard():

    if 'login' not in session:
        return redirect(url_for("index"))


    mydb.connect()
    cursor = mydb.cursor()

    cursor.execute("SELECT * FROM dataset")
    row = cursor.fetchall()

    lenpositif = 0
    lennegatif = 0

    for x in row:
        if x[1]=="Positif":
            lenpositif=lenpositif+1
        else:
            lennegatif=lennegatif+1

    cursor.close()
    mydb.close()

    email = session["email"]
    return render_template("dashboard.html",email=email,count=len(row),lenpositif=lenpositif,lennegatif=lennegatif)

# Route GET dan POST untuk endpoint /klasifikasicuitan
# Berfungsi untuk menghandle klasifikasi single cuitan

@app.route("/klasifikasicuitan", methods=["POST","GET"])
def klasifikasicuitan():

    if 'login' not in session:
        return redirect(url_for("index"))


    if request.method=="POST":

        cuitan = request.form["cuitan"]

        ## menambah preprocessing text pada klasifikasi cuitan/text
        cuitan = preprocessingtext(cuitan)

        model, vectorizer = create_model_bydataset()

        x = vectorizer.transform([cuitan])

        predicted = model.predict(x)
        email = session["email"]

        return render_template("klasifikasicuitan.html",email=email,predicted=predicted[0])


    email = session["email"]
    return render_template("klasifikasicuitan.html",email=email)

# Route GET dan POST untuk endpoint /klasifikasicuitanexcel
# Berfungsi untuk menghandle klasifikasi bulk cuitan

@app.route("/klasifikasicuitanexcel", methods=["POST","GET"])
def klasifikasicuitanexcel():
    if 'login' not in session:
        return redirect(url_for("index"))
    if request.method=="POST":

        if request.form["action"]=="pdfprint":
            
            json_ = request.form["json"]

            parsed = json.loads(json_)

            df = pd.DataFrame(parsed)

            df.to_html('print.html')

            pdf.from_file('print.html', 'print.pdf')

            return send_file("print.pdf")

        file = request.files["dataset"]
        excel = pd.read_excel(file)

        model, vectorizer = create_model_bydataset()

        text = [x[1]["Text"] for x in excel.iterrows()]

        cleansing = [(preprocessingtext(x),) for x in text]
        casefolding = [(casefoldingtext(x[0]),) for x in cleansing]
        tokenizing = [str(x[0].split(" ")) for x in casefolding]
        stopwordremoval = [str(stopwordremovaltext(x[0]).split(" ")) for x in casefolding]
        stemming = [stopwordremovaltext(x[0]) for x in casefolding]

        payload = []

        for index,x in enumerate(text):

            pre = preprocessingtext(x)
            txt = vectorizer.transform([pre])
            predicted = model.predict(txt)
        
            payload.append({
                "before":x,
                "after":pre,
                "cleansing":cleansing[index][0],
                "casefolding":casefolding[index][0],
                "tokenizing":tokenizing[index],
                "stopwordremoval":stopwordremoval[index],
                "stemming":stemming[index],
                "predicted":predicted[0]
            })
        print(payload)
        email = session["email"]
        return render_template("klasifikasicuitanexcel.html",email=email,payload=enumerate(payload),dump=json.dumps(payload))
    email = session["email"]
    return render_template("klasifikasicuitanexcel.html",email=email,payload=[],dump=json.dumps([]))


# Route GET dan POST untuk endpoint /importdataset
# Berisi fungsi untuk melakukan operasi import dataset dengan excel

@app.route("/importdataset",methods=["POST","GET"])
def importdataset():

    if 'login' not in session:
        return redirect(url_for("index"))

    if request.method=="POST":
        file = request.files["dataset"]
        excel = pd.read_excel(file)

        text = [(x[1]["Text"],x[1]["Labelling"]) for x in excel.iterrows()]

        cleansing = [(preprocessingtext(x[0]),) for x in text]
        casefolding = [(casefoldingtext(x[0]),) for x in cleansing]
        tokenizing = [(str(x[0].split(" ")),) for x in casefolding]
        stopwordremoval = [(str(stopwordremovaltext(x[0]).split(" ")),) for x in casefolding]
        stemming = [(stopwordremovaltext(x[0]),) for x in casefolding]
        
        mydb.connect()
        cursor = mydb.cursor()



        cursor.execute("DELETE FROM dataset")
        cursor.execute("DELETE FROM preprocessing")
        cursor.execute("DELETE FROM cleansing")
        cursor.execute("DELETE FROM casefolding")
        cursor.execute("DELETE FROM tokenizing")
        cursor.execute("DELETE FROM stopwordremoval")
        cursor.execute("DELETE FROM stemming")




        cursor.executemany("INSERT INTO dataset VALUES (%s,%s)",text)
        # cursor.executemany("INSERT INTO preprocessing VALUES (%s)",preprocessed)

        cursor.executemany("INSERT INTO cleansing VALUES (%s)",cleansing)
        cursor.executemany("INSERT INTO casefolding VALUES (%s)",casefolding)
        cursor.executemany("INSERT INTO tokenizing VALUES (%s)",tokenizing)
        cursor.executemany("INSERT INTO stopwordremoval VALUES (%s)",stopwordremoval)
        cursor.executemany("INSERT INTO stemming VALUES (%s)",stemming)


        mydb.commit()
        cursor.close()
        mydb.close()


        return redirect(url_for("importdataset"))

    email = session["email"]

    payload = []

    mydb.connect()
    cursor = mydb.cursor()

    cursor.execute("SELECT * FROM dataset")
    dataset = cursor.fetchall()

    # cursor.execute("SELECT * FROM preprocessing")
    # preprocessing = cursor.fetchall()

    cursor.execute("SELECT * FROM cleansing")
    cleansing = cursor.fetchall()

    cursor.execute("SELECT * FROM casefolding")
    casefolding = cursor.fetchall()

    cursor.execute("SELECT * FROM tokenizing")
    tokenizing = cursor.fetchall()

    cursor.execute("SELECT * FROM stopwordremoval")
    stopwordremoval = cursor.fetchall()

    cursor.execute("SELECT * FROM stemming")
    stemming = cursor.fetchall()

    cursor.close()
    mydb.close()

    payload = []

    for index, value in enumerate(dataset):
        payload.append({
            "before":value[0],
            "cleansing":cleansing[index][0],
            "casefolding":casefolding[index][0],
            "tokenizing":tokenizing[index][0],
            "stopwordremoval":stopwordremoval[index][0],
            "stemming":stemming[index][0],
            "label":value[1]
        })



    return render_template("importdataset.html",email=email,payload=enumerate(payload))

# GET Request untuk endpoint /keluar
# Berfungsi untuk menghapus session 

@app.route("/keluar")
def keluar():
    session.pop("login")
    session.pop("email")
    session.pop("password")
    return redirect(url_for("index"))

# Fungsi flask untuk menjalankan app

@app.route("/ujivalidasi", methods=["POST","GET"])
def ujivalidasi():
    if request.method=="POST":
        mydb.connect()
        cursor = mydb.cursor()

        cursor.execute("SELECT * FROM dataset")
        dataset = cursor.fetchall()
        cursor.execute("SELECT * FROM stemming")
        stemmed = cursor.fetchall()

        corpus = [x[0] for x in stemmed]
        vectorizer = TfidfVectorizer()

        
        X = vectorizer.fit_transform(corpus)

        arr = []

        for index,key in enumerate(vectorizer.vocabulary_):
            payload = {}
            payload["vocabulary"]=key
            payload["idf"]=vectorizer.idf_[index]
            arr.append(payload)

        cursor.close()
        mydb.close()


        ##############################

        mydb.connect()
        cursor = mydb.cursor()

        cursor.execute("SELECT * FROM dataset")
        dataset = cursor.fetchall()

        cursor.execute("SELECT * FROM stemming")
        preprocessing = cursor.fetchall()

        cursor.close()
        mydb.close()

        X = [x[0] for x in preprocessing]
        y = [x[1] for x in dataset]

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, train_size=0.8, random_state=45)
        vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, sublinear_tf=True, use_idf=True, stop_words='english')

        X_train_tf = vectorizer.fit_transform(X_train)

        model = MultinomialNB()
        model.fit(X_train_tf, y_train)


        X_test_tf = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_tf)

       
        report = classification_report(y_test, y_pred, target_names=["Positif","Negatif"])


        return render_template("ujivalidasi.html",datasetlen=len(dataset),arr=arr,report=str(report))
    mydb.connect()
    cursor = mydb.cursor()

    cursor.execute("SELECT * FROM dataset")
    dataset = cursor.fetchall()

    cursor.close()
    mydb.close()

    return render_template("ujivalidasi.html",datasetlen=len(dataset),arr=[])

    

if __name__=="__main__":
    app.run(debug=True)