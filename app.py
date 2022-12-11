import pandas as pd
from matplotlib.figure import Figure
from flask import Flask, render_template, request, redirect, url_for
from io import BytesIO
import base64
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import LogisticRegression as lg
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import train_test_split as tts

app = Flask(__name__)


#! const:
MODELS = ['regression', 'classification']
REG = ['Linear Regression']
CLASS = ['Logistic Regression', 'Random Rorest']

#! Classification:
df = pd.read_csv('iris.csv')
info = df.describe().to_html()
x = df.copy()
y = x.pop('Species')

X_train, X_test, y_train, y_test = tts(x, y, test_size=0.2)

# ? Logistic Regression
model_lg = lg()
model_lg.fit(X_train, y_train)


# ? Random Forest Classifier
model_rf = rf()
model_rf = rf(n_estimators=9,
              max_depth=7,
              min_samples_split=3,
              n_jobs=1,
              min_samples_leaf=0.01)
model_rf.fit(X_train, y_train)
m_lg_test = model_lg.score(X_test, y_test)



@app.route("/")
def index():
    # DATAFRAME
    df = pd.read_csv('iris.csv')
    info = df.describe().to_html(classes="two") # Render a DataFrame as an HTML table.
    x = df.copy()
    y = x.pop('Species')

    # PLOT
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.scatter(x.iloc[:, 0], x.iloc[:, 1]) # first and second columns
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return render_template("index.html", plot=data, info = info, df = df.head().to_html(), models = MODELS)


@app.route("/chs", methods=['POST', 'GET'])
def chs():
    try:
        if request.method == 'POST':
            models_c = request.form.get('model')
            if models_c == "regression":
                return redirect("/regression")

            if models_c == "classification":
                return redirect("/classification") 
        return render_template('index.html', models_c = models_c)
    except UnboundLocalError:
        return render_template('index.html')



@app.route("/classification")
def classification():
    df = pd.read_csv('iris.csv')
    info = df.describe().to_html()
    col_names = df.columns.values[1: -1]
    x = df.copy()
    y = x.pop('Species')
    return render_template("class.html", df=df.head().to_html(), info=info, col_names=col_names, classes=CLASS)

@app.route("/regression")
def regression():
    df = pd.read_csv('house_bishkek.csv')
    df2_info = df.describe().to_html(classes="two")
    col_names = df.columns.values[0:]
    x = df.copy()
    y = x.pop('price_usd')
    return render_template("class.html", info=df2_info, df=df.head().to_html(), col_names=col_names, classes = REG)

@app.route("/models_class", methods=['POST', 'GET'])
def model_class():
    x_axes = request.form.get('X_ax')
    y_axes = request.form.get('Y_ax')
    if x_axes in ['area', 'price_usd']:
        df = pd.read_csv('house_bishkek.csv')
        col_names = df.columns.values[0:]
        info = df.describe().to_html()
        X = df.copy()
        y = X.pop('price_usd')
        col_names = df.columns.values[0:]
        info = []
        for i in df:
            info.append(i)
        models = ['Linear Regression']
    else:
        df = pd.read_csv('iris.csv')
        info = df.describe().to_html()
        df = df.drop(columns = 'Id')
        X = df.copy()
        y = X.pop('Species')
        col_names = X.columns.values[0:]
        info = []
        for i in X:
            info.append(i)
        models = ['Logistic Regression', 'Random Forest Classifier']
    fig = Figure()
    ax = fig.subplots()
    ax.scatter(df[x_axes], df[y_axes])
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plot = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render_template('class.html', df=df.head().to_html(), info=info, classes=models, plot=plot, col_names=col_names)



@app.route("/build_your_m_c", methods=['POST', 'GET'])
def build_your_m_c():
    model = request.form.get('classes_c')
    if model == 'Linear Regression':
        df = pd.read_csv('house_bishkek.csv')
        info = df.head().to_html()
        info2 = df.describe().to_html()
        X = df.copy()
        y = X.pop('price_usd')
        col_names = []
        for i in df:
            col_names.append(i)
        models = ['Linear Regression']
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.20)
        model = lr()
        model.fit(X_train, y_train)
        tts_scores = model.score(X_test, y_test)


        fig = Figure()
        ax = fig.subplots()
        y_pred = model.predict(X_test)
        ax.scatter(df['area'], df['price_usd'])
        ax.plot(X_test, y_pred, color = 'r')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plot_sc = base64.b64encode(buf.getbuffer()).decode("ascii")

    else:
        if model == 'Logistic Regression':
            df = pd.read_csv('iris.csv')
            info = df.head().to_html()
            info2 = df.describe().to_html()
            df['Species'] = df['Species'].map({'Iris-setosa' : 0,
                    'Iris-versicolor' : 1,
                    'Iris-virginica' : 2})
            df = df.drop(columns = 'Id')
            X = df.copy()
            y = X.pop('Species')
            col_names = []
            for i in X:
                col_names.append(i)
            models = ['Logistic Regression', 'Random Forest Classifier']
            X_train, X_test, y_train, y_test = tts(X, y, test_size=0.20)
            model = lr()
            model.fit(X_train, y_train)
            tts_scores = model.score(X_test, y_test)

            fig = Figure()
            ax = fig.subplots()
            ax.scatter(df['PetalLengthCm'], df['PetalWidthCm'], c = df['Species'])
            buf = BytesIO()
            fig.savefig(buf, format="png")
            plot_sc = base64.b64encode(buf.getbuffer()).decode("ascii")
        else:
            df = pd.read_csv('iris.csv')
            info = df.head().to_html()
            info2 = df.describe().to_html()
            df['Species'] = df['Species'].map({'Iris-setosa' : 0,
                    'Iris-versicolor' : 1,
                    'Iris-virginica' : 2})
            df = df.drop(columns = 'Id')
            X = df.copy()
            y = X.pop('Species')
            col_names = []
            for i in X:
                col_names.append(i)
            models = ['Logistic Regression', 'Random Forest Classifier']
            X_train, X_test, y_train, y_test = tts(X, y, test_size=0.20)
            model = rf()
            model.fit(X_train, y_train)
            tts_scores = model.score(X_test, y_test)

            fig = Figure()
            ax = fig.subplots()
            ax.scatter(df['PetalLengthCm'], df['PetalWidthCm'], c = df['Species'])
            buf = BytesIO()
            fig.savefig(buf, format="png")
            plot_sc = base64.b64encode(buf.getbuffer()).decode("ascii")

    return render_template('class.html', info = info, col_names = col_names, info2 = info2, classes=models, tts_scores = tts_scores, plot_sc = plot_sc)



if __name__ == "__main__":
    app.run(debug=True)
