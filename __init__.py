from flask import Flask, render_template
import pandas as pd
import numpy  as np
from sklearn.decomposition.pca import PCA
from numpy import linalg as LA

source = pd.read_csv('/home/ashutosh/python/multicollinearity.csv')

# Create a pandas DataFrame object
frame = pd.DataFrame(source)
# Make sure we are working with the proper data -- drop the response variable
cols = [col for col in frame.columns if col not in ['response']]

app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template("main.html")


@app.route('/showDataTable')
def showDataTable():
    title = "Descriptive statistics"
    df = frame[cols]
    data_dsc = df.describe().transpose()
    # dsc = df.describe()

    pca = PCA(n_components=5)
    pca.fit(df)
    pc = pca.explained_variance_ratio_

    data_corr = df.corr()
    eigenValues, eigenVectors = LA.eig(data_corr)
    idx = eigenValues.argsort()[::-1]
    # print sorted(eigenValues, key=int, reverse=True)
    print  eigenValues.argsort()[::-1]
    print  eigenValues.argsort()
    eigenValues = pd.DataFrame(eigenValues[idx]).transpose()
    eigenVectors = pd.DataFrame(eigenVectors[:, idx])

    return render_template("showDataTable.html", title=title, data=df, data_dsc=data_dsc, pca=pd.DataFrame(pc).transpose(),data_corr=data_corr, w=eigenValues, v=eigenVectors)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=8080, passthrough_errors=True)
