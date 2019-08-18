from flask import Flask, render_template, request, redirect, Response, jsonify, json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import operator
from itertools import islice
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


import numpy as np
import numpy.matlib
app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    global df
    global data
    global data_random
    global data_stratified

    data = df[['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']]
    #print data
    scaler = MinMaxScaler()
    MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(data)
    scaler.transform(data)

    data_random = get_random_sampled_data()
    data_stratified = get_stratified_sampled_data()
    data_stratified = data_stratified.drop(columns = ['label'])

    exp_var, cum_exp_var = find_intrinsic_dimensionality_PCA(data_random)
    chart_data = {}

    line1 = []
    line2 = []
    #print len(exp_var)
    for i, val in enumerate(exp_var):
        obj = {}
        obj['key'] = i+1
        obj['value'] = val
        line1.append(obj)

    for i, val in enumerate(cum_exp_var):
        obj = {}
        obj['key'] = i+1
        obj['value'] = val
        line2.append(obj)

    chart_data['l1'] = line1
    chart_data['l2'] = line2

    chart_data = json.dumps(chart_data, indent=2)
    ret_data = {'chart_data': chart_data}

    if request.method == 'GET':
        return render_template("index.html", data=ret_data)
    else:
        return jsonify(ret_data)


@app.route("/intrinsic-stratified", methods=['GET', 'POST'])
def get_intrinsic_dimen_stratified():

    exp_var, cum_exp_var = find_intrinsic_dimensionality_PCA(data_stratified)
    chart_data = {}

    line1 = []
    line2 = []
    #print len(exp_var)
    for i, val in enumerate(exp_var):
        obj = {}
        obj['key'] = i+1
        obj['value'] = val
        line1.append(obj)

    for i, val in enumerate(cum_exp_var):
        obj = {}
        obj['key'] = i+1
        obj['value'] = val
        line2.append(obj)

    chart_data['l1'] = line1
    chart_data['l2'] = line2

    chart_data = json.dumps(chart_data, indent=2)
    ret_data = {'chart_data': chart_data}
    return jsonify(ret_data)

@app.route("/loadings-random", methods=["GET", 'POST'])
def get_pca_loadings_random():

    return compute_loadings(data_random)

@app.route("/loadings-stratified", methods=["GET", 'POST'])
def get_pca_loadings_stratified():

    return compute_loadings(data_stratified)


@app.route("/two-pca-random", methods=["GET", 'POST'])
def get_top_two_pca_random():

    return find_top_two_pca_vectors(data_random)


@app.route("/two-pca-stratified", methods=["GET", 'POST'])
def get_top_two_pca_stratified():

    return find_top_two_pca_vectors(data_stratified)

def find_top_two_pca_vectors(df):
    pca = PCA(n_components=2)

    df_mat = df.values

    print df_mat.shape

    df_res = pca.fit_transform(df_mat)

    chart_data = json.dumps(df_res.tolist(), indent=2)
    ret_data = {'chart_data': chart_data}
    return jsonify(ret_data)

@app.route("/mds-random-euclidean", methods=['GET', 'POST'])
def get_MDS_embeddings_euclidean_random():

    return find_MDS_embeddings(data_random, "euclidean")


@app.route("/mds-stratified-euclidean", methods=['GET', 'POST'])
def get_MDS_embeddings_euclidean_stratified():

    return find_MDS_embeddings(data_stratified, "euclidean")


@app.route("/mds-random-correlation", methods=['GET', 'POST'])
def get_MDS_embeddings_correlation_random():

    return find_MDS_embeddings(data_random, "correlation")


@app.route("/mds-stratified-correlation", methods=['GET', 'POST'])
def get_MDS_embeddings_correlation_stratified():

    return find_MDS_embeddings(data_stratified, "correlation")



def find_MDS_embeddings(df, dist):

    df_mat = df.values
    n = df_mat.shape[1]
    df_inp = metrics.pairwise.pairwise_distances(df_mat, metric=dist)
    embedding = MDS(n_components=2, dissimilarity="precomputed")
    df_trans = embedding.fit_transform(df_inp)
    print df_trans.shape
    print df_trans

    chart_data = json.dumps(df_trans.tolist(), indent=2)
    ret_data = {'chart_data': chart_data}
    return jsonify(ret_data)

def get_random_sampled_data():

    return data.sample(data.shape[0]/2)


def get_stratified_sampled_data():

    k = get_elbow()
    kmeanModel = KMeans(n_clusters=k).fit(data)
    labels =  kmeanModel.labels_
    data['label'] = labels

    df_strat = None
    for index in range(k):

        temp = data[data.label == index]
        sampled = temp.sample(temp.shape[0]/2)
        if df_strat is None:
            df_strat = sampled
        else:
            df_strat = df_strat.append(sampled)

    return df_strat


def get_elbow():

    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(data)
        distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

    values=list(distortions)

    nPoints = len(values)
    allCoord = np.vstack((range(nPoints), values)).T

    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))

    vecFromFirst = allCoord - firstPoint

    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel

    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))

    idxOfBestPoint = np.argmax(distToLine)

    return idxOfBestPoint + 1


def find_intrinsic_dimensionality_PCA(df):

    pca = PCA()

    df_mat = df.values

    pca.fit_transform(df_mat)
    exp_var = pca.explained_variance_ratio_

    cum_exp_var = np.cumsum(exp_var)


    return (exp_var.tolist(), cum_exp_var)

def count_principle_components(eig_val):
    count = 0
    for i in eig_val:
        if i > 1:
            count += 1
    return count

def compute_loadings(df):

    cov_ran = df.cov()

    eig_val, eig_vec = numpy.linalg.eig(cov_ran)

    principle_count = 2

    values = np.sum(np.square(eig_vec[:, : principle_count]), axis=1)

    temp_dict = {}
    for i, val in enumerate(values.tolist()):
        temp_dict[i] = val

    sorted_list = sorted(temp_dict.items(), key=operator.itemgetter(1), reverse=True)

    principle_columns = []

    for entry in sorted_list[:3]:
        principle_columns.append(entry[0])

    headers = list(df.columns.values)
    columns = []
    for ind in principle_columns:
        columns.append(headers[ind])

    new = df[columns].values
    obj = {}
    obj['headers'] = columns
    obj['values'] = new.tolist()
    chart_data = json.dumps(obj, indent=2)
    ret_data = {'chart_data': chart_data}
    return jsonify(ret_data)


if __name__ == "__main__":
    df = pd.read_csv('fires.csv')
    app.run(debug=True)
