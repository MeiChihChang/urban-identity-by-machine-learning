# Author Dr.Mei-Chih Chang
# Chair of Information Architecture ETH Zürich
# Taiwanese Workshop : Big-Data Informed Urban Design for Smart City : Workshop 17-19.Nov.2017

from fMRIToFeature_bc_cc_iso import *

trainDir_bc = 'citydata/bc/'
trainDir_cc = 'citydata/cc/'
trainDir_iso = 'citydata/iso/'

predictDir_bc = 'predict/bc/'
predictDir_cc = 'predict/cc/'
predictDir_iso = 'predict/iso/'

numberIsovist = 4
numberStats = 8
totalfeatures = numberIsovist+numberStats*2

featureTrain, filetable = mriToStatsFeature(trainDir_bc, trainDir_cc, trainDir_iso)
predictTrain, filetable2 = mriToStatsFeature(predictDir_bc, predictDir_cc, predictDir_iso)

from time import time
from sklearn import metrics
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from numpy.random import RandomState
from sklearn.preprocessing import StandardScaler
import openpyxl
import pandas as pd

X = featureTrain
Y = predictTrain
rng = RandomState(42)

ks = range(4, 10)
cs = range(4, 10)

inertias = []
inertias_temp = 9999.0

for n_comp in cs:
    for k in ks:
        avg_features_value = np.zeros([k, totalfeatures])
        total_label = np.zeros([k])
        X_std = StandardScaler().fit_transform(X)
        pca = PCA(n_components=n_comp, whiten=True).fit(X_std)
        X_pca = pca.transform(X_std)
        kmeans = KMeans(n_clusters=k, random_state=rng).fit(X_pca)
        Y_pca = pca.transform(Y)
        Z = kmeans.predict(Y_pca)

        np.round(kmeans.cluster_centers_, decimals=3)
        inertias.append(kmeans.inertia_)

        targetname = []
        for i in range(0, k):
            tmpstr = "c" + str(i)
            targetname.append(tmpstr)

        #plot_2D(X_pca, kmeans.labels_, targetname, 'result/bc_cc_iso_clusterlabel_' + str(k) + '_PCA_' + str(n_comp) + '.png')
        for l in range(len(filetable)):
            total_label[kmeans.labels_[l]] = total_label[kmeans.labels_[l]] + 1
            for n in range(totalfeatures):
                avg_features_value[kmeans.labels_[l], n] = avg_features_value[kmeans.labels_[l], n] + featureTrain[l,n]

        for m in range(k):
            for n in range(totalfeatures):
                avg_features_value[m, n] = avg_features_value[m, n] / total_label[m]


        dframe_labels = pd.DataFrame(kmeans.labels_)
        writer = pd.ExcelWriter('result/bc_cc_iso_clusterlabel_' + str(k) + '_PCA_' + str(n_comp) + '.xlsx')
        dframe_labels.to_excel(writer, header = None, index = None, encoding=None)
        writer.save()

        dframe_avg_features_value = pd.DataFrame(avg_features_value)
        writer2 = pd.ExcelWriter('result/bc_cc_iso_avg_features_value_label_' + str(k) + '_PCA_' + str(n_comp) + '.xlsx')
        dframe_avg_features_value.to_excel(writer2, header=None, index=None, encoding=None)
        writer2.save()

        dframe_predict_label = pd.DataFrame(Z)
        writer3 = pd.ExcelWriter(
            'result/bc_cc_iso_predict_label_' + str(k) + '_PCA_' + str(n_comp) + '.xlsx')
        dframe_predict_label.to_excel(writer3, header=None, index=None, encoding=None)
        writer3.save()


dframe_features = pd.DataFrame(featureTrain)
writer4 = pd.ExcelWriter('result/bc_cc_iso_features.xlsx')
dframe_features.to_excel(writer4, header = None, index = None, encoding=None)
writer4.save()

dframe_table = pd.DataFrame(filetable)
writer5 = pd.ExcelWriter('result/bc_cc_iso_index.xlsx')
dframe_table.to_excel(writer5, header = None, index = None, encoding=None)
writer5.save()

#csvOutput('result/bc_cc_iso_index.xls', listToMat(filetable))