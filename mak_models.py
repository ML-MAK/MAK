import time
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.multioutput import RegressorChain
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn import preprocessing


np.random.seed(24)

def load_data(species):
    phes = pd.read_pickle('data/phe_{}'.format(species)).astype(float)
    trait_names = np.array(phes.columns).astype(str)
    snp = pd.read_pickle('data/snp_{}'.format(species))
    k = cosine_similarity(snp)
    # k = rbf_kernel(snp)
    return k, np.array(phes), trait_names


def processing_data(target_trait, assistant_traits, k, ys):
    assistant_traits = np.array(assistant_traits)
    traits = np.append(assistant_traits, target_trait).astype(int)
    # print(traits)
    phe_traits = ys[:, traits]
    y1 = pd.DataFrame(phe_traits).isnull().sum(axis=1)
    nona_index = np.where(y1 == 0)[0]
    x = k[nona_index, :]
    x = x[:, nona_index]
    ys = phe_traits[nona_index, :]
    # print(ys.shape)
    return x, np.array(ys)

def ridgeRC(x, y, alpha=0.1, cross_folds=5, repeat_num=5, random_seed=0):
    rfk = RepeatedKFold(n_splits=cross_folds, n_repeats=repeat_num, random_state=random_seed)
    order = np.arange(y.shape[1])
    # estimator = LinearSVR(C=alpha, random_state=random_seed)
    estimator = Ridge(alpha=alpha, random_state=random_seed)
    lst = []
    for train_index, test_index in rfk.split(x):
        # print('target:',list(target_sort),'\n')
        clf = RegressorChain(base_estimator=estimator, order=order,random_state=random_seed)
        clf.fit(x[train_index, :], y[train_index, :])
        pre = clf.predict(x[test_index, :])
        res = pearsonr(pre[:, -1], y[test_index, -1])
        lst.append(res[0])
    # print('alpha: {}, cor: {}'.format(alpha, np.array(lst).mean()))
    lst = np.array(lst)
    return lst.mean(), np.array(lst)

def ridge(x, y, alphas_range=(0.000001, 5), cross_folds=5, repeat_num=5, random_seed=0):
    results = pd.DataFrame(np.zeros([20, cross_folds*repeat_num]))
    rfk = RepeatedKFold(n_splits=cross_folds, n_repeats=repeat_num, random_state=random_seed)
    k = 0
    alphas = np.linspace(alphas_range[0], alphas_range[1], 20)
    # alphas = (np.array([1.3**i for i in np.arange(20)])-0.9)/40
    for alpha in alphas:
        estimator = Ridge(alpha=alpha, random_state=random_seed)
        # estimator = LinearSVR(C=alpha, random_state=random_seed)
        lst = []
        for train_index, test_index in rfk.split(x, y):
            # print('target:',list(target_sort),'\n')
            clf = estimator
            clf.fit(x[train_index, :], y[train_index])
            pre = clf.predict(x[test_index, :])
            res = pearsonr(pre.flatten(), y[test_index].flatten())
            lst.append(res[0])
        # print('alpha: {}, cor: {}'.format(alpha, np.array(lst).mean()))
        results.iloc[k, :] = lst
        k += 1
    results_means = results.mean(axis=1).values
    best_alpha_index = np.argsort(results_means)[-3:]
    print("best_alpha_index: ",best_alpha_index)
    best_alpha = alphas[best_alpha_index].mean()
    accs = results.iloc[np.argmax(results_means), :].values
    acc = results_means.max()
    return acc, best_alpha, accs

def calculate_accuracy_matrix(k, ys, trait, alpha=0.1,cross_folds=5, repeat_num=5, random_seed=0):
    results_all = []
    for trait_sub in range(ys.shape[1]):
        if trait == trait_sub:
            results_all.append(0)
            continue
        else:
            assistant_traits = [trait_sub]
            x, y = processing_data(trait, assistant_traits, k, ys)
            rfk = RepeatedKFold(n_splits=cross_folds, n_repeats=repeat_num, random_state=random_seed)
            order = np.arange(y.shape[1])
            estimator = Ridge(alpha=alpha, random_state=random_seed)
            # estimator = LinearSVR(C=alpha, random_state=random_seed)
            lst = []
            for train_index, test_index in rfk.split(x):
                # print('target:',list(target_sort),'\n')
                clf = RegressorChain(base_estimator=estimator, order=order, random_state=random_seed)
                clf.fit(x[train_index, :], y[train_index, :])
                pre = clf.predict(x[test_index, :])
                res = pearsonr(pre[:, -1], y[test_index, -1])
                lst.append(res[0])
                # print('alpha: {}, cor: {}'.format(alpha, np.array(lst).mean()))
            results_all.append(np.array(lst).mean())
    # results.to_excel('data/{}/{}_accuracy_matrix_trait.xlsx'.format(species, species))
    return np.array(results_all)

def gblup(K, y, train_index):
    # 个体数量为n
    pi = 3.14159
    n = len(y)
    p = 1
    X = np.matrix([1] * n)
    X = X.T
    Z = np.identity(n)
    y = np.matrix(y).T
    Z = Z[train_index, :]
    X = X[train_index, :]
    n = len(train_index)
    y = y[train_index]
    xtx = np.dot(X.T, X)
    xtxinv = np.linalg.inv(xtx)
    s = np.identity(n) - np.dot(np.dot(X, xtxinv), X.T)
    # # y为表型值向量，转化为n*1的矩阵
    offset = np.sqrt(n)
    hb = np.dot(np.dot(Z, K), Z.T) + offset * np.identity(n)
    hb_eig = np.linalg.eig(hb)
    phi = hb_eig[0] - offset
    U = hb_eig[1]
    shbs = np.dot(np.dot(s, hb), s)
    shub_eig = np.linalg.eig(shbs)
    theta = shub_eig[0][1:] - offset
    Q = np.linalg.eig(shbs)[1][:, 1:]
    omega = np.dot(Q.T, y)
    omega_sq = np.array(omega) ** 2
    n_p = n - p
    theta = np.clip(theta, 0.0001, np.inf)
    reml = lambda lamda: n_p * np.log(np.sum(omega_sq.T / (theta + lamda))) + np.sum(np.log(theta + lamda))
    soln = minimize_scalar(reml, bounds=([10 ** -9, 10 ** 9]), options={"maxiter": 50000})
    # print(soln['success'])
    lamda_opt = soln["x"]
    Hinv = np.dot(U, U.T / (phi + lamda_opt))
    W = np.dot(X.T, np.dot(Hinv, X))
    beta = np.linalg.solve(W, np.dot(X.T, np.dot(Hinv, y)))
    KZt = np.dot(K, Z.T)
    KZt_Hinv = np.dot(KZt, Hinv)
    u = (np.dot(KZt_Hinv, y - np.dot(X, beta)))
    return u, beta


def calculate_gblup(X, y, cross_folds=5, repeat_num=5, random_seed=0):
    kfolds = RepeatedKFold(n_splits=cross_folds, n_repeats=repeat_num, random_state=random_seed)
    kfold_split = kfolds.split(y)
    res = []
    for i in range(cross_folds * repeat_num):
        time_gblup_start = time.time()
        split_index = kfold_split.__next__()
        test_index = split_index[1]
        y_test = y[test_index]
        train_index = split_index[0]
        u, beta = gblup(X, y, train_index)
        test_pre = u[test_index, :]
        time_gblup_end = time.time()
        res.append(pearsonr(y_test, np.array(test_pre).flatten())[0])
    res = np.array(res)
    print("GBLUP  accuracy: %.3f , std: %.3f" % (res.mean(), res.std()))
    return res.mean()


def calculate_multi_traits(species, trait,
                           alphas_range=(0.0001, 1), cross_folds=5, repeat_num=5, random_seed=0):
    # 先读取数据，获得k矩阵和表型数据
    time1 = time.time()
    k_matrix, ys, trait_names = load_data(species)
    # trait_index = np.where(trait_names == trait)[0][0]
    trait_index = trait
    print('---------------------Trait: {}------------------------'.format(trait_index))
    nona_index = np.where(ys[:, trait_index] >= -np.inf)[0]

    # 获取单性状准确性数据
    single_accuracy, best_alpha, single_accuracys_CV = ridge(k_matrix[nona_index, :], ys[nona_index, trait_index],
                                                                 alphas_range, cross_folds=cross_folds,
                                                                 repeat_num=repeat_num, random_seed=random_seed)

    accuracy_matrix = calculate_accuracy_matrix(k_matrix, ys, trait_index, alpha=best_alpha,
                                                        cross_folds=cross_folds,
                                                        repeat_num=repeat_num, random_seed=random_seed)
    print(accuracy_matrix)


    assistant_traits = np.where(accuracy_matrix >= single_accuracy * 1.005)[0]
    print(assistant_traits)
    time2 = time.time()
    if len(assistant_traits) == 0:
        multi_accuracy, multi_accuracys_cv = single_accuracy, single_accuracys_CV
    else:
        x, y = processing_data(trait_index, assistant_traits, k=k_matrix, ys=ys)
        multi_accuracy, multi_accuracys_cv = ridgeRC(x, y, alpha=best_alpha, random_seed=random_seed)
    time3 = time.time()
    print('Time: ', time2-time1+(time3-time2)/25)
    return single_accuracy,multi_accuracy,  single_accuracys_CV, multi_accuracys_cv



def calculate_multi_traits_random(species, trait,
                           alphas_range=(0.0001, 1), cross_folds=5, repeat_num=5, random_seed=0):
    # 先读取数据，获得k矩阵和表型数据
    k_matrix, ys, trait_names = load_data(species)
    # trait_index = np.where(trait_names == trait)[0][0]
    trait_index = trait
    print('---------------------Trait: {}------------------------'.format(trait_index))
    nona_index = np.where(ys[:, trait_index] >= -np.inf)[0]

    # 获取单性状准确性数据
    single_accuracy, best_alpha, single_accuracys_CV = ridge(k_matrix[nona_index, :], ys[nona_index, trait_index],
                                                                 alphas_range, cross_folds=cross_folds,
                                                                 repeat_num=repeat_num, random_seed=random_seed)

    traits_index = np.delete(np.arange(ys.shape[1]), trait)
    # assistant_traits = np.random.choice(traits_index, 5, replace=False,)
    assistant_traits = np.delete(np.array([0,1,2,3,4,5,6,7,8]), trait)
    print(assistant_traits)
    if len(assistant_traits) == 0:
        multi_accuracy, multi_accuracys_cv = single_accuracy, single_accuracys_CV
    else:

        x, y = processing_data(trait_index, assistant_traits, k=k_matrix, ys=ys)
        multi_accuracy, multi_accuracys_cv = ridgeRC(x, y, alpha=best_alpha, random_seed=random_seed)
    return single_accuracy,multi_accuracy,  single_accuracys_CV, multi_accuracys_cv

def calculate_multi_traits_genetic(species, trait,
                           alphas_range=(0.0001, 1), cross_folds=5, repeat_num=5, random_seed=0):
    # 先读取数据，获得k矩阵和表型数据
    genetic_matrix = pd.read_excel('data/{}/{}_genetic_correlation.xlsx'.format(species, species)).iloc[trait,1:].values
    k_matrix, ys, trait_names = load_data(species)
    # trait_index = np.where(trait_names == trait)[0][0]
    trait_index = trait
    print('---------------------Trait: {}------------------------'.format(trait_index))
    nona_index = np.where(ys[:, trait_index] >= -np.inf)[0]
    # 获取单性状准确性数据
    single_accuracy, best_alpha, single_accuracys_CV = ridge(k_matrix[nona_index, :], ys[nona_index, trait_index],
                                                                 alphas_range, cross_folds=cross_folds,
                                                                 repeat_num=repeat_num, random_seed=random_seed)
    assistant_traits_original = np.argsort(genetic_matrix)[-5:]
    assistant_traits = assistant_traits_original[np.where(genetic_matrix[ assistant_traits_original]>0.001)[0]]
    print(assistant_traits)
    if len(assistant_traits) == 0:
        multi_accuracy, multi_accuracys_cv = single_accuracy, single_accuracys_CV
    else:
        x, y = processing_data(trait_index, assistant_traits, k=k_matrix, ys=ys)
        multi_accuracy, multi_accuracys_cv = ridgeRC(x, y, alpha=best_alpha, random_seed=random_seed)
    return single_accuracy,multi_accuracy,  single_accuracys_CV, multi_accuracys_cv


def calculate_multi_traits_all(species, trait,
                           alphas_range=(0.0001, 1), cross_folds=5, repeat_num=5, random_seed=0):

    k_matrix, ys, trait_names = load_data(species)
    # trait_index = np.where(trait_names == trait)[0][0]
    trait_index = trait
    print('---------------------Trait: {}------------------------'.format(trait_index))
    nona_index = np.where(ys[:, trait_index] >= -np.inf)[0]
    # 获取单性状准确性数据
    single_accuracy, best_alpha, single_accuracys_CV = ridge(k_matrix[nona_index, :], ys[nona_index, trait_index],
                                                                 alphas_range, cross_folds=cross_folds,
                                                                 repeat_num=repeat_num, random_seed=random_seed)
    assistant_traits = np.delete(np.arange(ys.shape[1]), trait)
    np.random.shuffle(assistant_traits)
    print(assistant_traits)
    x, y = processing_data(trait_index, assistant_traits, k=k_matrix, ys=ys)
    multi_accuracy, multi_accuracys_cv = ridgeRC(x, y, alpha=best_alpha, random_seed=random_seed)
    return single_accuracy,multi_accuracy,  single_accuracys_CV, multi_accuracys_cv

def calculate_multi_traits_iter(species, trait,
                           alphas_range=(0.0001, 1), cross_folds=5, repeat_num=5, random_seed=0):
    # 先读取数据，获得k矩阵和表型数据
    time1 = time.time()
    k_matrix, ys, trait_names = load_data(species)
    # trait_index = np.where(trait_names == trait)[0][0]
    trait_index = trait
    print('---------------------Trait: {}------------------------'.format(trait_index))
    nona_index = np.where(ys[:, trait_index] >= -np.inf)[0]

    # 获取单性状准确性数据
    single_accuracy, best_alpha, single_accuracys_CV = ridge(k_matrix[nona_index, :], ys[nona_index, trait_index],
                                                                 alphas_range, cross_folds=cross_folds,
                                                                 repeat_num=repeat_num, random_seed=random_seed)

    accuracy_matrix = calculate_accuracy_matrix(k_matrix, ys, trait_index, alpha=best_alpha,
                                                        cross_folds=cross_folds,
                                                        repeat_num=repeat_num, random_seed=random_seed)
    print(accuracy_matrix)


    assistant_traits = np.where(accuracy_matrix >= single_accuracy * 1.01)[0]
    print(assistant_traits)
    time2 = time.time()
    if len(assistant_traits) == 0:
        multi_accuracy, multi_accuracys_cv = single_accuracy, single_accuracys_CV
    else:
        x, y = processing_data(trait_index, assistant_traits, k=k_matrix, ys=ys)
        multi_accuracy, multi_accuracys_cv = ridgeRC(x, y, alpha=best_alpha, random_seed=random_seed)
    time3 = time.time()
    print('Time: ', time2-time1+(time3-time2)/25)
    return single_accuracy,multi_accuracy,  single_accuracys_CV, multi_accuracys_cv
