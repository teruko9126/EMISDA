import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from kmeans_pytorch import kmeans

C = 2
K = 2
N = 500
A = 2
ans_K = 2
split_num = 10

#!10class,3cluster,50000num,64dimで収束するのかを確かめる！
# *5class,3cluster,10000num,24dimでできているので暫定的にOKとする！

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64

pi_ans = torch.rand(C, ans_K)
pi_ans = (pi_ans/(torch.sum(pi_ans, dim=1).expand(ans_K, C).T))

Ave_ans = torch.rand(C, ans_K, A) * 10

Cov_ans = torch.eye(A).expand(C, ans_K, A, A).clone() + torch.rand(1)

data_not_shuffle = torch.zeros(C, N//C, A)


# 共分散行列作成->piに従ってデータの生成を行う
for c in range(C):
    data_count = 0
    for k in range(ans_K):
        MultiVariate = torch.distributions.multivariate_normal.MultivariateNormal(
            Ave_ans[c][k], Cov_ans[c][k])
        data_num = ((N//C) * pi_ans[c][k]).to(torch.int64)
        for i in range(data_num):
            data_not_shuffle[c][data_count] = MultiVariate.sample()
            data_count += 1
    while data_count != (N//C):
        print(data_count)
        print("data plus!!")
        data_not_shuffle[c][data_count] = MultiVariate.sample()
        data_count += 1

data = data_not_shuffle.reshape(N, A)

labels = torch.zeros(N//C, 1)

for c in range(C-1):
    temp_labels = torch.ones(N//C, 1) * (c + 1)
    labels = torch.cat((labels, temp_labels), 0)

tensor_label_all = torch.cat((data, labels), 1)

idx = torch.randperm(N)
tensor_label_all = tensor_label_all[idx]

tensor_all = tensor_label_all[:, :A]
labels_all = tensor_label_all[:, A].to(torch.int64)

tensor_all = torch.tensor_split(tensor_all, split_num)
labels_all = torch.tensor_split(labels_all, split_num)

pi_off = torch.ones((C, K)) / K
Ave_off = torch.zeros(C, K, A)
Cov_off = torch.eye(A).expand(C, K, A, A).clone()


def calc_cov(X, cluster_ids_x, cluster_centers):
    ans_cov = torch.zeros(K, A, A)
    cluster_datanum = torch.zeros(K,)
    for i in range(X.shape[0]):
        ans_cov[cluster_ids_x[i]] += torch.einsum(
            'k,d->kd', X[i] - cluster_centers[cluster_ids_x[i]], X[i] - cluster_centers[cluster_ids_x[i]])
        # print(X[i] - cluster_centers[cluster_ids_x[i]])
        cluster_datanum[cluster_ids_x[i]] += 1
    for k in range(K):
        ans_cov[k] = ans_cov[k] / (cluster_datanum[k] - 1)

    return ans_cov, cluster_datanum


# print(data_not_shuffle.shape)
for c in range(C):
    if K == 1:
        cluster_ids_x = torch.zeros((N//C),).long()
        cluster_centers = torch.zeros(1, A)
        # print(cluster_ids_x)
        cluster_centers[0] = torch.sum(
            data_not_shuffle[c], dim=0) / (N//C)
        cluster_covariaces, cluster_datanum = calc_cov(
            data_not_shuffle[c], cluster_ids_x, cluster_centers)
        # print(cluster_datanum)
    else:
        cluster_ids_x, cluster_centers = kmeans(
            X=data_not_shuffle[c], num_clusters=K, distance='euclidean')
        cluster_covariaces, cluster_datanum = calc_cov(
            data_not_shuffle[c], cluster_ids_x, cluster_centers)
    Ave_off[c] = cluster_centers
    Cov_off[c] = cluster_covariaces
    pi_off[c] = cluster_datanum/(cluster_datanum.sum(axis=0, keepdims=True))


class GaussianMixture:

    def __init__(self, num_cluster, input_size, class_num, pi, mu, sigma, max_iter=50):
        self.num_cluster = num_cluster
        self.input_size = input_size
        self.max_iter = max_iter
        self.class_num = class_num

        self.pi = pi
        self.mu = mu
        self.sigma = sigma

    def calculate_nk(self, q, label):
        n = q.shape[0]
        c = self.class_num
        k = self.num_cluster

        label = label.reshape(n)
        onehot = torch.zeros(n, c)
        onehot[torch.arange(n), label] = 1

        one_for_q = torch.ones(n, 1)

        q = torch.cat((q, one_for_q), dim=1)

        q = torch.einsum('nk,nc->nck', q, onehot)

        nk_withclassnum = torch.sum(q, dim=0)

        nk = nk_withclassnum[:, :k]
        class_num = nk_withclassnum[:, k].reshape(c, 1)

        return nk, class_num

    def calculate_mu(self, X, label, q):
        n = X.shape[0]
        c = self.class_num

        mu = torch.einsum('nd,nk->nkd', X, q)

        label = label.reshape(n)
        onehot = torch.zeros(n, c)
        onehot[torch.arange(n), label] = 1

        mu = torch.einsum('nkd,nc->nckd', mu,
                          onehot)

        mu = torch.sum(mu, dim=0)

        return mu

    def calculate_sigma(self, X, label, q, mu):
        n = X.shape[0]
        c = self.class_num
        k = self.num_cluster
        d = self.input_size

        mu = mu[label]

        error = (X.reshape(n, 1, d).expand(
            n, k, d).clone() - mu)

        q += 10e-30

        sigma = torch.einsum('nk,nkd,nke->nkde', q,
                             error, error).reshape(n,
                                                   1, k, d, d).expand(n, c, k, d, d).clone()

        label = label.reshape(n)
        onehot = torch.zeros(n, c)
        onehot[torch.arange(n), label] = 1

        sigma = torch.einsum('nckde,nc->nckde', sigma,
                             onehot)

        sigma = torch.sum(sigma, dim=0)

        return sigma

    def fit(self, X, label):
        C = self.class_num
        K = self.num_cluster
        D = self.input_size

        self.plot()
        print("em start")
        print("update_sigma", self.sigma)
        print(Cov_ans)
        prev_elbo = np.inf
        flag_end = False
        for i in range(self.max_iter):
            mu = torch.zeros(C, K, D)
            sigma = torch.zeros(C, K, D, D)
            nk = torch.zeros(C, K)
            class_sum = torch.zeros(C, 1)
            elbo = 0
            for j in range(split_num):

                q = self._expect(X[j], label[j])

                tmp_nk, tmp_class_sum = self.calculate_nk(q, label[j])

                nk += tmp_nk
                class_sum += tmp_class_sum

                tmp_mu = self.calculate_mu(X[j], label[j], q)
                tmp_sigma = self.calculate_sigma(X[j], label[j], q, self.mu)
                mu += tmp_mu
                sigma += tmp_sigma
                elbo += self.compute_elbo(X[j], label[j], q)

            class_sum = (1 / class_sum).reshape(C)
            self.pi = torch.einsum("ck,c->ck", nk, class_sum)

            nk = 1 / (nk + 1e-20)
            self.sigma = torch.einsum("ckdh,ck->ckdh", sigma, nk)
            self.mu = torch.einsum("ckd,ck->ckd", mu, nk)

            elbo = elbo.cpu()
            self.plot()
            if np.allclose(elbo, prev_elbo):
                self.last_iter = i + 1
                flag_end = True
                break
            prev_elbo = elbo
            print(i)

    def compute_elbo(self, X, label, q):
        n = X.shape[0]
        k = self.num_cluster
        d = self.input_size

        q += 10e-20

        label = label.reshape(n)
        pi = self.pi[label]
        mu = self.mu[label]
        sigma = self.sigma[label]

        X = X.reshape((n, 1, d)).expand(n, k, d).clone()
        mu = mu.reshape((n, k, d))
        inv_sigma = torch.inverse(sigma)

        logdet_sigma = torch.linalg.slogdet(sigma)[1].reshape(n, 1, k)

        distance = torch.einsum('nkd,nkde,nke->nk', X - mu, inv_sigma, X - mu)
        constants = -d/2*torch.log(torch.tensor(2*np.pi))
        log_p_x_given_t = constants - logdet_sigma / \
            2 - distance.reshape((n, k))/2
        log_pi = torch.log(pi).reshape((n, 1, k))
        log_q = torch.log(q)

        return ((log_pi + log_p_x_given_t - log_q) * q).sum()

    def _expect(self, X, label):
        n = X.shape[0]
        k = self.num_cluster
        d = self.input_size

        # 各データごとにラベルに対応した平均、分散、寄与率を用意
        label = label.reshape(n)
        # print(self.mu)
        pi = self.pi[label]
        mu = self.mu[label]
        sigma = self.sigma[label]

        X = X.reshape((n, 1, d)).expand(n, k, d).clone()
        mu = mu.reshape((n, k, d))

        inv_sigma = torch.inverse(sigma)
        logdet_sigma = torch.linalg.slogdet(sigma)[1]

        distance = torch.einsum('nkd,nkde,nke->nk', X - mu, inv_sigma, X - mu)

        p_x_given_t = - (distance - d *
                         torch.log(torch.tensor(2 * np.pi)) + logdet_sigma) / 2

        p_x_given_t_max = torch.max(p_x_given_t, 1)
        p_x_given_t = torch.nn.functional.one_hot(
            p_x_given_t_max[1], num_classes=k)

        q_unnormalized = torch.einsum('nk,nk->nk', p_x_given_t, pi)

        return q_unnormalized/(q_unnormalized.sum(axis=1, keepdims=True) + 1e-30)

    def _maximize(self, X, label, q):
        k = self.num_cluster
        c = self.class_num

        nk, class_num = self.calculate_nk(q, label)

        mu = self.calculate_mu(X, label, q, nk)

        sigma = self.calculate_sigma(X, label, q, nk, mu)

        class_num = class_num.repeat_interleave(k, dim=1)
        nk = nk.reshape((c, k))
        pi = torch.div(nk, class_num).reshape((c, k))

        return pi, mu, sigma

    def predict(self, X):
        return self._expect(X)

    def plot(self):
        counter_size = 40
        Z = torch.zeros(counter_size, counter_size)
        data_color = ['Reds', 'Blues', 'Greens', 'Purples']
        data_color_plot = ['red', 'blue', 'green', 'purple']

        for c in range(C):
            plt.scatter(data_not_shuffle[c, :, 0],
                        data_not_shuffle[c, :, 1], c=data_color_plot[c], marker='o')

        count = 0
        x = np.linspace(-3, 11, counter_size)
        y = np.linspace(-3, 11, counter_size)
        XXX, YYY = np.meshgrid(x, y)
        mu_forplot = self.mu.to(torch.double)
        sigma_forplot = self.sigma.to(torch.double)
        for c in range(C):
            Z = torch.zeros(counter_size, counter_size)
            for k in range(K):
                MultiVariate = torch.distributions.multivariate_normal.MultivariateNormal(
                    mu_forplot[c][k], sigma_forplot[c][k])

                for i in range(counter_size):
                    for j in range(counter_size):
                        Z[i, j] += self.pi[c][k] * \
                            (pow(math.e, MultiVariate.log_prob(
                                torch.tensor([XXX[i, j], YYY[i, j]]))))

            plt.contour(XXX, YYY, Z, 10, cmap=data_color[count])
            count += 1
        plt.pause(1)
        plt.cla()


gmm = GaussianMixture(num_cluster=K, input_size=A, class_num=C,
                      pi=pi_off, mu=Ave_off, sigma=Cov_off)
gmm.fit(tensor_all, labels_all)
