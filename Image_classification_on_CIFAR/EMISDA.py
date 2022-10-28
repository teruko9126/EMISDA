import torch
import torch.nn as nn
import numpy as np
import time
from kmeans_pytorch import kmeans


class EstimatorCV_EM():
    def __init__(self, input_size, class_num, num_cluster,  max_iter=50):

        self.input_size = input_size
        self.class_num = class_num
        self.num_cluster = num_cluster
        self.max_iter = max_iter
        self.last_iter = 0

        self.pi = torch.ones((class_num, num_cluster)) / num_cluster
        self.mu = torch.zeros(class_num, num_cluster, input_size)
        self.sigma = torch.eye(input_size).expand(
            class_num, num_cluster, input_size, input_size).clone()

    def calc_kmeans_cov(self, X, cluster_ids_x, cluster_centers):
        K = self.num_cluster
        D = self.input_size
        ans_cov = torch.zeros(K, D, D).cuda()
        cluster_datanum = torch.zeros(K,).cuda()
        for i in range(X.shape[0]):
            ans_cov[cluster_ids_x[i]] += torch.einsum(
                'k,d->kd', X[i] - cluster_centers[cluster_ids_x[i]], X[i] - cluster_centers[cluster_ids_x[i]])
            cluster_datanum[cluster_ids_x[i]] += 1

        for k in range(K):
            ans_cov[k] = ans_cov[k] / (cluster_datanum[k] - 1)

        return ans_cov, cluster_datanum

    def fit(self, X, label, epoch, split_num, reset_each_epoch):
        C = self.class_num
        K = self.num_cluster
        D = self.input_size

        onehot_label_of_x = torch.nn.functional.one_hot(
            label, num_classes=C).cuda()
        data_num_each_label = onehot_label_of_x.sum(dim=0)
        X_for_kmeans = torch.einsum(
            'nd,nc -> ncd', X, onehot_label_of_x).cuda()

        if epoch == 0 or reset_each_epoch:
            for c in range(C):
                data_of_class_c = X_for_kmeans[:, c, :]
                data_of_class_c = data_of_class_c[torch.sum(data_of_class_c, dim=1).sort(
                    descending=True)[1]][:data_num_each_label[c], :].cuda()
                if K == 1:
                    cluster_ids_x = torch.zeros(data_num_each_label[c],).long()
                    cluster_centers = torch.sum(
                        data_of_class_c, dim=0) / data_num_each_label[c]
                    cluster_covariaces, cluster_datanum = self.calc_kmeans_cov(
                        data_of_class_c, cluster_ids_x, cluster_centers)
                else:
                    cluster_ids_x, cluster_centers = kmeans(
                        X=data_of_class_c, num_clusters=K, distance='euclidean', device=torch.device('cuda:0'))
                    cluster_ids_x = cluster_ids_x.cuda()
                    cluster_centers = cluster_centers.cuda()
                    cluster_covariaces, cluster_datanum = self.calc_kmeans_cov(
                        data_of_class_c, cluster_ids_x, cluster_centers)
                self.mu[c] = cluster_centers
                self.sigma[c] = cluster_covariaces
                self.pi[c] = cluster_datanum / \
                    (cluster_datanum.sum(axis=0, keepdims=True))
        X = torch.tensor_split(X, split_num)
        label = torch.tensor_split(label, split_num)

        prev_elbo = np.inf
        flag_end = False
        for i in range(self.max_iter):
            mu = torch.zeros(C, K, D).cuda()
            sigma = torch.zeros(C, K, D, D).cuda()
            nk = torch.zeros(C, K).cuda()
            class_sum = torch.zeros(C, 1).cuda()
            elbo = 0
            for j in range(split_num):

                q = self._expect(X[j], label[j])

                tmp_nk, tmp_class_sum = self.calculate_nk(q, label[j])

                nk += tmp_nk.cuda()
                class_sum += tmp_class_sum.cuda()

                tmp_mu = self.calculate_mu(X[j], label[j], q)
                tmp_sigma = self.calculate_sigma(X[j], label[j], q, self.mu)
                mu += tmp_mu.cuda()
                sigma += tmp_sigma.cuda()
                elbo += self.compute_elbo(X[j], label[j], q)

            class_sum = (1 / class_sum).reshape(C)
            self.pi = torch.einsum("ck,c->ck", nk, class_sum)

            nk = 1 / (nk + 1e-20)
            self.sigma = torch.einsum("ckdh,ck->ckdh", sigma, nk)
            self.mu = torch.einsum("ckd,ck->ckd", mu, nk)

            elbo = elbo.cpu()
            if np.allclose(elbo, prev_elbo):
                self.last_iter = i + 1
                flag_end = True
                break
            prev_elbo = elbo

        if not flag_end:
            self.last_iter = self.max_iter

    def _expect(self, X, label):
        n = X.shape[0]
        k = self.num_cluster
        d = self.input_size

        label = label.reshape(n).cpu()
        pi = self.pi[label].cuda()
        mu = self.mu[label].cuda()
        sigma = self.sigma[label].cuda()

        X = X.reshape((n, 1, d)).expand(n, k, d).clone()
        mu = mu.reshape((n, k, d)).cuda()

        inv_sigma = torch.inverse(sigma)
        logdet_sigma = torch.linalg.slogdet(sigma)[1].cuda()

        distance = torch.einsum('nkd,nkde,nke->nk', X - mu, inv_sigma, X - mu)

        p_x_given_t = - (distance - d *
                         torch.log(torch.tensor(2 * np.pi)) + logdet_sigma) / 2

        p_x_given_t_max = torch.max(p_x_given_t, 1)
        p_x_given_t = torch.nn.functional.one_hot(
            p_x_given_t_max[1], num_classes=k)

        q_unnormalized = torch.einsum('nk,nk->nk', p_x_given_t, pi)

        return q_unnormalized/(q_unnormalized.sum(axis=1, keepdims=True) + 1e-30)

    def calculate_nk(self, q, label):
        n = q.shape[0]
        c = self.class_num
        k = self.num_cluster

        label = label.reshape(n)
        onehot = torch.zeros(n, c).cuda()
        onehot[torch.arange(n), label] = 1

        one_for_q = torch.ones(n, 1).cuda()

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
        onehot = torch.zeros(n, c).cuda()
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

        mu = mu[label].cuda()

        error = (X.reshape(n, 1, d).expand(
            n, k, d).clone() - mu)

        q += 10e-30

        sigma = torch.einsum('nk,nkd,nke->nkde', q,
                             error, error).reshape(n,
                                                   1, k, d, d).expand(n, c, k, d, d).clone()

        label = label.reshape(n)
        onehot = torch.zeros(n, c).cuda()
        onehot[torch.arange(n), label] = 1

        sigma = torch.einsum('nckde,nc->nckde', sigma,
                             onehot)

        sigma = torch.sum(sigma, dim=0)

        return sigma

    def compute_elbo(self, X, label, q):
        n = X.shape[0]
        k = self.num_cluster
        d = self.input_size

        q += 10e-20

        label = label.reshape(n)
        pi = self.pi[label].cuda()
        mu = self.mu[label].cuda()
        sigma = self.sigma[label].cuda()

        X = X.reshape((n, 1, d)).expand(n, k, d).clone().cuda()
        mu = mu.reshape((n, k, d)).cuda()
        inv_sigma = torch.inverse(sigma).cuda()

        logdet_sigma = torch.linalg.slogdet(sigma)[1].reshape(n, 1, k)

        distance = torch.einsum('nkd,nkde,nke->nk', X - mu, inv_sigma, X - mu)
        constants = -d/2*torch.log(torch.tensor(2*np.pi))
        log_p_x_given_t = constants - logdet_sigma / \
            2 - distance.reshape((n, k))/2
        log_pi = torch.log(pi).reshape((n, 1, k))
        log_q = torch.log(q)

        return ((log_pi + log_p_x_given_t - log_q) * q).sum()


class ISDALoss_EM(nn.Module):
    def __init__(self, feature_num, class_num, K):
        super(ISDALoss_EM, self).__init__()

        self.estimator = EstimatorCV_EM(feature_num, class_num, K)

        self.class_num = class_num

        self.cross_entropy = nn.CrossEntropyLoss()
        self.K = K

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):

        N = features.size(0)
        C = self.class_num
        A = features.size(1)
        K = self.K
        cv_matrix = cv_matrix.cuda()

        weight_m = list(fc.parameters())[0]

        NxW_ij = weight_m.expand(N, C, A).to(torch.float64)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A)).to(torch.float64)

        q = self.estimator._expect(features, labels)

        cv_matrix = cv_matrix[labels].to(torch.float64)

        q_max = torch.max(q, 1)
        onehot_q = torch.nn.functional.one_hot(
            q_max[1], num_classes=K).to(torch.float64)

        cv_matrix_for_aug = torch.einsum('nkde,nk->nde', cv_matrix,
                                         onehot_q)

        CV_temp = cv_matrix_for_aug.view(N, A, A)

        sigma2 = ratio * \
            torch.bmm(torch.bmm(NxW_ij - NxW_kj,
                                CV_temp),
                      (NxW_ij - NxW_kj).permute(0, 2, 1))

        sigma2 = sigma2.mul(torch.eye(C)
                            .expand(N, C, C).cuda()).sum(2).view(N, C)

        aug_result = y + 0.5 * sigma2

        return aug_result

    def forward(self, model, fc, x, target_x, ratio):

        features = model(x)

        y = fc(features)

        loss = 0

        isda_aug_y = self.isda_aug(
            fc, features, y, target_x,  self.estimator.sigma.detach(), ratio)

        loss += self.cross_entropy(isda_aug_y, target_x)

        return loss, y

    def adjust_em(self, model, train_loader, epoch, reset_each_epoch):

        split_num = 20
        for i, (data, label) in enumerate(train_loader):
            with torch.no_grad():
                feature = model(data)
            if i == 0:
                features_all = feature
                labels_all = label
            else:
                features_all = torch.cat((features_all, feature), 0)
                labels_all = torch.cat((labels_all, label), 0)

        self.estimator.fit(features_all, labels_all, epoch,
                           split_num, reset_each_epoch)
