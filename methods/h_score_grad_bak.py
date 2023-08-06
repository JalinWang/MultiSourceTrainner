from typing import List
import numpy as np
import sys
import copy
from tqdm import tqdm



#####  H-score definition start
def getCov(X):
    X_mean = X-np.mean(X, axis=0, keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1)
    return cov


def get_score(features, labels):
    #Z=np.argmax(Z, axis=1)
    Covf = getCov(features)
    alphabetZ = list(set(labels))
    g = np.zeros_like(features)
    for z in alphabetZ:
        l = labels == z
        fl = features[labels == z, :]
        # conditional expectation
        Ef_z = np.mean(fl, axis=0)
        g[labels == z] = Ef_z

    Covg = getCov(g)
    dif = np.trace(Covg) / np.trace(Covf)
    return dif


def simple_Hscore(alpha, features, labels):

    target_feature = np.array([alpha[i]* features[i] for i in range(len(alpha))]).sum(axis=0)
    hscore = get_score(target_feature, labels.cpu().detach().numpy())

    return hscore
#####  H-score definition ends


# maximinze `f` func given `alpha`
# epsilon: len of the interval for gradient calc
def maximize_f(f, alpha, lr=0.001, epsilon = 0.0001, num_iters = 100):
    score = np.zeros(num_iters)
    for i in tqdm(range(num_iters)):
        grad = np.zeros_like(alpha)
        for j in range(len(alpha)):
            a_plus, a_minus = np.copy(alpha), np.copy(alpha)
            a_plus[j] += epsilon
            a_minus[j] -= epsilon
            grad[j] = (f(a_plus)-f(a_minus))/(2*epsilon)
        alpha += lr*grad
        alpha[alpha < 0] = 0
        alpha = alpha / alpha.sum() if alpha.sum() > 1 else alpha
        print(alpha)
        score[i] = f(alpha)

    return alpha, score


def update_alpha(features,):
    n_s_tasks = len(s_id)

    f = lambda a: simple_Hscore(a, features, labels)
    alpha_opt, hscore_curve = maximize_f(f, np.ones(n_s_tasks) / (n_s_tasks), lr=lr)
    return alpha_opt, hscore_curve



# if __name__ == '__main__':
#     lr_list = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
#     save_path = '/home/viki/Codes/MultiSource/3/multi_source_exp/MultiSourceExp/alpha/'
#     for lr in lr_list:
#         a,s = update_alpha(0, list(range(1,21)), lr=lr, include_target=True)
#         np.savetxt(save_path+'simple_hscore_grad_lr='+str(lr)+'_alpha.npy', a)
#         np.savetxt(save_path+'simple_hscore_grad_lr='+str(lr)+'_scorelist.npy', s)
#     # from matplotlib import pyplot as plt
#     # plt.plot(s)


def validate_given_alpha(alpha):

    g_hat = get_g(alpha, target_feature)
    # g_rand = np.random.random(g_y_hat.shape)
    print(f"acc: {get_accuracy(g_hat, g_y_hat)}")


def normalize(f):
    e_f = f.mean(axis=0)
    n_f = f - e_f
    return n_f



def get_g(feature, label, alpha):
    # g = (1-alpha) * g + alpha * g_s

    # expectation and normalization of f and g
    feature = normalize(feature)

    gamma_f = feature.T.dot(feature) / feature.shape[0]

    
    ce_f_s = np.array([get_conditional_exp(
        images=target_images,
        label=target_labels,
        feature=model_f_i(target_images)
        ) for i in range(self.n_source)])

    g_y_hat = np.linalg.inv(gamma_f).dot((ce_f_s.transpose(1,2,0).dot(alpha)).T).T        


    return g_y_hat


def get_accuracy(gc):
    "classification accuracy with different gy"
    acc = 0
    total = 0
    for images, labels in test_data:
        labels= labels.numpy()
        fc=model_f(Variable(images).to(device)).data.cpu().numpy()

        fc_normalized=fc-np.sum(fc,axis=0)/fc.shape[0]
        
        gc_normalized=gc-np.sum(gc,axis=0)/n_label

        score=np.dot(fc_normalized,gc_normalized.T) # bs * bs
        acc += (np.argmax(score, axis = 1) == labels).sum()

        total += len(images)
    acc = float(acc) / total
    return acc