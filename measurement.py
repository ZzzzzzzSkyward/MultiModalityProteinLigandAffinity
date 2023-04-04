from header import *


def preprocess(y_pred, y_true):
    return y_pred.squeeze(), y_true.squeeze()


def pearson(y_pred, y_true):
    return scipy.stats.pearsonr(y_pred, y_true)


def rmse(y_pred, y_true):
    mse = nn.functional.mse_loss(y_pred, y_true)
    rmse = torch.sqrt(mse)
    return rmse.item()


def tau(y_pred, y_true):
    return scipy.stats.kendalltau(y_pred, y_true)


def rho(y_pred, y_true):
    return scipy.stats.spearmanr(y_pred, y_true)


def four_of_them(y_pred, y_true):
    y_pred, y_true = preprocess(y_pred, y_true)
    y_pearson = pearson(y_pred, y_true)
    y_rmse = rmse(y_pred, y_true)
    y_tau = tau(y_pred, y_true)
    y_rho = rho(y_pred, y_true)
    return y_pearson, y_rmse, y_tau, y_rho

# TODO
# 同时计算AUPRC与AUROC
# 并且在只考虑结合位点（取接触点中的最大值）时再次计算

#TODO改写接触的AUPRC统计
def evaluate_contact(model, loader, prot_length, comp_length):
    y_pred, y_true, ind = np.zeros((len(loader.dataset), 1000, 56)), np.zeros(
        (len(loader.dataset), 1000, 56)), np.zeros(len(loader.dataset))
    batch = 0
    for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in loader:
        with torch.no_grad():
            inter, _ = model.forward_inter_affn(
                prot_data, drug_data_ver, drug_data_adj, prot_contacts)

        if batch != len(loader.dataset) // 32:
            y_true[batch * 32:(batch + 1) * 32] = prot_inter.cpu().numpy()
            y_pred[batch * 32:(batch + 1) * 32] = inter.detach().cpu().numpy()
            ind[batch * 32:(batch + 1) * 32] = prot_inter_exist.cpu().numpy()
        else:
            y_true[batch * 32:] = prot_inter.cpu().numpy()
            y_pred[batch * 32:] = inter.detach().cpu().numpy()
            ind[batch * 32:] = prot_inter_exist.cpu().numpy()
        batch += 1
    contact_au(y_pred, y_true, prot_length, comp_length, ind)


def contact_au_original(y_pred, y_true, prot_length, comp_length, ind):
    N = y_true.shape[0]
    AP = []
    AUC = []
    AP_margin = []
    AUC_margin = []
    count = 0
    for i in range(N):
        if ind[i] != 0:
            count += 1
            length_prot = int(prot_length[i])
            length_comp = int(comp_length[i])
            true_label_cut = np.asarray(y_true[i])[:length_prot, :length_comp]
            true_label = np.reshape(true_label_cut, (length_prot * length_comp))

            full_matrix = np.asarray(y_pred[i])[:length_prot, :length_comp]
            pred_label = np.reshape(full_matrix, (length_prot * length_comp))

            average_precision_whole = average_precision_score(
                true_label, pred_label)
            AP.append(average_precision_whole)
            fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label)
            roc_auc_whole = auc(fpr_whole, tpr_whole)
            AUC.append(roc_auc_whole)

            true_label = np.amax(true_label_cut, axis=1)
            pred_label = np.amax(full_matrix, axis=1)

            average_precision_whole = average_precision_score(
                true_label, pred_label)
            AP_margin.append(average_precision_whole)

            fpr_whole, tpr_whole, _ = roc_curve(true_label, pred_label)
            roc_auc_whole = auc(fpr_whole, tpr_whole)
            AUC_margin.append(roc_auc_whole)

    return np.mean(AP), np.mean(AUC), np.mean(AP_margin), np.mean(AUC_margin)


def calculate_AP_AUC(y_pred, y_true, prot_length, comp_length, ind, i):
    length_prot = int(prot_length[i])
    length_comp = int(comp_length[i])
    true_label = y_true[i][:length_prot, :length_comp]
    full_matrix = y_pred[i][:length_prot, :length_comp]

    # 计算每行的最大值
    true_label_max = np.amax(true_label, axis=1)
    full_matrix_max = np.amax(full_matrix, axis=1)

    # 计算AP和AUC
    average_precision, fpr, tpr, roc_auc = [], [], [], []
    for label, pred in [(true_label, full_matrix),
                        (true_label_max, full_matrix_max)]:
        average_precision.append(
            average_precision_score(
                label.ravel(), pred.ravel()))
        fpr_, tpr_, _ = roc_curve(label.ravel(), pred.ravel())
        fpr.append(fpr_)
        tpr.append(tpr_)
        roc_auc.append(auc(fpr_, tpr_))

    return np.array([average_precision[0], roc_auc[0],
                    average_precision[1], roc_auc[1]])


def contact_au(y_pred, y_true, protein_length, compound_length, ind):
    # 类型转换为整型
    protein_length = protein_length.astype(int)
    compound_length = compound_length.astype(int)

    # 创建进程池
    pool = multiprocessing.Pool()

    # 并行计算AP和AUC
    length = y_true.shape[0]
    AAAA = np.empty((length, 4))
    for i in range(length):
        if np.any(ind[i] != 0):
            AAAA[i] = calculate_AP_AUC(
                y_pred,
                y_true,
                protein_length,
                compound_length,
                ind,
                i)

    # 关闭进程池
    pool.close()
    pool.join()

    # 返回平均值
    AAAA_mean = np.mean(AAAA, axis=0)
    # return AAAA_mean[0], AAAA_mean[1], AAAA_mean[2], AAAA_mean[3]
    return AAAA_mean
