from header import *


def _preprocess(y_pred, y_true, cpu=False):
    y_pred, y_true = y_pred.detach(), y_true.detach()
    y_pred, y_true = y_pred.squeeze(), y_true.squeeze()
    if cpu:
        y_pred, y_true = y_pred.cpu().numpy(), y_true.cpu().numpy()
    return y_pred, y_true


def _tocpu(y_pred, y_true):
    y_pred, y_true = y_pred.cpu().numpy(), y_true.cpu().numpy()
    return y_pred, y_true


def getloss(l):
    return l.detach().cpu().numpy()


def pearson(y_pred, y_true):
    return scipy.stats.pearsonr(y_pred, y_true)


def rmse_torch(y_pred, y_true):
    mse = F.mse_loss(y_pred, y_true)
    rmse = torch.sqrt(mse)
    # .item会自动把数据搬回CPU
    return rmse.item()


def rmse(y_pred, y_true):
    mse = np.mean((y_pred - y_true)**2)
    rmse = np.sqrt(mse)
    return rmse


def tau(y_pred, y_true):
    return scipy.stats.kendalltau(y_pred, y_true)


def rho(y_pred, y_true):
    return scipy.stats.spearmanr(y_pred, y_true)


def four_of_them(y_pred, y_true):
    #y_pred, y_true = _preprocess(y_pred, y_true)
    y_rmse = rmse(y_pred, y_true)
    # 以下计算需要搬回CPU上做，因为numpy不支持GPU
    #默认已经搬回来了
    y_pred, y_true = _tocpu(y_pred, y_true)
    y_pearson = pearson(y_pred, y_true)
    y_tau = tau(y_pred, y_true)
    y_rho = rho(y_pred, y_true)
    return y_pearson, y_rmse, y_tau, y_rho


def evaluate_affinity(model, loader):
    device = next(model.parameters()).device
    val_list = []
    pred_list = []
    with torch.no_grad():
        for batch in loader:
            # 将数据移动到指定设备
            batch = [item.to(device) for item in batch]
            # 获取输入数据和目标数据
            inputs, targets = batch[:-1], batch[-1]
            # 计算模型的预测结果
            outputs = model(*inputs)
            # 将预测结果和目标数据转移到cpu上，并将tensor转换为numpy数组
            pred, val = _preprocess(outputs, targets, True)
            # 将预测结果和目标数据添加到列表中
            pred_list.append(pred)
            val_list.append(val)
    # 将所有验证集和预测结果合并为一个大的数组
    val_list = np.concatenate(val_list, axis=0)
    pred_list = np.concatenate(pred_list, axis=0)
    return four_of_them(pred_list, val_list)


# TODO改写接触的AUPRC统计


def evaluate_contact(model, loader, prot_length, comp_length):
    y_pred, y_true, ind = np.zeros((len(loader.dataset), 1000, 56)), np.zeros(
        (len(loader.dataset), 1000, 56)), np.zeros(len(loader.dataset))
    batch = 0
    for prot_data, drug_data_ver, drug_data_adj, prot_contacts, prot_inter, prot_inter_exist, label in loader:
        with torch.no_grad():
            inter, _ = model.forward_inter_affn(prot_data, drug_data_ver,
                                                drug_data_adj, prot_contacts)

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
            true_label = np.reshape(true_label_cut,
                                    (length_prot * length_comp))

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


def calculate_AP_AUC(args):
    y_pred, y_true, protein_length, compound_length = args
    y_pred, y_true = _preprocess(y_pred, y_true, cpu=True)
    # 提取出标签
    true_label = y_true[:protein_length, :compound_length]
    full_matrix = y_pred[:protein_length, :compound_length]

    # 计算每行的最大值
    true_label_max = np.amax(true_label, axis=1)
    full_matrix_max = np.amax(full_matrix, axis=1)

    # 计算AP和AUC
    average_precision, fpr, tpr, roc_auc = [], [], [], []
    for label, pred in [(true_label, full_matrix),
                        (true_label_max, full_matrix_max)]:
        average_precision.append(
            average_precision_score(label.ravel(), pred.ravel()))
        fpr_, tpr_, _ = roc_curve(label.ravel(), pred.ravel())
        fpr.append(fpr_)
        tpr.append(tpr_)
        roc_auc.append(auc(fpr_, tpr_))

    return np.array(
        [average_precision[0], roc_auc[0], average_precision[1], roc_auc[1]])


cpu = multiprocessing.cpu_count()


def contact_au(y_pred, y_true, protein_length, compound_length, ind):
    # 类型转换为整型
    protein_length = protein_length.astype(int)
    compound_length = compound_length.astype(int)

    # 创建进程池
    global cpu
    length = y_true.shape[0]
    num_processes = min(length, cpu)
    pool = multiprocessing.Pool(processes=num_processes)

    # 并行计算AP和AUC
    AAAA = pool.map(
        calculate_AP_AUC,
        [(y_pred, y_true, int(protein_length), int(compound_length))
         for i in range(length) if ind[i] != 0],
        chunksize=1)

    # 关闭进程池
    pool.close()
    pool.join()

    # 返回平均值
    AAAA_mean = np.mean(np.array(AAAA), axis=0)
    # return AAAA_mean[0], AAAA_mean[1], AAAA_mean[2], AAAA_mean[3]
    return AAAA_mean
