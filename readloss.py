import csv


def read_all_losses_from_log(log_file_path):
    train_losses, test_losses = [], []
    with open(log_file_path, 'r',encoding='utf-8') as f:
        for line in f:
            if 'train.loss' in line and 'test.loss' in line:
                # 提取训练和测试损失
                line_parts = line.split(',')
                for part in line_parts:
                    if 'train.loss' in part:
                        train_losses.append(float(part.split('=')[1]))
                    elif 'test.loss' in part:
                        test_losses.append(float(part.split('=')[1]))
    # 返回训练和测试损失列表
    return train_losses, test_losses


def save_losses_to_csv(train_losses, test_losses, csv_file_path):
    with open(csv_file_path, 'w', newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Train Losses', 'Test Losses'])
        for i in range(len(train_losses)):
            writer.writerow([train_losses[i], test_losses[i]])
    print(f'Losses saved to {csv_file_path}.')


def read(path):
    tr, te = read_all_losses_from_log(path)
    save_losses_to_csv(tr, te, "loss.csv")


if __name__ == '__main__':
    try:
        read("C:/Users/Zhu/Desktop/fsdownload/log.txt")
    except:
        read("C:/Users/Administrator/Desktop/fsdownload/log.txt")
