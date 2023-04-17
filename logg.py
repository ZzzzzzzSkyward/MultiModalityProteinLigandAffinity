from header import *


def calculate_loss_slope(loss_values):
    """Calculates the slope of the linear regression line of a given list of loss values."""

    # Create a numpy array of the loss values
    y = np.array(loss_values)

    # Create an array of the x values (epochs, iterations, etc.)
    x = np.arange(len(loss_values))

    # Calculate the slope of the linear regression line using numpy's polyfit
    # function
    slope = np.polyfit(x, y, 1)[0]

    return slope


# create logger
logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)

_logdate = '%m.%d.%H:%M:%S'
# create formatter
formatter = logging.Formatter(
    '%(asctime)s[%(levelname)s]%(message)s',
    datefmt=_logdate)

screen_length = 10
loss_val = []
average_loss_val = []
delta_loss_val = []
positive_loss_count = []
net_loss_sum = []


def tofile(path="log.txt"):
    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def toscreen():
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # add formatter to ch
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def stringify_float_array(arr):
    cat = []
    fmt = '{:.4f}'
    for num in arr:
        cat.append(fmt.format(num))
    return '[' + ",".join(cat) + ']'


def loss(epoch, batch, loss):
    logger.info(f"{epoch}.{batch}.loss={loss:.3f}")


def epochloss2(epoch, trainloss, testloss):
    logger.info(
        f"{epoch}:train.loss={trainloss:.3f},test.loss={testloss:.3f}")


def addloss(loss):
    loss_val.append(loss)
    monitor_loss()


def addloss4(p, r, t, h):
    log(f'pearson={p[0]:.4f},rmse={r:.2f},tau={t[0]:.4f},rho=[{h[0]:.4f}')
    if p[1] > 0.01 or r[1] > 0.01 or t[1] > 0.01:
        log(f'pearsonp={p[1]:.4f},taup={r[1]:.4f},rhop=[{h[1]:.4f}])')


def predictloss():
    logger.info("predict.loss.k=%.5f".format(calculate_loss_slope(loss_val)))

# this can only be called after addloss


def monitor_loss():
    l = len(loss_val)
    global screen_length
    sl = screen_length
    if sl > l:
        sl = l
    average_loss = sum(loss_val[l - sl:]) / sl
    last_average_loss = average_loss
    if len(average_loss_val) > 0:
        last_average_loss = average_loss_val[l - 2]
    average_loss_val.append(average_loss)
    delta_loss_val.append(last_average_loss - average_loss)
    positive_loss_count.append(
        sum([i for i in delta_loss_val[l - sl:] if i > 0]))
    net_loss_sum.append(average_loss_val[l - sl] - average_loss_val[l - 1])

# this can only be called upon monitor


def evaluate_loss():
    l = len(loss_val) - 1
    this_loss = loss_val[l]
    max_loss = max(loss_val[-screen_length * 10:])
    min_loss = min(loss_val[-screen_length * 10:])
    loss_range = max(max_loss - min_loss, 0.001)
    pct_loss = this_loss / loss_range * 100
    pct_delta_loss = sum(delta_loss_val[-screen_length:]
                         ) / max(1, min(screen_length, len(delta_loss_val))) / loss_range
    recent_loss_count = positive_loss_count[l]
    return pct_loss, pct_delta_loss, recent_loss_count


def decide_terminate():
    loss, dloss, count = evaluate_loss()
    if dloss < 0 and loss > 20:
        return True
    if dloss < 0 and count > screen_length / 2:
        return True
    return False


def log(*args):
    # 将所有参数转换为字符串，并拼接成一个大的字符串
    msg = ' '.join(str(arg) for arg in args)
    # 将拼接后的字符串作为参数传递给 logger.info
    logger.info(msg)


def warn(*args):
    logger.warning(*args)


if __name__ == '__main__':
    tofile()
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')
    from loss_data import *
    for i in loss_data:
        addloss(i)
        monitor_loss()
    logger.info(stringify_float_array(average_loss_val))
    logger.info(stringify_float_array(delta_loss_val))
    logger.info(stringify_float_array(net_loss_sum))
