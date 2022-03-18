import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# data config (all methods)
DATA_PATH = './RetailRocket/raw/'
DATA_PATH_PROCESSED = './RetailRocket/prepared/'
DATA_FILE = 'event1'
SESSION_LENGTH = 30 * 60  # 30 minutes  # 一个会话中相邻两个交互的最大时间间隔

# filtering config (all methods)
MIN_SESSION_LENGTH = 2  # 会话的最小长度
MAX_SESSION_LENGTH = 50 # 会话最大长度
MIN_ITEM_SUPPORT = 5    # 物品的最小交互数

# min date config
MIN_DATE = '2000-04-01'

# days test default config
DAYS_TEST = 1

# slicing default config
NUM_SLICES = 1
DAYS_OFFSET = 0
DAYS_SHIFT = 5
DAYS_TRAIN = 131    # 训练数据包含的天数

DAYS_TEST = 7   # 测试数据包含的天数


# preprocessing from original gru4rec
def preprocess_org(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT,
                   min_session_length=MIN_SESSION_LENGTH, max_session_length=MAX_SESSION_LENGTH):
    data, buys = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length, max_session_length)
    split_data_org(data, path_proc + file)


# preprocessing from original gru4rec but from a certain point in time
def preprocess_org_min_date(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED,
                            min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                            max_session_length=MAX_SESSION_LENGTH,
                            min_date=MIN_DATE):
    data, buys = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length, max_session_length)
    data = filter_min_date(data, min_date)
    split_data_org(data, path_proc + file)


# preprocessing adapted from original gru4rec
def preprocess_days_test(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED,
                         min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                         max_session_length=MAX_SESSION_LENGTH,
                         days_test=DAYS_TEST):
    data, buys = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length, max_session_length)
    split_data(data, path_proc + file, days_test)


# preprocessing from original gru4rec but from a certain point in time
def preprocess_days_test_min_date(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED,
                                  min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                                  max_session_length=MAX_SESSION_LENGTH,
                                  days_test=DAYS_TEST, min_date=MIN_DATE):
    data, buys = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length, max_session_length)
    data = filter_min_date(data, min_date)
    split_data(data, path_proc + file, days_test)


# preprocessing to create data slices with a sliding window
def preprocess_slices(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT,
                      min_session_length=MIN_SESSION_LENGTH, max_session_length=MAX_SESSION_LENGTH,
                      num_slices=NUM_SLICES, days_offset=DAYS_OFFSET, days_shift=DAYS_SHIFT, days_train=DAYS_TRAIN,
                      days_test=DAYS_TEST):
    data, buys = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length, max_session_length)
    slice_data(data, path_proc + file, num_slices, days_offset, days_shift, days_train, days_test)


# just load and show info
def preprocess_info(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT,
                    min_session_length=MIN_SESSION_LENGTH, max_session_length=MAX_SESSION_LENGTH):
    data, buys = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length, max_session_length)


def preprocess_save(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT,
                    min_session_length=MIN_SESSION_LENGTH, max_session_length=MAX_SESSION_LENGTH):
    data, buys = load_data(path + file)
    data = filter_data(data, min_item_support, min_session_length, max_session_length)
    data.to_csv(path_proc + file + '_preprocessed.txt', sep='\t', index=False)


# preprocessing to create a file with buy actions
def preprocess_buys(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED):
    data, buys = load_data(path + file)
    store_buys(buys, path_proc + file)


def load_data(file):
    # 读取数据，并且按照设置会话间隔对会话进行划分
    # load csv
    data = pd.read_csv(file + '.csv', sep=',', header=0, usecols=[0, 1, 2, 3],
                       dtype={0: np.int64, 1: np.int32, 2: str, 3: np.int32})
    # specify header names
    data.columns = ['Time', 'UserId', 'Type', 'ItemId']
    data['Time'] = (data.Time / 1000).astype(int)   # 把毫秒转成秒
    # 按照用户和时间进行排序
    data.sort_values(['UserId', 'Time'], ascending=True, inplace=True)

    # sessionize
    data['TimeTmp'] = pd.to_datetime(data.Time, unit='s')   # 把时间戳转换为可读时间

    data.sort_values(['UserId', 'TimeTmp'], ascending=True, inplace=True)
    #     users = data.groupby('UserId')

    data['TimeShift'] = data['TimeTmp'].shift(1)    # 全部数据向下移一位
    data['TimeDiff'] = (data['TimeTmp'] - data['TimeShift']).dt.total_seconds().abs()   # 计算出两个交互的时间差，并转换成秒
    data['SessionIdTmp'] = (data['TimeDiff'] > SESSION_LENGTH).astype(int)  # 如果时间差大于SESSION_LENGTH,则说明是个新的session，那么tmp为1
    data['SessionId'] = data['SessionIdTmp'].cumsum(skipna=False)   # 对tmp做累加。同session的话tmp为0，新session为1
    del data['SessionIdTmp'], data['TimeShift'], data['TimeDiff']

    data.sort_values(['SessionId', 'Time'], ascending=True, inplace=True)

    cart = data[data.Type == 'addtocart']   # 区分两个交互类型的数据
    data = data[data.Type == 'view']
    del data['Type']

    #print(data)

    # output

    #print(data.Time.min())
    #print(data.Time.max())
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    del data['TimeTmp']

    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    return data, cart;


def filter_data(data, min_item_support, min_session_length, max_session_length):
    # y?
    session_lengths = data.groupby('SessionId').size()  # 统计每个会话的长度
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]    # 过滤掉长度为1会话

    # filter item support
    item_supports = data.groupby('ItemId').size()   # 统计每个物品出现过的次数
    data = data[np.in1d(data.ItemId, item_supports[item_supports >= min_item_support].index)]   # 过滤掉小于最小交互次数的物品

    # filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= min_session_length].index)]  # 再次过滤掉小于最小长度的会话
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths <= max_session_length].index)]  # 过滤掉大于最大长度的会话

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    return data;


def filter_min_date(data, min_date='2014-04-01'):
    min_datetime = datetime.strptime(min_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')

    # filter
    session_max_times = data.groupby('SessionId').Time.max()
    session_keep = session_max_times[session_max_times > min_datetime.timestamp()].index

    data = data[np.in1d(data.SessionId, session_keep)]

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    return data;


def split_data_org(data, output_file):
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax - 86400].index
    session_test = session_max_times[session_max_times >= tmax - 86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)

    tmax = train.Time.max()
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax - 86400].index
    session_valid = session_max_times[session_max_times >= tmax - 86400].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(),
                                                                        train_tr.ItemId.nunique()))
    train_tr.to_csv(output_file + '_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(),
                                                                             valid.ItemId.nunique()))
    valid.to_csv(output_file + '_train_valid.txt', sep='\t', index=False)


def split_data(data, output_file, days_test):
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)
    test_from = data_end - timedelta(days_test)

    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < test_from.timestamp()].index
    session_test = session_max_times[session_max_times >= test_from.timestamp()].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)


def slice_data(data, output_file, num_slices, days_offset, days_shift, days_train, days_test):
    for slice_id in range(0, num_slices):
        split_data_slice(data, output_file, slice_id, days_offset + (slice_id * days_shift), days_train, days_test)


def split_data_slice(data, output_file, slice_id, days_offset, days_train, days_test):
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format(slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(),
                 data_end.isoformat()))
    # 根据天数进行划分，计算三个时间点
    start = datetime.fromtimestamp(data.Time.min(), timezone.utc) + timedelta(days_offset)
    middle = start + timedelta(days_train)
    end = middle + timedelta(days_test)

    # prefilter the timespan
    session_max_times = data.groupby('SessionId').Time.max()    # 一个会话中最后一个交互的时间记为该会话的时间
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data_filtered = data[np.in1d(data.SessionId, greater_start.intersection(lower_end))]    # 过滤出train+test的数据

    print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} / {}'.
          format(slice_id, len(data_filtered), data_filtered.SessionId.nunique(), data_filtered.ItemId.nunique(),
                 start.date().isoformat(), middle.date().isoformat(), end.date().isoformat()))

    # split to train and test
    session_max_times = data_filtered.groupby('SessionId').Time.max()
    sessions_train = session_max_times[session_max_times < middle.timestamp()].index
    sessions_test = session_max_times[session_max_times >= middle.timestamp()].index

    train = data[np.in1d(data.SessionId, sessions_train)]

    print('Train set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format(slice_id, len(train), train.SessionId.nunique(), train.ItemId.nunique(), start.date().isoformat(),
                 middle.date().isoformat()))

    train.to_csv(output_file + '_train_full.' + str(slice_id) + '.txt', sep='\t', index=False)

    test = data[np.in1d(data.SessionId, sessions_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]

    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format(slice_id, len(test), test.SessionId.nunique(), test.ItemId.nunique(), middle.date().isoformat(),
                 end.date().isoformat()))

    test.to_csv(output_file + '_test.' + str(slice_id) + '.txt', sep='\t', index=False)


def store_buys(buys, target):
    buys.to_csv(target + '_buys.txt', sep='\t', index=False)


# -------------------------------------
# MAIN TEST
# --------------------------------------
if __name__ == '__main__':
    preprocess_slices()