import pandas as pd
import numpy as np
import os
import pickle
from copy import deepcopy
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import random
import torch
from tqdm import tqdm
import warnings
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from sortedcontainers import SortedDict
warnings.filterwarnings("ignore")

# 请注意以下路径无需修改
# 框架相关路径
global_root_path = rf'/home/lwyxyz'
minute_fea_path = rf'{global_root_path}/Stock60sBaseDataAll/Feather'
minute_mmap_path = rf'{global_root_path}/Stock60sBaseDataAll/Mmap'
support_data_path = rf'{global_root_path}/Stock60sConfig/support_data'
# 日频原始数据
data79_root_path = rf'{global_root_path}/2.79'
stock_daily_data_path1 = rf'{data79_root_path}/tonglian_data/ohlc_fea'
stock_daily_data_path2 = rf'{data79_root_path}/tonglian_data/support_data'
stock_daily_data_path3 = rf'{data79_root_path}/update/短周期依赖数据'
# 高频原始数据
local_data_path = rf'{global_root_path}/254.35/data/LocalDataLoader/LocalData'
trans_data_path = rf'{local_data_path}/StockTransData'
order_data_path = rf'{local_data_path}/StockOrderData'
lob_data_path = rf'{global_root_path}/hft_database/nas3/sec_lobdata'


def get_all_trade_days():
    read_file = rf"{stock_daily_data_path2}/trade_days_dict.pkl"
    all_trade_days = pd.read_pickle(read_file)['trade_days']
    all_trade_days = [x.strftime('%Y%m%d') for x in all_trade_days]
    all_trade_days.sort()
    return all_trade_days


def get_trade_days(start_date, end_date):
    all_trade_days = get_all_trade_days()
    trade_days = [date for date in all_trade_days if start_date <= date <= end_date]
    return trade_days


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(path, data):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_stock_daily_data(field):
    read_path = rf"{local_data_path}/StockDailyData"
    read_file = "%s/%s.fea" % (read_path, field)
    read_data = pd.read_feather(read_file).set_index("date")
    return read_data


def read_high_freq_data(date, name):
    if name == "StockLob":
        read_file = rf"{lob_data_path}/{date}.fea"
    else:
        read_file = "%s/%sData/%s.fea" % (local_data_path, name, date)
    read_data = pd.read_feather(read_file)
    return read_data


def read_all_support_dict():
    support_dict_list = [x for x in os.listdir(support_data_path) if "_loc_dict" in x]
    all_dict = {}
    for x in support_dict_list:
        try:
            dict_name = x.split("trade_")[1].split("_loc_dict")[0]
            all_dict[dict_name] = load_pickle(rf"{support_data_path}/{x}")
        except:
            pass
    return all_dict


def multiple_data(data_all, proc_num):
    step = len(data_all) // proc_num
    data_multiple = []
    spilt_code_list = [data_all.iloc[i * step]["code"] for i in range(proc_num)]
    spilt_num_list = list(
        data_all.loc[data_all.code.isin(spilt_code_list)].drop_duplicates(["code"], keep="last").index)
    spilt_num_list[0] = -1
    spilt_num_list += [len(data_all)]
    for i1, i2 in zip(spilt_num_list[:-1], spilt_num_list[1:]):
        data_multiple.append(data_all.loc[i1 + 1:i2])

    return data_multiple


def get_second_60s(time_10ms):
    div = 100000
    return round(round(time_10ms // div) * div)


# TODO 需要设定：本批字段的名称，计算的起止日期
base_data_name = rf'guadan'
start_date = '20210104'
end_date = '20240930'

# 计算和保存每日数据
# 基础字段保存在minute_fea_path的base_data_name子目录下
# minute_fea_path这个路径会映射到当前用户的home目录下的data_share/Stock60sBaseDataAll/Feather下，是同一个路径
fea_save_path = rf'{minute_fea_path}/{base_data_name}'
os.makedirs(fea_save_path, exist_ok=True)
#os.chmod(fea_save_path, 0o755)
trade_date_list = get_trade_days(start_date, end_date)

all_support_dict = read_all_support_dict()
standard_time_list = list(all_support_dict[rf"time_60s"].keys())  # 分钟标记
standard_time_num = len(standard_time_list)  # 分钟数

# 处理委托数据
def proc_order(order_data_cut):
    temp_list = []
    order_data_cut = order_data_cut.sort_values(["code", "time", "order"]).reset_index(drop=True)
    for side in ["B", "S"]:
        data = order_data_cut[order_data_cut["functionCode"] == side]
        data["second"] = data["time"].map(get_second_60s)
        # 市价单的委托价格是无效的，因此通过orderKind字段剔除市价单，仅保留限价单
        data2 = data.loc[(data["orderKind"] == "A") | (data["orderKind"] == "0")]
        
        data_group2 = data2.groupby(["code", "second"])

        temp1 = data_group2.agg({"orderPrice": ["count", "mean", "std"]})
        temp1.columns = ["OrderNumSum", "OrderPriceMean", "OrderPriceStd"]
        temp2 = data_group2.agg({"orderAmount": ["sum", "max"]})
        temp2.columns = ["OrderAmtSum", "OrderAmtMax"]

        temp = pd.concat([temp1, temp2], axis=1)
        temp.columns = [col + "_%s" % side for col in temp.columns]
        temp_list.append(temp)

    second_data_ori = pd.concat(temp_list, axis=1)
    second_data = proc_second_data(second_data_ori)
    return second_data


# 处理成交数据
def proc_trans(trans_data_cut):
    temp_list = []
    # 分为撤单和成交两个大部分
    cancel_data = trans_data_cut[trans_data_cut["functionCode"] == "C"]  # 撤单数据
    dealed_data = trans_data_cut[trans_data_cut["functionCode"] != "C"]  # 成交数据

    # 处理撤单部分
    for side in ["B", "S"]:
        data = cancel_data[cancel_data["bsFlag"] == side]
        data["second"] = data["time"].map(get_second_60s)
        data_group = data.groupby(["code", "second"])

        temp = data_group.agg({"tradeVolume": ["count", "sum", "max"]})
        temp.columns = ["CancelNumSum", "CancelVolSum", "CancelVolMax"]
        temp.columns = [col + "_%s" % side for col in temp.columns]
        temp_list.append(temp)

    # 处理成交部分
    for side in ["B", "S"]:
        data = dealed_data[dealed_data["bsFlag"] == side]
        data["second"] = data["time"].map(get_second_60s)
        data_group = data.groupby(["code", "second"])

        temp1 = data_group.agg({"tradePrice": ["count", "mean", "std"]})
        temp1.columns = ["TradeNumSum", "TradePriceMean", "TradePriceStd"]
        temp2 = data_group.agg({"tradeAmount": ["sum", "max"]})
        temp2.columns = ["TradeAmtSum", "TradeAmtMax"]

        temp = pd.concat([temp1, temp2], axis=1)
        temp.columns = [col + "_%s" % side for col in temp.columns]
        temp_list.append(temp)

    second_data_ori = pd.concat(temp_list, axis=1)
    second_data = proc_second_data(second_data_ori)
    return second_data


# 处理lob数据
def proc_lob(lob_data_cut):
    lob_data_cut["second"] = lob_data_cut["time"].map(get_second_60s)
    lob_data_cut = lob_data_cut.drop_duplicates(["second", "stock"], keep="last")
    lob_data_cut = lob_data_cut.set_index(["code", "second"])
    lob_data_cut = lob_data_cut.replace(0.0, np.nan)
    keep_list = []
    for side in ['b', 's']:
        for num in [1, 5]:
            if num == 1:
                lob_data_cut[f"{side}p1"] = lob_data_cut[f"{side}p_1"]
                lob_data_cut[f"{side}v1"] = lob_data_cut[f"{side}v_1"]
            if num in [5]:
                lob_data_cut[f"{side}v1{num}"] = lob_data_cut[
                    [f'{y}_{x}' for x in range(1, num + 1) for y in [f"{side}v"]]].sum(axis=1)
                lob_data_cut[f"{side}amt1{num}"] = np.nansum(
                    lob_data_cut[[f'{y}_{x}' for x in range(1, num + 1) for y in [f"{side}p"]]].values *
                    lob_data_cut[[f'{y}_{x}' for x in range(1, num + 1) for y in [f"{side}v"]]].values, axis=1)

        keep_list += [f"{side}p1", f"{side}v1", f"{side}v15", f"{side}amt15", ]

    second_data_ori = lob_data_cut[keep_list]
    second_data = proc_second_data(second_data_ori)
    return second_data


# 对60秒聚合的数据进行进一步处理
def proc_second_data(second_data_ori):
    second_data = {}
    for field in second_data_ori.columns:
        df = second_data_ori[field].unstack(0)

        # 必须处理的部分：时间轴对齐，保证股票代码为六位
        df = df.reindex(standard_time_list)  # 标准化时间轴
        df.columns = df.columns.map(lambda x: x[:6])  # 标准化代码

        # 自由处理的部分：对价格数据进行填前值，对量和金额填0等等，和字段的计算逻辑相关
        # 例如，对价格等序列，可以填前值
        # if (field in price_field):
        #     df = df.replace(0, np.nan).ffill().fillna(today_stock_pre_close)
        # 例如，对量额等序列，可以填0
        # elif (field in volume_amount_field) :
        #     df = df.fillna(0.0)
        # else:
        #     pass

        df = df.fillna(0.0)

        second_data[field] = df.stack()

    second_data = pd.concat(second_data, axis=1)
    return second_data


def format_second_data(second_data):
    float_cols = [x for x in second_data.columns if x not in ["code", "second"]]
    # 所有字段按float32保存
    second_data[float_cols] = second_data[float_cols].astype('float32')
    # 这里需要把列名改为code+second+字段名的顺序，否则后续转存为mmap时会有异常
    second_data = second_data[["code", "second"] + float_cols]
    return second_data

bid_prices = SortedDict()  # 价格按升序排列，最高价在最后
ask_prices = SortedDict()  # 价格按升序排列，最低价在前

def processbid(price,volume):
    global bid_prices
    for bid_price in sorted(bid_prices.keys(),reverse=True):
        if bid_price < price:
            break
        bid_volume=bid_prices[bid_price]
        if bid_volume>volume:
            bid_prices[bid_price]-=volume
            break
        else:
            volume-=bid_volume
            del bid_prices[bid_price]
            
def processask(price,volume):
    global ask_prices
    for ask_price in sorted(ask_prices.keys()):
        if ask_price > price:
            break
        ask_volume=ask_prices[ask_price]
        if ask_volume>volume:
            ask_prices[ask_price]-=volume
            break
        else:
            volume-=ask_volume
            del ask_prices[ask_price]

date='20210104'   
# 读取数据
order_data = read_high_freq_data(date, name="StockOrder")
trans_data = read_high_freq_data(date, name="StockTrans")
# 价格和金额字段需要除以10000变为原始值（单位：元）
order_data["orderPrice"] = order_data["orderPrice"] / 10000.0
order_data["orderAmount"] = order_data["orderAmount"] / 10000.0
trans_data["tradePrice"] = trans_data["tradePrice"] / 10000.0
trans_data["tradeAmount"] = trans_data["tradeAmount"] / 10000.0

def get_second_s(time_10ms):
    div = 100000
    return round(round(time_10ms // div) * div)
trans_data["minute"] = trans_data["time"].map(get_second_s)
order_data["minute"] = order_data["time"].map(get_second_s)

trans_data1=trans_data
"""
# 市价单的委托价格是无效的，因此通过orderKind字段剔除市价单1，仅保留限价单0A
order_data_nomarket = order_data.loc[order_data["orderKind"].isin(["A", "0"])]

#市价单 本方最优 委托价格用成交价填充
# 买单为例，卖单同理不赘述
market_order_buy = pd.merge(
    order_data[(order_data['functionCode']=='B')&(order_data['orderKind'].isin(["1", "U"]))],
    trans_data[['code','bidOrder', 'tradePrice','tradeVolume']],
    left_on=['code','order'],
    right_on=['code','bidOrder'],
    how='inner'
)
market_order_buy['orderPrice'] = market_order_buy['tradePrice']
market_order_buy['orderVolume'] = market_order_buy['tradeVolume']
market_order_buy.drop(columns=['bidOrder','tradePrice','tradeVolume'],inplace=True)


# 买单为例，卖单同理不赘述
market_order_sell = pd.merge(
    order_data[(order_data['functionCode']=='S')&(order_data['orderKind'].isin(["1", "U"]))],
    trans_data[['code','askOrder', 'tradePrice','tradeVolume']],
    left_on=['code','order'],
    right_on=['code','askOrder'],
    how='inner'
)
market_order_sell['orderPrice'] = market_order_sell['tradePrice']
market_order_sell['orderVolume'] = market_order_sell['tradeVolume']
market_order_sell.drop(columns=['askOrder','tradePrice','tradeVolume'],inplace=True)


order_data=pd.concat([order_data_nomarket,market_order_buy,market_order_sell],axis=0)"""

order_data1 = order_data.rename(columns={"functionCode": "bsFlag"})
order_data1['askOrder']=np.where(order_data1['bsFlag']=='S',order_data1['order'],pd.NA)
order_data1['bidOrder']=np.where(order_data1['bsFlag']=='B',order_data1['order'],pd.NA)
order_data.drop(columns=['order'], inplace=True)

trans_data1['type']='trans'
order_data1['type']='order'
combined = pd.concat([order_data1, trans_data1], axis=0).sort_values(['code', 'time'], ascending=True,kind='stable')
#撤单 成交价用委托价填充
# 创建价格填充辅助列（处理ask和bid方向）
combined['ask_price'] = np.where(
    (combined['type'] == 'order') & (combined['askOrder'].notna()),
    combined['orderPrice'],
    np.nan
)
combined['bid_price'] = np.where(
    (combined['type'] == 'order') & (combined['bidOrder'].notna()),
    combined['orderPrice'],
    np.nan
)
# 分组填充逻辑（关键步骤）
combined['ask_price'] = combined.groupby(
    ['code', combined['askOrder'].astype(str)]  # 转为字符串处理NaN
)['ask_price'].ffill()

combined['bid_price'] = combined.groupby(
    ['code', combined['bidOrder'].astype(str)]
)['bid_price'].ffill()

# 填充交易价格
mask = (combined['type'] == 'trans') & (combined['functionCode'] == 'C')& (combined['orderKind'] == '0')
combined.loc[mask, 'tradePrice'] = np.select(
    condlist=[
        combined.loc[mask, 'bsFlag']=='S',  # 优先处理askOrder
        combined.loc[mask, 'bsFlag']=='B'
    ],
    choicelist=[
        combined.loc[mask, 'ask_price'],
        combined.loc[mask, 'bid_price']
    ],
    default=0  # 无匹配时保持0值
)

combined.drop(['ask_price', 'bid_price'], axis=1, inplace=True)

#成交单的委托价用之前的委托单的委托价填充 深市要填 沪市不用填因为沪市成交单不一定找得到之前的委托单
# 生成动态分组键（bidOrder用于B类，askOrder用于S类）
combined_sz=combined[(combined['orderKind']=='0')&(combined['type']=='trans')]
bid_ask_groups = np.where(
    combined_sz['bsFlag'] == 'B',
    combined_sz['bidOrder'].astype(str),
    combined_sz['askOrder'].astype(str)
)
# 单次分组完成双向填充
combined[(combined['orderKind']=='0')&(combined['type']=='trans')]['orderPrice'] \
    = combined[(combined['orderKind']=='0')&(combined['type']=='trans')].groupby(
    ['code', 'bsFlag', bid_ask_groups]
)['orderPrice'].ffill()

# 初始化数据结构
last_code = None
last_minute = None
prcvol_list = []
result = []

for row in combined.itertuples():

    code = row.code  # 使用 . 访问属性，而非 ['code']
    time = row.time
    minute = row.minute
    # 股票切换时清空所有数据
    if last_code is not None and code != last_code:

        bid_prices.clear()
        ask_prices.clear()
        prcvol_list.clear()
        last_minute = None
        last_code = code
    # 分钟切换处理
    if last_minute and minute > last_minute:
                # 计算分钟级指标
                total_vol = sum(v for _,v,_ in prcvol_list)
                total_amt = sum(p*v for p,v,_ in prcvol_list)
                
                vwap = total_amt / total_vol if total_vol > 0 else 0
                if prcvol_list:
                    open,_,_=prcvol_list[0]
                    high=max(p for p,_,_ in prcvol_list)
                    low=min(p for p,_,_ in prcvol_list)
                    close,_,_=prcvol_list[-1]
                 # 获取买卖五档（利用 SortedDict 的有序性）
                # 买方档位：SortedDict 按价格升序，最高价在最后
                bid_items = bid_prices.items()
                bids = list(bid_items)[-5:][::-1]  # 取最后5个并反转，得到降序排列
                # 卖方档位：SortedDict 按价格升序，最低价在前
                asks = list(ask_prices.items())[:5]


                # 生成记录
                record = {
                    'code': code,
                    'second': last_minute,
                    'vwap': round(vwap, 4),  # 价格单位转换 成交价
                    'volume': total_vol,#成交量
                    'trades': len(prcvol_list)#成交笔数
                }
                
                # 填充买卖档
                for i, (p, v) in enumerate(bids, 1):
                    record.update({
                        f'bid_price_{i}': p,
                        f'bid_vol_{i}': v
                    })
                for i, (p, v) in enumerate(asks, 1):
                    record.update({
                        f'ask_price_{i}': p,
                        f'ask_vol_{i}': v
                    })
                
                result.append(record)
                prcvol_list = []
                last_minute = minute
                
                
    # 首次初始化
    if not last_minute:
                last_minute = minute
    # 首次初始化
    if not last_code:
                last_code = code   
                
    if row.type =='order':
        if row.orderKind=="1":
           pass
        elif row.orderKind=='U':
            if (row.bsFlag == 'B' and not bid_prices) or (row.bsFlag == 'S' and not ask_prices):
                continue  # 跳过无本方订单的情况
            price,v= list(bid_prices.items())[-1] if row.bsFlag == 'B' else list(ask_prices.items())[0]
            volume = row.orderVolume
            target_dict = bid_prices if row.bsFlag == 'B' else ask_prices
            target_dict[price] = target_dict.get(price, 0) + volume
        else:
            price = row.orderPrice
            volume = row.orderVolume
            target_dict = bid_prices if row.bsFlag == 'B' else ask_prices
            target_dict[price] = target_dict.get(price, 0) + volume
            
    else:
        #成交单中集合竞价成交单
        if (row.time>=92500000)&(row.time<93000000):
            processbid(row.tradePrice,row.tradeVolume)
            processask(row.tradePrice,row.tradeVolume)

        #成交单中撤单
        elif row.functionCode in ['C'] or row.orderKind in ['D']:
            prc = row.tradePrice
            vol = row.tradeVolume
            target_dict = bid_prices if row.bsFlag == 'B' else ask_prices
            # 买方撤单 更新买方挂单
            if prc in target_dict:
                target_dict[prc] -= vol
                if target_dict[prc] <= 0:
                    del target_dict[prc]
            
        #成交单中成交        
        else:
            prc = row.tradePrice
            vol = row.tradeVolume
            prcvol_list.append((prc, vol, code))
            if row.orderKind=='0':
                if row.bsFlag == 'S':
                    #深市成交单 先委托后成交 两边都要减
                    target_dict = bid_prices
                    # 卖方主动成交 更新买方挂单
                    if prc in target_dict:
                        target_dict[prc] -= vol
                        if target_dict[prc] <= 0:
                            del target_dict[prc]
                    target_dict = ask_prices
                    # 卖方主动成交 更新卖方挂单
                    if row.orderPrice in target_dict:
                        target_dict[row.orderPrice] -= vol
                        if target_dict[row.orderPrice] <= 0:
                            del target_dict[row.orderPrice]
                else:
                        #深市成交单 先委托后成交 两边都要减
                    target_dict = bid_prices
                    #买方主动成交 更新买方挂单
                    if row.orderPrice in target_dict:
                        target_dict[row.orderPrice] -= vol
                        if target_dict[row.orderPrice] <= 0:
                            del target_dict[row.orderPrice]
                    target_dict = ask_prices
                    # 买方主动成交 更新卖方挂单
                    if prc in target_dict:
                        target_dict[prc] -= vol
                        if target_dict[prc] <= 0:
                            del target_dict[prc]
                    
                    
            else:
                #沪市成交单 成交后剩下的委托委托 成交 
                target_dict = bid_prices if row.bsFlag == 'S' else ask_prices
                
                # 卖方主动成交 更新买方挂单
                if prc in target_dict:
                    target_dict[prc] -= vol
                    if target_dict[prc] <= 0:
                        del target_dict[prc]

            
result=pd.DataFrame(result)
seconddt= proc_second_data(result)
         #输入 输出都有两列索引
         #second_data = second_data.reset_index().sort_values(["code", "second"]).reset_index(drop=True)
seconddt=seconddt.reset_index()
second_data = format_second_data(seconddt)
     #输入输出都无索引
     # 保存文件并设置文件读写权限
     #存文件无索引
#second_data .to_feather(rf'{fea_save_path}/{date}.fea', compression="zstd")
#os.chmod(rf'{fea_save_path}/{date}.fea', 0o755)                     