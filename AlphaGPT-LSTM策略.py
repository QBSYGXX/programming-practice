# 标题：AlphaGPT-LSTM策略-截面因子挖掘示例
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from jqdata import *
import datetime as dt
from datetime import timedelta

# ========== 工具函数：股票过滤 ==========
def filter_kcbj_stock(stock_list):
    """过滤科创板、北交所股票"""
    filtered_list = []
    for stock in stock_list[:]:
        if not (stock.startswith('688') or stock.startswith('8') or stock.startswith('4')):
            filtered_list.append(stock)
    return filtered_list

def filter_st_stock(stock_list):
    """过滤ST及退市风险股票"""
    current_data = get_current_data()  
    filtered = [
        stock for stock in stock_list
        if not current_data[stock].is_st
        and 'ST' not in current_data[stock].name
        and '*' not in current_data[stock].name
        and '退' not in current_data[stock].name
    ]
    return filtered

def filter_paused_stock(stock_list):
    """过滤停牌股票）"""
    current_data = get_current_data()  
    return [stock for stock in stock_list if not current_data[stock].paused]

def filter_new_stock(context, stock_list, days=90):
    """过滤上市不足days天的新股）"""
    yesterday = context.previous_date  
    return [
        stock for stock in stock_list 
        if not (yesterday - get_security_info(stock).start_date) < dt.timedelta(days=days)
    ]

def select_small_cap_stocks(context):
    """选股：中小综指中市值最小的20只股票"""
    yesterday = context.previous_date  
    
    # 1. 获取中小综指成分股
    try:
        initial_list = get_index_stocks('399101.XSHE', yesterday)
        log.info(f"[选股] 初始选股池（中小综指）：{len(initial_list)}只")
    except Exception as e:
        log.error(f"获取中小综指成分股失败: {str(e)}，使用备用股票池")
        initial_list = []
    
    if not initial_list:
        return []
    
    # 2. 基础过滤
    initial_list = filter_new_stock(context, initial_list)  # 先过滤新股
    initial_list = filter_kcbj_stock(initial_list)         # 过滤科创北交
    initial_list = filter_st_stock(initial_list)           # 过滤ST
    initial_list = filter_paused_stock(initial_list)       # 过滤停牌
    log.info(f"[选股] 基础过滤后选股池：{len(initial_list)}只")
    
    # 3. 市值筛选（<35亿，取最小的20只）
    try:
        q = query(
            valuation.code, 
            valuation.market_cap,
            valuation.circulating_market_cap
        ).filter(
            valuation.code.in_(initial_list),
            valuation.market_cap < 35  # 总市值<35亿
        ).order_by(
            valuation.circulating_market_cap.asc()  # 按流通市值升序
        )
        fund_df = get_fundamentals(q, date=yesterday)
        stock_list = list(fund_df.code)[:20]
        log.info(f"[选股] 市值筛选（<35亿，最小20只）：{len(stock_list)}只")
        return stock_list
    except Exception as e:
        log.error(f"市值筛选失败: {str(e)}")
        return initial_list[:20] if initial_list else []

# ========== 基于基础order函数实现所有下单逻辑 ==========
def get_stock_current_price(security, current_dt):
    """获取股票当前价格"""
    try:
        # 尝试从data对象获取
        return get_price(
            security,
            start_date=current_dt,
            end_date=current_dt,
            frequency='daily',
            fields=['open'],
            fq='pre'
        )['open'].iloc[0]
    except:
        # 保底：获取最新收盘价
        df = get_price(
            security,
            start_date=current_dt - pd.Timedelta(days=5),
            end_date=current_dt,
            frequency='daily',
            fields=['close'],
            fq='pre'
        )
        return df['close'].iloc[-1] if not df.empty else 0

def calculate_target_shares(context, security, target_percent):
    """计算目标持仓数量（基于仓位比例）"""
    # 获取当前股票价格
    price = get_stock_current_price(security, context.current_dt)
    if price <= 0:
        return 0
    
    # 计算目标市值
    total_value = context.portfolio.total_value
    target_value = total_value * target_percent
    
    # 计算目标数量（100股为单位）
    target_shares = int(target_value / price / 100) * 100
    
    # 确保最小交易数量
    if target_shares < 100 and target_value > 1000:
        target_shares = 100
    
    return target_shares

def safe_order_by_percent(context, security, target_percent):
    """按目标仓位比例下单（基于基础order函数）"""
    try:
        # 获取当前持仓数量
        current_shares = context.portfolio.positions.get(security, 0).total_amount if security in context.portfolio.positions else 0
        
        # 计算目标持仓数量
        target_shares = calculate_target_shares(context, security, target_percent)
        
        # 计算需要买卖的数量
        diff = target_shares - current_shares
        
        if diff > 0:
            # 买入
            log.info(f" 买入 {security} | 目标仓位: {target_percent:.2%} | 当前持仓: {current_shares}股 | 需买入: {diff}股")
            order(security, diff)  # 基础order函数，正数买入
            return True
        elif diff < 0:
            # 卖出
            log.info(f" 卖出 {security} | 目标仓位: {target_percent:.2%} | 当前持仓: {current_shares}股 | 需卖出: {abs(diff)}股")
            order(security, diff)  # 基础order函数，负数卖出
            return True
        else:
            log.info(f" {security} 持仓已达标，无需操作")
            return True
    except Exception as e:
        log.error(f"下单失败: {security} | {target_percent:.2%} | 错误: {str(e)}")
        return False

def safe_order_close_position(context, security):
    """清仓指定股票（基于基础order函数）"""
    try:
        if security in context.portfolio.positions:
            current_shares = context.portfolio.positions[security].total_amount
            if current_shares > 0:
                log.info(f" 清仓 {security} | 当前持仓: {current_shares}股")
                order(security, -current_shares)  # 卖出全部
                return True
        return True
    except Exception as e:
        log.error(f"清仓失败: {security} | 错误: {str(e)}")
        return False

# ========== 全局配置类==========
class GlobalConfig:
    pass
g = GlobalConfig()

# ========== 策略核心配置 ==========
g.BATCH_SIZE = 64
g.TRAIN_ITERATIONS = 30  # 每月训练，减少迭代次数加快速度
g.MAX_SEQ_LEN = 8
g.COST_RATE = 0.001
g.TOP_K = 10
g.DEVICE = torch.device("cpu")
g.TRAIN_WINDOW_MONTHS = 12  # 训练窗口：12个月
g.ALLOW_TRAIN_PERIOD_TRADING = True

# --- 算子配置 ---
g.OPS_CONFIG = [
    ('ADD', lambda x, y: clean_torch_tensor(x + y), 2),
    ('SUB', lambda x, y: clean_torch_tensor(x - y), 2),
    ('MUL', lambda x, y: clean_torch_tensor(x * y), 2),
    ('DIV', lambda x, y: clean_torch_tensor(x / (y + 1e-6 * torch.sign(y))), 2),
    ('NEG', lambda x: clean_torch_tensor(-x), 1),
    ('ABS', lambda x: clean_torch_tensor(torch.abs(x)), 1),
    ('SIGN', lambda x: clean_torch_tensor(torch.sign(x)), 1),
    ('DELTA5', lambda x: clean_torch_tensor(x - torch.roll(x, 5, dims=1)), 1),
    ('MA20', lambda x: clean_torch_tensor(torch.nn.functional.avg_pool1d(
        x.unsqueeze(1), 20, stride=1, padding=9).squeeze(1)), 1),
    ('CS_RANK', lambda x: _cross_section_rank(x), 1),
    ('CS_ZSCORE', lambda x: _cross_section_zscore(x), 1),
    ('CS_TOP20', lambda x: _cross_section_quantile(x, 0.8), 1),
    ('CS_BOT20', lambda x: _cross_section_quantile(x, 0.2), 1),
    ('CS_MOM10', lambda x: _cross_section_momentum(x, 10), 1),
]

# 特征集
g.FEATURES = ['RET', 'RET5', 'VOL_CHG', 'V_RET', 'TREND', 'LIQUIDITY', 'VOLATILITY', 'SIZE']
g.VOCAB = g.FEATURES + [cfg[0] for cfg in g.OPS_CONFIG]
g.VOCAB_SIZE = len(g.VOCAB)
g.OP_FUNC_MAP = {i + len(g.FEATURES): cfg[1] for i, cfg in enumerate(g.OPS_CONFIG)}
g.OP_ARITY_MAP = {i + len(g.FEATURES): cfg[2] for i, cfg in enumerate(g.OPS_CONFIG)}

# ========== 数值清理函数 ==========
def clean_numpy_array(x):
    """清理数值：替换NaN/Inf为0，限制范围"""
    if x.size == 0:
        return x
    x = np.where(np.isnan(x), 0.0, x)
    x = np.where(np.isinf(x), 0.0, x)
    x = np.where(x == -np.inf, 0.0, x)
    x = np.clip(x, -1e6, 1e6)
    return x

def clean_torch_tensor(x):
    """清理张量：替换NaN/Inf为0，限制范围"""
    x = torch.where(torch.isnan(x), torch.tensor(0.0, device=x.device), x)
    x = torch.where(torch.isinf(x), torch.tensor(0.0, device=x.device), x)
    x = torch.where(x == -float('inf'), torch.tensor(0.0, device=x.device), x)
    x = torch.clamp(x, -1e6, 1e6)
    return x

# ========== 截面因子专用算子） ==========
def _cross_section_rank(x: torch.Tensor) -> torch.Tensor:
    """截面排序"""
    B, T = x.shape
    ranks = torch.zeros_like(x)
    for t in range(T):
        valid_mask = ~torch.isnan(x[:, t])
        if valid_mask.sum() > 0:
            ranks[valid_mask, t] = torch.argsort(torch.argsort(x[valid_mask, t]))
            ranks[valid_mask, t] /= (valid_mask.sum() + 1e-6)
    return clean_torch_tensor(ranks)

def _cross_section_zscore(x: torch.Tensor) -> torch.Tensor:
    """截面标准化"""
    B, T = x.shape
    result = torch.zeros_like(x)
    for t in range(T):
        valid_mask = ~torch.isnan(x[:, t])
        if valid_mask.sum() > 1:
            valid_data = x[valid_mask, t]
            mean_val = valid_data.mean()
            std_val = valid_data.std() + 1e-6
            result[valid_mask, t] = (valid_data - mean_val) / std_val
    return clean_torch_tensor(result)

def _cross_section_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    """截面分位数"""
    B, T = x.shape
    quantiles = torch.zeros_like(x)
    for t in range(T):
        valid_mask = ~torch.isnan(x[:, t])
        if valid_mask.sum() > 0:
            valid_data = x[valid_mask, t]
            threshold = torch.quantile(valid_data, q)
            quantiles[valid_mask, t] = (valid_data > threshold).float()
    return clean_torch_tensor(quantiles)

def _cross_section_momentum(x: torch.Tensor, lookback: int) -> torch.Tensor:
    """截面动量"""
    if lookback <= 0:
        return clean_torch_tensor(x)
    B, T = x.shape
    momentum = torch.zeros_like(x)
    for t in range(lookback, T):
        valid_mask = ~torch.isnan(x[:, t]) & ~torch.isnan(x[:, t - lookback])
        if valid_mask.sum() > 0:
            momentum[valid_mask, t] = x[valid_mask, t] / (x[valid_mask, t - lookback] + 1e-6) - 1
    return clean_torch_tensor(momentum)

# ========== 模型定义 ==========
class AlphaGPT(nn.Module):
    def __init__(self, d_model=64, n_head=4, n_layer=2):
        super(AlphaGPT, self).__init__()
        self.embedding = nn.Embedding(g.VOCAB_SIZE, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layer,
            batch_first=True,
            bidirectional=False
        )
        self.fc_actor = nn.Linear(d_model, g.VOCAB_SIZE)
        self.fc_critic = nn.Linear(d_model, 1)
        
    def forward(self, idx):
        B, T = idx.size()
        x = self.embedding(idx)
        x, (hn, cn) = self.lstm(x)
        last_hidden = x[:, -1, :]
        logits = self.fc_actor(last_hidden)
        value = self.fc_critic(last_hidden)
        return logits, value

# ========== 数据引擎 (支持滚动窗口) ==========
class JQCrossSectionDataEngine:
    def __init__(self, stock_list, start_date, end_date):
        self.stocks = stock_list
        self.num_stocks = len(self.stocks)
        self.start_date = start_date
        self.end_date = end_date
        self.feat_data = None
        self.target_ret = None
        self.dates = None
        self.split_idx = None
        self.time_dim = None

    def load(self):
        """加载数据：完全不依赖基本面接口，只用日线数据生成所有特征"""
        if not self.stocks:
            log.error("股票池为空，无法加载数据")
            return self
            
        log.info(f" 从聚宽获取 {self.num_stocks} 只股票的日线数据...")
        dfs = []
        
        for code in self.stocks:
            try:
                # 1. 只获取核心日线数据
                df = get_price(
                    code,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    frequency='daily',
                    fields=['open', 'close', 'high', 'low', 'volume', 'money'],
                    fq='pre'
                )
                
                if df.empty:
                    log.info(f" {code} 日线数据为空，跳过")
                    continue
                
                # 2. 初始化所有字段
                df['trade_date'] = pd.to_datetime(df.index)
                df['code'] = code
                df = df.reset_index(drop=True)
                
                # 3. 生成替代特征
                # 流通市值替代：成交额的60日移动平均
                df['circulating_market_cap'] = np.log(df['money'].rolling(60).mean().fillna(df['money']) + 1e-6)
                
                # 换手率替代：成交量/60日均量
                vol_ma60 = df['volume'].rolling(60).mean().fillna(1)
                df['turnover_ratio'] = df['volume'] / (vol_ma60 + 1e-6) - 1
                
                # 保底填充
                df['circulating_market_cap'] = df['circulating_market_cap'].fillna(0)
                df['turnover_ratio'] = df['turnover_ratio'].fillna(0)
                
                for col in ['open', 'close', 'high', 'low', 'volume', 'money']:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                dfs.append(df)
                log.info(f"{code} 数据加载成功 (共{len(df)}条)")
                
            except Exception as e:
                log.info(f" 处理 {code} 时出错: {str(e)[:30]}... 继续尝试")
                continue
        
        # 保底检查
        if not dfs:
            log.info(" 未获取到任何股票数据，使用测试数据继续...")
            test_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            test_df = pd.DataFrame({
                'trade_date': test_dates,
                'code': self.stocks[0] if self.stocks else '000001.XSHE',
                'open': 100 + np.random.randn(len(test_dates)) * 5,
                'close': 100 + np.random.randn(len(test_dates)) * 5,
                'high': 105 + np.random.randn(len(test_dates)) * 5,
                'low': 95 + np.random.randn(len(test_dates)) * 5,
                'volume': 1e8 + np.random.randn(len(test_dates)) * 1e7,
                'money': 1e10 + np.random.randn(len(test_dates)) * 1e9,
                'circulating_market_cap': np.log(1e11) + np.random.randn(len(test_dates)) * 0.1,
                'turnover_ratio': np.random.randn(len(test_dates)) * 0.01
            })
            dfs = [test_df for _ in range(min(5, len(self.stocks)))]
            
        # 合并数据
        df_all = pd.concat(dfs).sort_values(['trade_date', 'code']).reset_index(drop=True)
        
        # 数据透视
        def pivot_col(col):
            pivot = df_all.pivot(index='trade_date', columns='code', values=col)
            for s in self.stocks[:len(dfs)]:
                if s not in pivot.columns:
                    pivot[s] = 0
            pivot = pivot[[col for col in self.stocks if col in pivot.columns]]
            return pivot.ffill().bfill().values.astype(np.float32)

        # 获取核心矩阵
        self.dates = pd.to_datetime(df_all['trade_date'].unique()).sort_values()
        opens = pivot_col('open').T          
        closes = pivot_col('close').T        
        vols = pivot_col('volume').T         
        highs = pivot_col('high').T          
        lows = pivot_col('low').T            
        circ_mv = pivot_col('circulating_market_cap').T
        turnover = pivot_col('turnover_ratio').T

        # 更新维度
        self.num_stocks = opens.shape[0]
        self.time_dim = opens.shape[1]
        log.info(f" 数据维度 - 股票数: {self.num_stocks}, 时间数: {self.time_dim}")

        # 计算截面特征
        # 1. 基础收益特征
        ret = np.zeros_like(closes)
        ret[:, 1:] = (closes[:, 1:] - closes[:, :-1]) / (closes[:, :-1] + 1e-6)
        ret = clean_numpy_array(ret)
        
        # 2. 5日收益率
        ret5 = np.zeros_like(closes)
        for i in range(self.num_stocks):
            ret5[i] = pd.Series(closes[i]).pct_change(5).fillna(0).values
        ret5 = clean_numpy_array(ret5)
        
        # 3. 成交量变化率
        vol_chg = np.zeros_like(vols)
        for i in range(self.num_stocks):
            v_ma = pd.Series(vols[i]).rolling(20).mean().fillna(1).values
            v_ma = clean_numpy_array(v_ma)
            mask = v_ma > 1e-6
            vol_chg[i][mask] = vols[i][mask] / v_ma[mask] - 1
        vol_chg = clean_numpy_array(vol_chg)
        
        # 4. 量价结合
        v_ret = ret * (vol_chg + 1)
        v_ret = clean_numpy_array(v_ret)

        # 5. 趋势指标
        trend = np.zeros_like(closes)
        for i in range(self.num_stocks):
            ma60 = pd.Series(closes[i]).rolling(60).mean().fillna(1).values
            ma60 = clean_numpy_array(ma60)
            mask = ma60 > 1e-6
            trend[i][mask] = closes[i][mask] / ma60[mask] - 1
        trend = clean_numpy_array(trend)

        # 6. 流动性指标
        liquidity = clean_numpy_array(turnover)

        # 7. 波动率指标
        volatility = np.zeros_like(closes)
        for i in range(self.num_stocks):
            volatility[i] = pd.Series(ret[i]).rolling(20).std().fillna(0).values
        volatility = clean_numpy_array(volatility)
        
        # 8. 规模因子
        size_factor = clean_numpy_array(circ_mv)

        # 截面标准化
        def cross_section_norm(x):
            result = np.zeros_like(x)
            for t in range(x.shape[1]):
                valid_data = x[:, t]
                valid_mask = ~np.isnan(valid_data)
                if valid_mask.sum() > 1:
                    median_val = np.nanmedian(valid_data)
                    mad_val = np.nanmedian(np.abs(valid_data - median_val)) + 1e-6
                    result[valid_mask, t] = (valid_data[valid_mask] - median_val) / mad_val
            result = np.clip(result, -3, 3).astype(np.float32)
            return clean_numpy_array(result)

        # 整理特征数据
        self.feat_data = torch.stack([
            torch.from_numpy(cross_section_norm(ret)).to(g.DEVICE),
            torch.from_numpy(cross_section_norm(ret5)).to(g.DEVICE),
            torch.from_numpy(cross_section_norm(vol_chg)).to(g.DEVICE),
            torch.from_numpy(cross_section_norm(v_ret)).to(g.DEVICE),
            torch.from_numpy(cross_section_norm(trend)).to(g.DEVICE),
            torch.from_numpy(cross_section_norm(liquidity)).to(g.DEVICE),
            torch.from_numpy(cross_section_norm(volatility)).to(g.DEVICE),
            torch.from_numpy(cross_section_norm(size_factor)).to(g.DEVICE)
        ])

        # 计算目标收益
        open_tensor = torch.from_numpy(opens).to(g.DEVICE)
        next_open = torch.roll(open_tensor, -1, dims=1)
        next_next_open = torch.roll(open_tensor, -2, dims=1)
        self.target_ret = (next_next_open - next_open) / (next_open + 1e-6)
        self.target_ret[:, -2:] = 0.0
        self.target_ret = clean_torch_tensor(self.target_ret)

        # 训练/测试集分割 (滚动窗口下测试集占20%)
        self.split_idx = int(opens.shape[1] * 0.8)
        log.info(f"数据加载完成 - 有效股票数: {self.num_stocks}, 交易日数: {self.time_dim}")
        return self

# ========== 截面因子挖掘器 ==========
class JQCrossSectionMiner:
    def __init__(self, engine):
        self.engine = engine
        self.model = AlphaGPT().to(g.DEVICE)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.best_sharpe = -10.0
        self.best_formula_tokens = None

    def get_strict_mask(self, open_slots, step):
        B = open_slots.shape[0]
        mask = torch.full((B, g.VOCAB_SIZE), float('-inf'), device=g.DEVICE)
        remaining_steps = g.MAX_SEQ_LEN - step
        done_mask = (open_slots == 0)
        mask[done_mask, 0] = 0.0
        active_mask = ~done_mask
        must_pick_feat = (open_slots >= remaining_steps)
        mask[active_mask, :len(g.FEATURES)] = 0.0
        can_pick_op_mask = active_mask & (~must_pick_feat)
        if can_pick_op_mask.any():
            mask[can_pick_op_mask, len(g.FEATURES):] = 0.0
        return mask

    def solve_one(self, tokens):
        stack = []
        try:
            for t in reversed(tokens):
                if t < len(g.FEATURES):
                    stack.append(self.engine.feat_data[t])
                else:
                    arity = g.OP_ARITY_MAP[t]
                    if len(stack) < arity:
                        raise ValueError
                    args = [stack.pop() for _ in range(arity)]
                    func = g.OP_FUNC_MAP[t]
                    res = func(*args)
                    res = clean_torch_tensor(res)
                    stack.append(res)
            if len(stack) >= 1:
                final = stack[-1]
                if final.std() < 1e-4:
                    return None
                if final.shape[1] != self.engine.time_dim:
                    if final.shape[1] > self.engine.time_dim:
                        final = final[:, :self.engine.time_dim]
                    else:
                        pad_len = self.engine.time_dim - final.shape[1]
                        final = torch.cat([final, torch.zeros(final.shape[0], pad_len, device=final.device)], dim=1)
                return final
        except:
            return None
        return None

    def backtest(self, factors):
        """截面因子回测"""
        split = self.engine.split_idx
        if split >= factors.shape[2]:
            split = factors.shape[2] - 1
        target = self.engine.target_ret[:, :split]
        B = factors.shape[0]
        rewards = torch.full((B,), -2.0, device=g.DEVICE)

        for i in range(B):
            f = factors[i, :, :split]
            f = clean_torch_tensor(f)
            if torch.all(f == 0):
                continue
            
            # 截面选股
            rank = torch.argsort(f, dim=0, descending=True)
            long_mask = torch.zeros_like(f)
            top_k = min(g.TOP_K, self.engine.num_stocks)
            for k in range(top_k):
                long_mask.scatter_(0, rank[k:k+1, :], 1.0)
            pos = long_mask / top_k

            # 换手率计算
            turnover = torch.abs(pos - torch.roll(pos, 1, dims=1))
            turnover[:, 0] = 0.0
            port_ret = (pos * target).sum(dim=0) - turnover.sum(dim=0) * g.COST_RATE

            # 夏普比率
            mu = port_ret.mean()
            std = port_ret.std() + 1e-6
            sharpe = mu / std * 15.87

            # IC信息系数计算
            ic_series = []
            for t in range(split):
                valid_mask = ~torch.isnan(f[:, t]) & ~torch.isnan(target[:, t])
                if valid_mask.sum() > 5:
                    factor_vals = f[valid_mask, t]
                    target_vals = target[valid_mask, t]
                    try:
                        if factor_vals.std() < 1e-6 or target_vals.std() < 1e-6:
                            continue
                        ic = torch.corrcoef(torch.stack([factor_vals, target_vals]))[0, 1]
                        if not torch.isnan(ic) and not torch.isinf(ic):
                            ic_series.append(ic)
                    except:
                        continue
            
            ir = 0.0
            if len(ic_series) > 0:
                ic_mean = torch.tensor(ic_series).mean()
                ic_std = torch.tensor(ic_series).std() + 1e-6
                ir = ic_mean / ic_std

            # 综合评分
            combined_score = sharpe + ir * 2.0
            if mu < 0:
                combined_score -= 2.0
            rewards[i] = combined_score
            
        return torch.clamp(rewards, -3, 10)

    def train(self):
        """训练模型"""
        log.info(f" 开始训练截面因子挖掘模型...")
        for iter_idx in range(g.TRAIN_ITERATIONS):
            B = min(g.BATCH_SIZE, 32)
            open_slots = torch.ones(B, dtype=torch.long, device=g.DEVICE)
            log_probs, tokens = [], []
            curr_inp = torch.zeros((B, 1), dtype=torch.long, device=g.DEVICE)
            
            for step in range(g.MAX_SEQ_LEN):
                logits, _ = self.model(curr_inp)
                mask = self.get_strict_mask(open_slots, step)
                dist = Categorical(logits=(logits + mask))
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                tokens.append(action)
                curr_inp = torch.cat([curr_inp, action.unsqueeze(1)], dim=1)
                
                is_op = action >= len(g.FEATURES)
                arity_tens = torch.zeros(g.VOCAB_SIZE, dtype=torch.long, device=g.DEVICE)
                for k, v in g.OP_ARITY_MAP.items():
                    arity_tens[k] = v
                delta = torch.where(is_op, arity_tens[action] - 1, torch.tensor(-1, device=g.DEVICE))
                open_slots += delta

            seqs = torch.stack(tokens, dim=1)
            with torch.no_grad():
                f_batch = []
                valid_mask = []
                target_shape = (self.engine.num_stocks, self.engine.time_dim)
                
                for i in range(B):
                    res = self.solve_one(seqs[i].cpu().tolist())
                    if res is not None and res.shape == target_shape:
                        f_batch.append(res)
                        valid_mask.append(True)
                    else:
                        valid_mask.append(False)
                
                rewards = torch.full((B,), -1.0, device=g.DEVICE)
                if len(f_batch) > 0:
                    factors = torch.stack(f_batch)
                    scores = self.backtest(factors)
                    valid_idx = [i for i, v in enumerate(valid_mask) if v]
                    rewards[valid_idx] = scores
                    
                    best_id = torch.argmax(scores)
                    if scores[best_id] > self.best_sharpe:
                        self.best_sharpe = scores[best_id].item()
                        self.best_formula_tokens = seqs[valid_idx[best_id]].cpu().tolist()
                        log.info(f" 迭代 {iter_idx+1} 找到更优因子，评分: {self.best_sharpe:.2f}")

            # 强化学习损失计算
            adv = rewards - rewards.mean()
            loss = -(torch.stack(log_probs, 1).sum(1) * adv).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            if (iter_idx + 1) % 10 == 0:
                log.info(f"训练进度: {iter_idx+1}/{g.TRAIN_ITERATIONS} | 最优综合评分: {self.best_sharpe:.2f}")

    def decode(self, tokens=None):
        """解码因子公式"""
        if tokens is None:
            tokens = self.best_formula_tokens
        if tokens is None:
            return "N/A"
        stream = list(tokens)
        def _parse():
            if not stream:
                return ""
            t = stream.pop(0)
            if t < len(g.FEATURES):
                return g.FEATURES[t]
            args = [_parse() for _ in range(g.OP_ARITY_MAP[t])]
            return f"{g.VOCAB[t]}({','.join(args)})"
        try:
            return _parse()
        except:
            return "Invalid"

# ========== 因子分析函数 ==========
def cross_section_analysis(miner, engine):
    log.info("\n" + "="*50)
    log.info(" 截面因子表现分析")
    log.info("="*50)
    
    formula = miner.decode()
    log.info(f"挖掘到的最优截面因子公式: {formula}")
    
    factor_all = miner.solve_one(miner.best_formula_tokens)
    if factor_all is None:
        log.info(" 无有效因子公式")
        return
    
    split = engine.split_idx
    if split >= factor_all.shape[1]:
        split = factor_all.shape[1] - 1
    train_f = factor_all[:, :split].cpu().numpy()
    test_f = factor_all[:, split:].cpu().numpy()
    train_target = engine.target_ret[:, :split].cpu().numpy()
    test_target = engine.target_ret[:, split:].cpu().numpy()

    train_f = clean_numpy_array(train_f)
    test_f = clean_numpy_array(test_f)
    train_target = clean_numpy_array(train_target)
    test_target = clean_numpy_array(test_target)

    def analyze_factor_performance(factor, target, period_name):
        """
        修复关键：
        1. 增加quantile_returns非空检查
        2. 增加quantile_matrix维度检查
        3. 处理各种边界情况，避免索引越界
        """
        log.info(f"\n--- {period_name} ---")
        # IC分析
        ic_series = []
        for t in range(factor.shape[1]):
            valid_mask = ~np.isnan(factor[:, t]) & ~np.isnan(target[:, t])
            if valid_mask.sum() > 5:
                try:
                    factor_vals = factor[valid_mask, t]
                    target_vals = target[valid_mask, t]
                    if np.std(factor_vals) < 1e-6 or np.std(target_vals) < 1e-6:
                        continue
                    ic = np.corrcoef(factor_vals, target_vals)[0, 1]
                    if not np.isnan(ic) and not np.isinf(ic):
                        ic_series.append(ic)
                except:
                    continue
        
        if ic_series:
            ic_mean = np.mean(ic_series)
            ic_std = np.std(ic_series)
            ic_ir = ic_mean / (ic_std + 1e-6)
            ic_hit_rate = np.mean(np.array(ic_series) > 0)
            log.info(f"IC均值: {ic_mean:.4f} | IC标准差: {ic_std:.4f}")
            log.info(f"信息比率: {ic_ir:.4f} | IC胜率: {ic_hit_rate:.2%}")
        else:
            log.info("无有效IC数据")

        # 分位数收益 
        quantile_returns = []
        for t in range(factor.shape[1]):
            valid_mask = ~np.isnan(factor[:, t]) & ~np.isnan(target[:, t])
            if valid_mask.sum() > 5:
                valid_factor = factor[valid_mask, t]
                valid_target = target[valid_mask, t]
                try:
                    # 确保至少能分成2组
                    unique_vals = len(np.unique(valid_factor))
                    if unique_vals < 2:
                        continue
                    # 最多分成5组，最少分成2组
                    n_groups = min(5, unique_vals)
                    if n_groups < 2:
                        continue
                    # 使用qcut分组，处理重复值
                    groups = pd.qcut(valid_factor, n_groups, labels=False, duplicates='drop')
                    if len(np.unique(groups)) >= 2:
                        group_returns = [valid_target[groups == g].mean() for g in np.unique(groups)]
                        quantile_returns.append(group_returns)
                except Exception as e:
                    # 捕获所有异常，避免程序崩溃
                    continue
        
        # 增加多层维度检查
        if quantile_returns and len(quantile_returns) > 0:
            # 过滤掉长度不一致的记录
            min_len = min([len(x) for x in quantile_returns]) if quantile_returns else 0
            if min_len >= 2:
                # 只保留长度>=min_len的记录
                filtered_returns = [x[:min_len] for x in quantile_returns if len(x) >= min_len]
                if filtered_returns:
                    quantile_matrix = np.array(filtered_returns)
                    # 确保是二维数组且列数>=2
                    if len(quantile_matrix.shape) == 2 and quantile_matrix.shape[1] >= 2:
                        top_bottom = quantile_matrix[:, -1] - quantile_matrix[:, 0]
                        sharpe = np.mean(top_bottom)/(np.std(top_bottom)+1e-6)*np.sqrt(252)
                        log.info(f"多空收益均值: {np.mean(top_bottom):.4f} | 多空夏普: {sharpe:.4f}")
                    else:
                        log.info("分位数矩阵维度不足，跳过多空收益计算")
                else:
                    log.info("过滤后无有效分位数数据")
            else:
                log.info("分位数组数不足，跳过多空收益计算")
        else:
            log.info("无有效分位数收益数据")

    analyze_factor_performance(train_f, train_target, "训练集")
    analyze_factor_performance(test_f, test_target, "测试集")

# ========== 聚宽标准策略函数） ==========
def initialize(context):
    """策略初始化：聚宽标准配置"""
    # 聚宽标准配置
    set_benchmark('399101.XSHE')  # 中小综指作为基准
    set_option('use_real_price', True)
    log.set_level('order', 'error')
    
    # 交易成本设置
    set_order_cost(OrderCost(
        close_tax=0.001,          # 卖出印花税
        open_commission=0.0003,   # 买入佣金
        close_commission=0.0003,  # 卖出佣金
        min_commission=5          # 最低佣金5元
    ), type='stock')
    
    # 初始化全局变量
    context.last_train_month = -1  # 上一次训练的月份
    context.current_stock_list = []
    context.miner = None
    context.trade_count = 0
    context.engine = None
    
    log.info("策略初始化完成，等待每月调仓日进行滚动训练")

def is_stock_tradable(security, current_dt):
    """判断股票是否可交易"""
    try:
        # 尝试用聚宽原生API
        return not is_suspended(security, current_dt)
    except:
        # 保底方案
        df = get_price(
            security,
            start_date=current_dt,
            end_date=current_dt,
            frequency='daily',
            fields=['close']
        )
        return not df.empty

# ========== 核心调仓函数 (每月滚动训练+调仓) ==========
def handle_data(context, data):
    """聚宽标准调仓函数：每月滚动训练+调仓"""
    today = context.current_dt
    today_date = today.date()
    current_month = today.month
    current_year = today.year
    
    # 只在每月第一个交易日进行滚动训练和调仓
    if (today.day != 1) and (context.last_train_month == current_month):
        return
    
    # 更新训练月份标记
    context.last_train_month = current_month
    
    log.info(f"\n{'='*60}")
    log.info(f" 月度滚动训练 - {current_year}年{current_month}月")
    log.info(f"{'='*60}")
    
    # 1. 选股：中小综指最小市值20只
    context.current_stock_list = select_small_cap_stocks(context)
    if not context.current_stock_list:
        log.error("选股失败，本月不进行交易")
        return
    
    # 2. 设置滚动训练窗口：过去12个月
    end_date = today - pd.Timedelta(days=1)
    start_date = end_date - pd.DateOffset(months=g.TRAIN_WINDOW_MONTHS)
    
    # 3. 加载数据
    context.engine = JQCrossSectionDataEngine(
        stock_list=context.current_stock_list,
        start_date=start_date,
        end_date=end_date
    ).load()
    
    # 4. 训练因子挖掘模型
    context.miner = JQCrossSectionMiner(context.engine)
    context.miner.train()
    
    # 5. 因子分析
    formula = context.miner.decode()
    log.info(f"\n 本月最优截面因子公式: {formula}")
    log.info(f" 最优综合评分: {context.miner.best_sharpe:.2f}")
    cross_section_analysis(context.miner, context.engine)
    
    # 6. 执行调仓
    execute_trade(context, today)

def execute_trade(context, today):
    """执行调仓"""
    engine = context.engine
    miner = context.miner
    
    # 日志
    log.info(f"\n 开始月度调仓 - {today.date()} | 当前已交易次数: {context.trade_count}")
    
    # 获取当日因子值
    factor_all = miner.solve_one(miner.best_formula_tokens)
    if factor_all is None:
        log.info(" 无有效因子值，跳过调仓")
        return
    
    # 日期匹配
    try:
        dates_np = engine.dates.values
        today_np = np.datetime64(today)
        date_diff = np.abs((dates_np - today_np).astype('timedelta64[D]').astype(int))
        date_idx = np.argmin(date_diff)
        matched_date = pd.to_datetime(dates_np[date_idx]).date()
        log.info(f"匹配到最近日期: {matched_date} (相差 {date_diff[date_idx]} 天)")
    except Exception as e:
        log.info(f" 日期匹配出错: {e}，使用最后一个交易日")
        date_idx = engine.time_dim - 1 if engine.time_dim > 0 else 0
    
    # 确保索引有效
    date_idx = min(max(date_idx, 0), engine.time_dim - 1)
    
    # 截面选股
    factor_today = factor_all[:, date_idx].squeeze()
    factor_today = clean_torch_tensor(factor_today)
    rank = torch.argsort(factor_today, descending=True)
    
    # 适配实际股票数
    top_k = min(g.TOP_K, engine.num_stocks)
    top_indices = rank[:top_k].cpu().numpy()
    top_indices = [i for i in top_indices if i < len(engine.stocks)]
    top_stocks = [engine.stocks[i] for i in top_indices]
    
    log.info(f" 本月选股结果 - Top{len(top_stocks)}: {top_stocks}")
    
    # 真实交易逻辑 (基于基础order函数)
    current_positions = list(context.portfolio.positions.keys())
    log.info(f" 当前持仓: {current_positions}")
    
    # 1. 卖出非持仓股
    for stock in current_positions:
        if stock not in top_stocks:
            try:
                # 使用基础函数清仓
                safe_order_close_position(context, stock)
                context.trade_count += 1
            except Exception as e:
                log.info(f" 卖出 {stock} 失败: {str(e)[:50]}")
    
    # 2. 买入Top-K股票
    if len(top_stocks) == 0:
        log.info(" 无股票可买，跳过")
        return
    
    # 计算每只股票的目标仓位
    target_percent = 1.0 / len(top_stocks)
    
    for stock in top_stocks:
        try:
            # 判断股票是否可交易
            if is_stock_tradable(stock, today):
                # 使用基础函数按仓位比例下单
                success = safe_order_by_percent(context, stock, target_percent)
                if success:
                    context.trade_count += 1
            else:
                log.info(f" {stock} 停牌或数据异常，跳过买入")
        except Exception as e:
            log.info(f" 买入 {stock} 失败: {str(e)[:50]}")
    
    log.info(f"月度调仓完成 - 累计交易次数: {context.trade_count}")

# ========== 收盘后日志函数 ==========
def after_market_close(context):
    """收盘后打印成交记录"""
    log.info(f"函数运行时间(after_market_close): {str(context.current_dt.time())}")
    # 得到当天所有成交记录
    trades = get_trades()
    for _trade in trades.values():
        log.info(f"成交记录：{str(_trade)}")
    log.info("一天结束")
    log.info("##############################################################")