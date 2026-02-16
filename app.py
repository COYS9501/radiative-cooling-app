import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
import warnings
warnings.filterwarnings('ignore')
from io import BytesIO

# -------------------------- 全局配置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 物理常数（CODATA 2018）
H_PLANCK = 6.62607015e-34  # 普朗克常数 (J·s)
C_LIGHT = 299792458        # 光速 (m/s)
K_BOLTZMANN = 1.380649e-23 # 玻尔兹曼常数 (J/K)

# 默认数据文件路径（根据你的实际路径调整）
DEFAULT_SUN_FILE = 'AM15太阳辐射_处理后.csv'
DEFAULT_ATM_FILE = '大气透过率_处理后.csv'

# -------------------------- 基础函数（终极稳定版） --------------------------
import chardet

def load_default_data(file_path, desc):
    """加载默认CSV文件，自动检测编码"""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
        df = pd.read_csv(file_path, encoding=encoding)
        return df, f"✅ 加载成功：{desc}（{len(df)}行，编码：{encoding}）"
    except Exception as e:
        return pd.DataFrame(), f"❌ 加载失败：{str(e)}"

def planck_law(T_rad, lmbda_m):
    """普朗克黑体辐射定律计算
    T_rad: 温度 (K)
    lmbda_m: 波长 (m)
    返回：辐射出射度 (W/m³)
    """
    numerator = 2 * H_PLANCK * C_LIGHT**2 / (lmbda_m**5)
    denominator = np.exp(H_PLANCK * C_LIGHT / (lmbda_m * K_BOLTZMANN * T_rad)) - 1
    return numerator / denominator

def interpolate_curve(x_target, x_source, y_source, desc):
    """终极稳定版插值函数
    确保始终返回一维NumPy数组，解决所有类型错误
    """
    # 第一步：强制转换为数组
    x_target = np.asarray(x_target, dtype=np.float64)
    x_source = np.asarray(x_source, dtype=np.float64)
    y_source = np.asarray(y_source, dtype=np.float64)
    
    # 第二步：校验数据量
    if len(x_source) < 2 or len(y_source) < 2 or len(x_target) < 1:
        st.error(f"{desc}数据不足，无法插值，返回全零数组")
        return np.zeros(len(x_target), dtype=np.float64)
    
    # 第三步：过滤NaN和无穷值
    valid_mask = ~(np.isnan(x_source) | np.isnan(y_source) | np.isinf(x_source) | np.isinf(y_source))
    x_valid = x_source[valid_mask]
    y_valid = y_source[valid_mask]
    
    if len(x_valid) < 2:
        st.error(f"{desc}数据清洗后有效点数不足，返回全零数组")
        return np.zeros(len(x_target), dtype=np.float64)
    
    # 第四步：执行插值，确保返回一维数组
    try:
        f = interpolate.interp1d(x_valid, y_valid, bounds_error=False, fill_value='extrapolate')
        result = f(x_target)
        # 确保返回一维数组
        if result.ndim > 1:
            result = result.flatten()
        return np.asarray(result, dtype=np.float64)
    except Exception as e:
        st.error(f"{desc}插值失败：{str(e)}，返回全零数组")
        return np.zeros(len(x_target), dtype=np.float64)

# -------------------------- UI页面开发 --------------------------
st.title("🌞 辐射制冷净功率自动计算系统")
st.markdown("---")

# 侧边栏参数配置
st.sidebar.title("🔧 计算参数输入（默认值可修改）")
st.sidebar.markdown("### 1. 基础固定参数（默认值适配常规场景）")

# 入射角设置
theta_deg = st.sidebar.number_input(
    "入射角 θ（度）",
    value=0.0, step=1.0, min_value=0.0, max_value=90.0
)
theta_rad = np.radians(theta_deg)
st.sidebar.caption(f"当前θ（弧度）：{theta_rad:.4f} rad | cosθ：{np.cos(theta_rad):.4f}")

# 波长范围设置
lambda_min = st.sidebar.number_input(
    "波长下限（μm）",
    value=0.25, step=0.1, min_value=0.25, max_value=5.0
)
lambda_max = st.sidebar.number_input(
    "波长上限（μm）",
    value=25.0, step=1.0, min_value=10.0, max_value=25.0
)
st.sidebar.caption(f"最终计算波长范围：{lambda_min:.2f}-{lambda_max:.2f} μm")

# 数据文件配置
st.sidebar.markdown("### 2. 内置数据文件（支持自定义替换）")

# 太阳辐射数据
st.sidebar.subheader("太阳辐射数据（AM1.5）")
sun_df_default, sun_msg_default = load_default_data(DEFAULT_SUN_FILE, "AM1.5太阳辐射")
st.sidebar.caption(f"默认文件：{DEFAULT_SUN_FILE.split('/')[-1]} | {sun_msg_default}")
uploaded_sun = st.sidebar.file_uploader(
    "上传自定义太阳辐射CSV（仅需两列：第一列=波长(μm)，第二列=太阳辐射强度）",
    type="csv"
)

# 大气透过率数据
st.sidebar.subheader("大气透过率数据（τatm）")
atm_df_default, atm_msg_default = load_default_data(DEFAULT_ATM_FILE, "大气透过率")
st.sidebar.caption(f"默认文件：{DEFAULT_ATM_FILE.split('/')[-1]} | {atm_msg_default}")
uploaded_atm = st.sidebar.file_uploader(
    "上传自定义大气透过率CSV（仅需两列：第一列=波长(μm)，第二列=透过率τ）",
    type="csv"
)

# 昼夜模式与批量参数
st.sidebar.markdown("### 3. 昼夜模式与批量计算参数")
day_night = st.sidebar.radio("计算模式", ["白天（含太阳辐射）", "夜晚（无太阳辐射）"], index=0)
is_day = (day_night == "白天（含太阳辐射）")

# 环境温度Tamb设置
st.sidebar.subheader("环境温度 Tamb（K）")
tamb_min = st.sidebar.number_input("Tamb最小值", value=280.0 if not is_day else 290.0, step=1.0, min_value=250.0, max_value=330.0)
tamb_max = st.sidebar.number_input("Tamb最大值", value=290.0 if not is_day else 300.0, step=1.0, min_value=tamb_min, max_value=330.0)
tamb_step = st.sidebar.number_input("Tamb步长", value=5.0, step=1.0, min_value=0.5, max_value=10.0)
tamb_list = np.arange(tamb_min, tamb_max + tamb_step/2, tamb_step).round(2)
st.sidebar.caption(f"Tamb计算列表：{tamb_list} K")

# 冷却器温度Trad设置
st.sidebar.subheader("辐射冷却器温度 Trad（K）")
trad_min = st.sidebar.number_input("Trad最小值", value=270.0, step=1.0, min_value=250.0, max_value=tamb_max)
trad_max = st.sidebar.number_input("Trad最大值", value=285.0, step=1.0, min_value=trad_min, max_value=tamb_max)
trad_step = st.sidebar.number_input("Trad步长", value=2.0, step=0.5, min_value=0.5, max_value=5.0)
trad_list = np.arange(trad_min, trad_max + trad_step/2, trad_step).round(2)
st.sidebar.caption(f"Trad计算列表：{trad_list} K")

# 对流换热系数q设置
st.sidebar.subheader("对流换热系数 q（W/(m²·K)）")
q_min = st.sidebar.number_input("q最小值", value=3.0, step=0.5, min_value=0.5, max_value=20.0)
q_max = st.sidebar.number_input("q最大值", value=8.0, step=0.5, min_value=q_min, max_value=20.0)
q_step = st.sidebar.number_input("q步长", value=1.0, step=0.5, min_value=0.5, max_value=5.0)
q_list = np.arange(q_min, q_max + q_step/2, q_step).round(2)
st.sidebar.caption(f"q计算列表：{q_list} W/(m²·K)")

# 发射率数据上传（必需）
st.sidebar.markdown("### 4. 辐射冷却器发射率数据（必需）")
uploaded_eps = st.sidebar.file_uploader(
    "上传发射率CSV（仅需两列：第一列=波长(μm)，第二列=发射率ε）",
    type="csv", accept_multiple_files=False
)

# 处理发射率数据
if uploaded_eps:
    try:
        file_content = uploaded_eps.getvalue()
        result = chardet.detect(file_content)
        encoding = result['encoding'] or 'utf-8'
        eps_df = pd.read_csv(BytesIO(file_content), encoding=encoding)
        
        if len(eps_df.columns) != 2:
            st.sidebar.error(f"❌ 发射率CSV需为**两列数据**（波长+发射率），当前列数：{len(eps_df.columns)}列")
            eps_df = pd.DataFrame()
        else:
            # 重命名列并清洗数据
            original_cols = eps_df.columns.tolist()
            eps_df.columns = ["波长_μm", "发射率ε"]
            
            eps_df["波长_μm"] = pd.to_numeric(eps_df["波长_μm"], errors='coerce')
            eps_df["发射率ε"] = pd.to_numeric(eps_df["发射率ε"], errors='coerce')
            eps_df_clean = eps_df.dropna()
            
            if len(eps_df_clean) == 0:
                st.sidebar.error("❌ 数据中无有效数值！请检查：\n1. 列1是否为波长（数字）\n2. 列2是否为发射率（数字）")
                eps_df = pd.DataFrame()
            else:
                # 发射率限制在0-1之间
                eps_df_clean["发射率ε"] = eps_df_clean["发射率ε"].clip(0, 1)
                eps_df_clean = eps_df_clean.sort_values("波长_μm").reset_index(drop=True)
                
                st.sidebar.success(
                    f"✅ 发射率数据加载成功！\n"
                    f"📌 列名映射：\n"
                    f"  原始列1「{original_cols[0]}」→ 波长_μm\n"
                    f"  原始列2「{original_cols[1]}」→ 发射率ε\n"
                    f"📊 有效数据：{len(eps_df_clean)}行\n"
                    f"📏 波长范围：{eps_df_clean['波长_μm'].min():.2f}-{eps_df_clean['波长_μm'].max():.2f}μm\n"
                    f"📈 发射率范围：{eps_df_clean['发射率ε'].min():.3f}-{eps_df_clean['发射率ε'].max():.3f}"
                )
                eps_df = eps_df_clean
    except Exception as e:
        st.sidebar.error(f"❌ 发射率数据加载失败：{str(e)}")
        eps_df = pd.DataFrame()
else:
    st.sidebar.warning(
        "请上传发射率CSV文件（示例格式，列名可自定义）：\n"
        "Wavelength,Emissivity\n"
        "0.3,0.1\n"
        "8.0,0.95\n"
        "15.0,0.98"
    )
    eps_df = pd.DataFrame()

# ================================= 主页面输出区 =================================
st.markdown("### 📊 计算条件汇总")
with st.expander("点击查看当前计算参数（确认后再运行）", expanded=True):
    cond_data = {
        "参数类别": [
            "基础参数", "基础参数", "基础参数",
            "数据文件", "数据文件", "数据文件",
            "计算模式", "温度参数", "温度参数", "换热系数"
        ],
        "参数名称": [
            "入射角θ", "计算波长范围", "物理常数标准",
            "太阳辐射文件", "大气透过率文件", "发射率文件",
            "昼夜模式", "Tamb计算列表（K）", "Trad计算列表（K）", "q计算列表（W/(m²·K)）"
        ],
        "当前值": [
            f"{theta_deg:.1f}°（cosθ={np.cos(theta_rad):.4f}）",
            f"{lambda_min:.2f}-{lambda_max:.2f} μm",
            "CODATA 2018（h=6.626e-34 J·s, c=2.998e8 m/s）",
            uploaded_sun.name if uploaded_sun else DEFAULT_SUN_FILE.split('/')[-1],
            uploaded_atm.name if uploaded_atm else DEFAULT_ATM_FILE.split('/')[-1],
            uploaded_eps.name if uploaded_eps else "未上传（必需）",
            day_night,
            f"{tamb_list}（共{len(tamb_list)}个点）",
            f"{trad_list}（共{len(trad_list)}个点）",
            f"{q_list}（共{len(q_list)}个点）"
        ]
    }
    st.dataframe(pd.DataFrame(cond_data), use_container_width=True)

# 检查是否可以开始计算
can_calculate = (len(eps_df) > 0) and (len(tamb_list) > 0) and (len(trad_list) > 0) and (len(q_list) > 0)
if not can_calculate:
    st.warning("请完成必需输入：1. 上传发射率CSV；2. 确认Tamb/Trad/q的范围和步长（确保列表非空）")

# 计算按钮
calculate_btn = st.button("🚀 开始批量计算辐射制冷净功率", disabled=not can_calculate)

if calculate_btn:
    with st.spinner("正在计算...（批量计算可能需要10-30秒，请耐心等待）"):
        # 1. 加载太阳辐射数据
        if uploaded_sun:
            try:
                file_content = uploaded_sun.getvalue()
                result = chardet.detect(file_content)
                encoding = result['encoding'] or 'utf-8'
                sun_df = pd.read_csv(BytesIO(file_content), encoding=encoding)
                
                if len(sun_df.columns) != 2:
                    st.error("❌ 太阳辐射CSV需为两列数据（波长+太阳辐射强度）")
                    st.stop()
                original_sun_cols = sun_df.columns.tolist()
                sun_df.columns = ["波长_μm", "太阳辐射强度_Wm-2μm-1"]
                
                # 数据清洗
                sun_df["波长_μm"] = pd.to_numeric(sun_df["波长_μm"], errors='coerce')
                sun_df["太阳辐射强度_Wm-2μm-1"] = pd.to_numeric(sun_df["太阳辐射强度_Wm-2μm-1"], errors='coerce')
                sun_df = sun_df.dropna()
                if len(sun_df) == 0:
                    st.error("❌ 太阳辐射数据无有效数值")
                    st.stop()
            except Exception as e:
                st.error(f"自定义太阳辐射文件加载失败：{str(e)}")
                st.stop()
        else:
            if sun_df_default.empty:
                st.error("默认太阳辐射文件加载失败，请检查文件路径或上传自定义文件")
                st.stop()
            sun_df = sun_df_default
            if len(sun_df.columns) == 2:
                sun_df.columns = ["波长_μm", "太阳辐射强度_Wm-2μm-1"]

        # 2. 加载大气透过率数据
        if uploaded_atm:
            try:
                file_content = uploaded_atm.getvalue()
                result = chardet.detect(file_content)
                encoding = result['encoding'] or 'utf-8'
                atm_df = pd.read_csv(BytesIO(file_content), encoding=encoding)
                
                if len(atm_df.columns) != 2:
                    st.error("❌ 大气透过率CSV需为两列数据（波长+透过率τ）")
                    st.stop()
                original_atm_cols = atm_df.columns.tolist()
                atm_df.columns = ["波长_μm", "大气透过率_τatm"]
                
                # 数据清洗
                atm_df["波长_μm"] = pd.to_numeric(atm_df["波长_μm"], errors='coerce')
                atm_df["大气透过率_τatm"] = pd.to_numeric(atm_df["大气透过率_τatm"], errors='coerce')
                atm_df = atm_df.dropna()
                atm_df["大气透过率_τatm"] = atm_df["大气透过率_τatm"].clip(0, 1)
                if len(atm_df) == 0:
                    st.error("❌ 大气透过率数据无有效数值")
                    st.stop()
            except Exception as e:
                st.error(f"自定义大气透过率文件加载失败：{str(e)}")
                st.stop()
        else:
            if atm_df_default.empty:
                st.error("默认大气透过率文件加载失败，请检查文件路径或上传自定义文件")
                st.stop()
            atm_df = atm_df_default
            if len(atm_df.columns) == 2:
                atm_df.columns = ["波长_μm", "大气透过率_τatm"]

        # 3. 生成统一波长网格（强制一维数组）
        lambda_grid = np.arange(lambda_min, lambda_max + 0.005, 0.01).round(2)
        lambda_grid = np.asarray(lambda_grid, dtype=np.float64)  # 强制数组
        st.success(f"生成统一波长网格：{len(lambda_grid)}个点（{lambda_min:.2f}-{lambda_max:.2f}μm，间隔0.01μm）")

        # 4. 所有曲线插值到统一网格（强制一维数组）
        # 发射率插值
        eps_interp = interpolate_curve(lambda_grid, eps_df["波长_μm"], eps_df["发射率ε"], "发射率")
        eps_interp = np.asarray(eps_interp, dtype=np.float64).flatten()  # 强制一维

        # 大气透过率插值（先清洗）
        atm_df["波长_μm"] = pd.to_numeric(atm_df["波长_μm"], errors='coerce')
        atm_df["大气透过率_τatm"] = pd.to_numeric(atm_df["大气透过率_τatm"], errors='coerce')
        atm_df_clean = atm_df.dropna(subset=["波长_μm", "大气透过率_τatm"])
        if len(atm_df_clean) < 2:
            st.error("❌ 大气透过率数据清洗后有效点数不足（<2），无法插值！请检查文件是否包含有效数值。")
            st.stop()
        tau_atm_interp = interpolate_curve(lambda_grid, atm_df_clean["波长_μm"], atm_df_clean["大气透过率_τatm"], "大气透过率")
        tau_atm_interp = np.asarray(tau_atm_interp, dtype=np.float64).flatten()  # 强制一维

        # 太阳辐射插值（先清洗，仅白天用，强制一维）
        sun_interp = np.zeros(len(lambda_grid), dtype=np.float64)  # 初始化默认值
        if is_day:
            sun_df["波长_μm"] = pd.to_numeric(sun_df["波长_μm"], errors='coerce')
            sun_df["太阳辐射强度_Wm-2μm-1"] = pd.to_numeric(sun_df["太阳辐射强度_Wm-2μm-1"], errors='coerce')
            sun_df_clean = sun_df.dropna(subset=["波长_μm", "太阳辐射强度_Wm-2μm-1"])
            if len(sun_df_clean) < 2:
                st.error("❌ 太阳辐射数据清洗后有效点数不足（<2），无法插值！请检查文件是否包含有效数值。")
                st.stop()
            sun_interp = interpolate_curve(lambda_grid, sun_df_clean["波长_μm"], sun_df_clean["太阳辐射强度_Wm-2μm-1"], "太阳辐射")
            sun_interp = np.asarray(sun_interp, dtype=np.float64).flatten()  # 强制一维

        # 5. 批量计算净功率
        result_list = []
        # 构建插值函数（用于积分）
        eps_interp_func = interpolate.interp1d(lambda_grid, eps_interp, bounds_error=False, fill_value='extrapolate')
        tau_atm_interp_func = interpolate.interp1d(lambda_grid, tau_atm_interp, bounds_error=False, fill_value='extrapolate')
        
        for tamb in tamb_list:
            for trad in trad_list:
                for q in q_list:
                    # 计算P_rad（材料辐射功率）
                    def p_rad_integrand(lmbda_μm):
                        lmbda_m = lmbda_μm * 1e-6  # 转换为米
                        ibb = planck_law(trad, lmbda_m)
                        eps = eps_interp_func(lmbda_μm)
                        return ibb * eps * np.cos(theta_rad) * 1e6  # 单位转换

                    p_rad, _ = integrate.quad(p_rad_integrand, lambda_min, lambda_max)
                    p_rad *= 2 * np.pi  # 立体角积分

                    # 计算P_atm（大气逆辐射）
                    def p_atm_integrand(lmbda_μm):
                        lmbda_m = lmbda_μm * 1e-6
                        ibb = planck_law(tamb, lmbda_m)
                        eps = eps_interp_func(lmbda_μm)
                        tau_atm = tau_atm_interp_func(lmbda_μm)
                        cos_theta = np.cos(theta_rad)
                        
                        # 避免除零错误
                        if cos_theta < 1e-6:
                            eps_atm = 0.9
                        else:
                            tau_atm = max(tau_atm, 1e-8)  # 避免0的幂次
                            eps_atm = 1 - (tau_atm ** (1 / cos_theta))
                        return ibb * eps * eps_atm * cos_theta * 1e6

                    p_atm, _ = integrate.quad(p_atm_integrand, lambda_min, lambda_max)
                    p_atm *= 2 * np.pi

                    # 计算P_sun（太阳辐射，终极安全校验）
                    if is_day:
                        try:
                            # 终极保险：强制转换+维度检查
                            sun_interp_check = np.asarray(sun_interp, dtype=np.float64).flatten()
                            eps_interp_check = np.asarray(eps_interp, dtype=np.float64).flatten()
                            lambda_grid_check = np.asarray(lambda_grid, dtype=np.float64).flatten()
                            
                            # 检查所有数组都是一维且长度相同
                            if (sun_interp_check.ndim == 1 and 
                                eps_interp_check.ndim == 1 and 
                                lambda_grid_check.ndim == 1):
                                
                                if (len(sun_interp_check) == len(eps_interp_check) == len(lambda_grid_check)):
                                    # 执行积分计算
                                    p_sun = integrate.trapz(sun_interp_check * eps_interp_check, lambda_grid_check)
                                else:
                                    st.warning(f"数组长度不匹配（太阳辐射：{len(sun_interp_check)}，发射率：{len(eps_interp_check)}，波长网格：{len(lambda_grid_check)}），P_sun按0计算")
                                    p_sun = 0.0
                            else:
                                st.warning("数组维度错误（需一维数组），P_sun按0计算")
                                p_sun = 0.0
                        except Exception as e:
                            st.error(f"P_sun计算失败：{str(e)}，P_sun按0计算")
                            p_sun = 0.0
                    else:
                        p_sun = 0.0

                    # 计算非辐射损失
                    p_cond_conv = q * (tamb - trad)
                    # 计算净制冷功率
                    p_net = p_rad - p_atm - p_sun - p_cond_conv

                    # 保存结果
                    result_list.append({
                        "昼夜模式": day_night,
                        "环境温度Tamb（K）": tamb,
                        "冷却器温度Trad（K）": trad,
                        "对流换热系数q（W/(m²·K)）": q,
                        "材料辐射功率P_rad（W/m²）": round(p_rad, 2),
                        "大气逆辐射P_atm（W/m²）": round(p_atm, 2),
                        "太阳辐射P_sun（W/m²）": round(p_sun, 2) if is_day else 0.0,
                        "非辐射损失P_cond+conv（W/m²）": round(p_cond_conv, 2),
                        "净制冷功率P_net（W/m²）": round(p_net, 2),
                        "制冷状态": "✅ 制冷" if p_net > 0 else "❌ 不制冷"
                    })

        # 6. 结果处理与展示
        result_df = pd.DataFrame(result_list)
        st.markdown("### 📈 批量计算结果（共{}组数据）".format(len(result_df)))

        # 完整结果表格
        with st.expander("查看完整结果表格", expanded=False):
            st.dataframe(result_df, use_container_width=True, height=400)

        # 可视化：P_net随Trad变化
        st.markdown("### 📊 净功率P_net随Trad变化曲线（固定中间Tamb和q）")
        tamb_mid = tamb_list[len(tamb_list)//2]
        q_mid = q_list[len(q_list)//2]
        plot_df = result_df[(result_df["环境温度Tamb（K）"] == tamb_mid) & (result_df["对流换热系数q（W/(m²·K)）"] == q_mid)]
        
        if len(plot_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(plot_df["冷却器温度Trad（K）"], plot_df["净制冷功率P_net（W/m²）"],
                    'o-', color='darkred', linewidth=2, markersize=6, label=f"Tamb={tamb_mid}K, q={q_mid}W/(m²·K)")
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, label="P_net=0（制冷临界点）")
            
            # 标注最大净功率点
            max_pnet_idx = plot_df["净制冷功率P_net（W/m²）"].idxmax()
            max_pnet_row = plot_df.loc[max_pnet_idx]
            ax.scatter(max_pnet_row["冷却器温度Trad（K）"], max_pnet_row["净制冷功率P_net（W/m²）"],
                      color='gold', s=100, zorder=5, label=f"最大P_net={max_pnet_row['净制冷功率P_net（W/m²）']:.2f}W/m²（Trad={max_pnet_row['冷却器温度Trad（K）']}K）")
            
            ax.set_xlabel("辐射冷却器温度 Trad（K）", fontsize=12)
            ax.set_ylabel("净制冷功率 P_net（W/m²）", fontsize=12)
            ax.set_title(f"{day_night}：P_net随Trad变化（Tamb={tamb_mid}K, q={q_mid}W/(m²·K)）", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("无匹配的可视化数据（请检查Tamb/q的中间值是否在计算列表中）")

        # 结果下载（适配本地环境）
        st.markdown("### 📥 结果下载")
        excel_file = "辐射制冷功率计算结果.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            result_df.to_excel(writer, sheet_name=day_night, index=False)
            cond_df = pd.DataFrame(cond_data)
            cond_df.to_excel(writer, sheet_name="计算条件", index=False)
        
        with open(excel_file, 'rb') as f:
            st.download_button(
                label=f"下载{day_night}计算结果（Excel，含{len(result_df)}组数据）",
                data=f,
                file_name=f"辐射制冷功率计算结果_{day_night.replace('（', '_').replace('）', '')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # 关键统计
        st.markdown("### 📊 关键统计")
        total_cooling = len(result_df[result_df["制冷状态"] == "✅ 制冷"])
        max_pnet = result_df["净制冷功率P_net（W/m²）"].max()
        min_pnet = result_df["净制冷功率P_net（W/m²）"].min()
        st.info(f"""
        - 总计算组数：{len(result_df)} 组
        - 实现制冷的组数：{total_cooling} 组（占比 {total_cooling/len(result_df)*100:.1f}%）
        - 最大净制冷功率：{max_pnet:.2f} W/m²（对应Tamb={result_df[result_df['净制冷功率P_net（W/m²）']==max_pnet]['环境温度Tamb（K）'].iloc[0]}K, Trad={result_df[result_df['净制冷功率P_net（W/m²）']==max_pnet]['冷却器温度Trad（K）'].iloc[0]}K）
        - 最小净制冷功率：{min_pnet:.2f} W/m²
        """)
