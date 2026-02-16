import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
import warnings
import os
warnings.filterwarnings('ignore')
from io import BytesIO

# -------------------------- 全局配置 & 图表乱码终极修复 --------------------------
# 方案1：尝试系统中可用的中文字体
def get_chinese_font():
    """自动查找系统中可用的中文字体，解决乱码"""
    from matplotlib import font_manager
    font_names = [
        'WenQuanYi Micro Hei',  # Linux/Streamlit Cloud
        'SimHei',                # Windows
        'Microsoft YaHei',       # Windows
        'PingFang SC',           # Mac
        'Arial Unicode MS',      # Mac备用
        'DejaVu Sans'            # 最后兜底
    ]
    
    system_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for name in font_names:
        if name in system_fonts:
            return name
    return 'DejaVu Sans'

# 应用字体配置
CHINESE_FONT = get_chinese_font()
plt.rcParams['font.sans-serif'] = [CHINESE_FONT]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 120  # 提高图表清晰度

# 物理常数（固定不变，行业标准值）
H_PLANCK = 6.62607015e-34  # J·s
C_LIGHT = 299792458        # m/s
K_BOLTZMANN = 1.380649e-23 # J/K
SIGMA_STEFAN = 5.670374419e-8 # W/(m²·K^4)

# 温度转换便捷函数
def c_to_k(c):
    return c + 273.15
def k_to_c(k):
    return k - 273.15

# scipy版本全兼容
try:
    from scipy.integrate import trapezoid
except ImportError:
    from scipy.integrate import trapz as trapezoid

# 默认数据文件路径
DEFAULT_SUN_FILE = 'AM15太阳辐射_处理后.csv'
DEFAULT_ATM_FILE = '大气透过率_处理后.csv'

# -------------------------- 核心函数 --------------------------
import chardet

def load_and_clean_csv(file_path_or_buffer, desc, required_cols=2):
    """通用CSV加载&清洗函数"""
    try:
        # 读取文件&检测编码
        if isinstance(file_path_or_buffer, str):
            if not os.path.exists(file_path_or_buffer):
                return pd.DataFrame(), f"❌ 文件不存在：{file_path_or_buffer}"
            with open(file_path_or_buffer, 'rb') as f:
                file_content = f.read()
        else:
            file_content = file_path_or_buffer.getvalue()
        
        result = chardet.detect(file_content)
        encoding = result['encoding'] or 'utf-8'
        df = pd.read_csv(BytesIO(file_content), encoding=encoding)
        
        if len(df.columns) != required_cols:
            return pd.DataFrame(), f"❌ 需为{required_cols}列，当前列数：{len(df.columns)}"
        
        df.columns = ["波长_μm", "数值"]
        df["波长_μm"] = pd.to_numeric(df["波长_μm"], errors='coerce')
        df["数值"] = pd.to_numeric(df["数值"], errors='coerce')
        df_clean = df.dropna().reset_index(drop=True)
        
        if len(df_clean) < 2:
            return pd.DataFrame(), f"❌ 有效数据不足2行"
        
        if df_clean["波长_μm"].min() < 0 or df_clean["数值"].min() < 0:
            return pd.DataFrame(), f"❌ 包含负数"
        
        return df_clean, f"✅ 成功（{len(df_clean)}行）"
    
    except Exception as e:
        return pd.DataFrame(), f"❌ 加载失败：{str(e)}"

def planck_law(T_rad, lmbda_m):
    lmbda_m = np.maximum(lmbda_m, 1e-20)
    exponent = H_PLANCK * C_LIGHT / (lmbda_m * K_BOLTZMANN * np.maximum(T_rad, 1e-10))
    exponent = np.minimum(exponent, 700)
    numerator = 2 * H_PLANCK * C_LIGHT**2 / (lmbda_m**5)
    denominator = np.exp(exponent) - 1
    denominator = np.maximum(denominator, 1e-10)
    return numerator / denominator

def interpolate_curve(x_target, x_source, y_source, desc):
    try:
        x_target = np.asarray(x_target, dtype=np.float64).flatten()
        x_source = np.asarray(x_source, dtype=np.float64).flatten()
        y_source = np.asarray(y_source, dtype=np.float64).flatten()
    except Exception as e:
        return np.zeros(len(x_target), dtype=np.float64)
    
    if len(x_source) < 2 or len(y_source) < 2 or len(x_target) < 1:
        return np.zeros(len(x_target), dtype=np.float64)
    
    valid_mask = ~(np.isnan(x_source) | np.isnan(y_source) | np.isinf(x_source) | np.isinf(y_source))
    x_valid = x_source[valid_mask]
    y_valid = y_source[valid_mask]
    
    if len(x_valid) < 2:
        return np.zeros(len(x_target), dtype=np.float64)
    
    try:
        f = interpolate.interp1d(x_valid, y_valid, bounds_error=False, fill_value='extrapolate')
        return np.asarray(f(x_target).flatten(), dtype=np.float64)
    except Exception as e:
        return np.zeros(len(x_target), dtype=np.float64)

# -------------------------- UI页面 --------------------------
st.title("🌞 辐射制冷净功率自动计算系统")
st.markdown("---")

# 侧边栏参数配置
st.sidebar.title("🔧 计算参数配置")
st.sidebar.markdown("### 1. 基础参数")

# 入射角
theta_deg = st.sidebar.number_input(
    "入射角 θ（度）",
    value=0.0, step=1.0, min_value=0.0, max_value=90.0
)
theta_rad = np.radians(theta_deg)
cos_theta = np.cos(theta_rad)

# 波长范围
lambda_min = st.sidebar.number_input(
    "波长下限（μm）", value=0.25, step=0.1, min_value=0.2, max_value=5.0
)
lambda_max = st.sidebar.number_input(
    "波长上限（μm）", value=25.0, step=1.0, min_value=10.0, max_value=30.0
)

# 数据文件配置
st.sidebar.markdown("### 2. 数据文件（支持自定义上传）")

# 太阳辐射数据
st.sidebar.subheader("太阳辐射数据（AM1.5）")
uploaded_sun = st.sidebar.file_uploader(
    "上传自定义太阳辐射CSV", type="csv", key="sun_upload"
)

# 大气透过率数据
st.sidebar.subheader("大气透过率数据")
uploaded_atm = st.sidebar.file_uploader(
    "上传自定义大气透过率CSV", type="csv", key="atm_upload"
)

# 计算模式与温度参数
st.sidebar.markdown("### 3. 计算模式与温度")
day_night = st.sidebar.radio("计算模式", ["白天（含太阳辐射）", "夜晚（无太阳辐射）"], index=0)
is_day = (day_night == "白天（含太阳辐射）")

# --- 修改点3：Tamb改为固定值，分昼夜自动切换默认值 ---
st.sidebar.subheader("环境温度 Tamb")
# 白天默认30°C (303.15K)，晚上默认15°C (288.15K)
default_tamb_k = c_to_k(30.0) if is_day else c_to_k(15.0)
default_tamb_c = k_to_c(default_tamb_k)

tamb_k = st.sidebar.number_input(
    f"Tamb（K）",
    value=default_tamb_k, step=0.5, min_value=200.0, max_value=350.0,
    help=f"默认值：{default_tamb_c:.1f}°C ({default_tamb_k:.2f}K)"
)
tamb_list = np.array([tamb_k]) # 只有一个值，但保持数组格式兼容代码
st.sidebar.caption(f"当前温度：{k_to_c(tamb_k):.2f}°C")

# --- 修改点4：Trad无限制扫描，默认273-313K，步长5K ---
st.sidebar.subheader("冷却器温度 Trad（扫描范围）")
trad_min = st.sidebar.number_input(
    "Trad最小值（K）", value=273.0, step=1.0,
    help="无上限限制，可自由设置"
)
trad_max = st.sidebar.number_input(
    "Trad最大值（K）", value=313.0, step=1.0, min_value=trad_min,
    help="无上限限制，可自由设置"
)
trad_step = st.sidebar.number_input(
    "Trad步长（K）", value=5.0, step=0.5, min_value=0.1
)
trad_list = np.arange(trad_min, trad_max + trad_step/2, trad_step).round(2)
st.sidebar.caption(f"Trad扫描列表：{trad_list} K")

# 对流换热系数q
st.sidebar.subheader("对流换热系数 q（W/(m²·K)）")
q_min = st.sidebar.number_input("q最小值", value=3.0, step=0.5, min_value=0.0, max_value=20.0)
q_max = st.sidebar.number_input("q最大值", value=8.0, step=0.5, min_value=q_min, max_value=20.0)
q_step = st.sidebar.number_input("q步长", value=1.0, step=0.5, min_value=0.5, max_value=5.0)
q_list = np.arange(q_min, q_max + q_step/2, q_step).round(2)

# 发射率数据（必需）
st.sidebar.markdown("### 4. 冷却器发射率数据（必需）")
uploaded_eps = st.sidebar.file_uploader(
    "上传发射率CSV（两列：波长μm、发射率0-1）",
    type="csv", key="eps_upload"
)

# 处理发射率数据
eps_df = pd.DataFrame()
if uploaded_eps:
    eps_df, eps_status = load_and_clean_csv(uploaded_eps, "发射率数据", required_cols=2)
    if len(eps_df) > 0:
        eps_df["数值"] = eps_df["数值"].clip(0.0, 1.0)
        st.sidebar.success(f"{eps_status}")
    else:
        st.sidebar.error(eps_status)

# ================================= 主页面：数据加载状态展示 & 计算 =================================
# --- 修改点2：在主界面明确展示默认数据文件的加载状态 ---
st.markdown("### 📂 内置数据文件状态")
col1, col2 = st.columns(2)

# 太阳辐射数据状态
with col1:
    if uploaded_sun:
        sun_df, sun_status = load_and_clean_csv(uploaded_sun, "自定义太阳辐射", required_cols=2)
        st.markdown(f"**☀️ 太阳辐射数据**\n- 状态：{sun_status}\n- 来源：用户上传")
    else:
        sun_df, sun_status = load_and_clean_csv(DEFAULT_SUN_FILE, "默认太阳辐射", required_cols=2)
        st.markdown(f"**☀️ 太阳辐射数据**\n- 状态：{sun_status}\n- 来源：内置默认文件 (AM1.5)")

# 大气透过率数据状态
with col2:
    if uploaded_atm:
        atm_df, atm_status = load_and_clean_csv(uploaded_atm, "自定义大气透过率", required_cols=2)
        st.markdown(f"**🌫️ 大气透过率数据**\n- 状态：{atm_status}\n- 来源：用户上传")
    else:
        atm_df, atm_status = load_and_clean_csv(DEFAULT_ATM_FILE, "默认大气透过率", required_cols=2)
        st.markdown(f"**🌫️ 大气透过率数据**\n- 状态：{atm_status}\n- 来源：内置默认文件")

# 检查数据有效性
data_valid = True
if len(sun_df) == 0:
    st.error(f"❌ 太阳辐射数据无效：{sun_status}")
    data_valid = False
if len(atm_df) == 0:
    st.error(f"❌ 大气透过率数据无效：{atm_status}")
    data_valid = False
if len(eps_df) == 0:
    st.warning("⚠️ 请上传发射率CSV文件")
    data_valid = False

st.markdown("---")

# 计算条件汇总
st.markdown("### 📊 计算条件汇总")
with st.expander("点击查看详细参数", expanded=True):
    cond_data = {
        "参数名称": [
            "计算模式", "入射角θ", "计算波长范围",
            "环境温度Tamb", "冷却器温度Trad扫描范围", "对流换热系数q范围"
        ],
        "当前值": [
            day_night,
            f"{theta_deg:.1f}°（cosθ={cos_theta:.4f}）",
            f"{lambda_min:.2f}-{lambda_max:.2f} μm",
            f"{tamb_k:.2f} K ({k_to_c(tamb_k):.2f}°C)",
            f"{trad_min:.2f} - {trad_max:.2f} K (步长 {trad_step} K)",
            f"{q_list} W/(m²·K)"
        ]
    }
    st.dataframe(pd.DataFrame(cond_data), use_container_width=True)

# 计算按钮
calculate_btn = st.button("🚀 开始计算", disabled=not data_valid)

if calculate_btn:
    with st.spinner("正在计算中..."):
        # 数据预处理
        atm_df["数值"] = atm_df["数值"].clip(0.0, 1.0)
        
        # 生成波长网格
        lambda_grid = np.arange(lambda_min, lambda_max + 0.005, 0.01).round(2)
        lambda_grid = np.asarray(lambda_grid, dtype=np.float64).flatten()

        # 插值
        eps_interp = interpolate_curve(lambda_grid, eps_df["波长_μm"], eps_df["数值"], "发射率")
        tau_atm_interp = interpolate_curve(lambda_grid, atm_df["波长_μm"], atm_df["数值"], "大气透过率")
        
        sun_interp = np.zeros(len(lambda_grid), dtype=np.float64)
        if is_day:
            sun_interp = interpolate_curve(lambda_grid, sun_df["波长_μm"], sun_df["数值"], "太阳辐射")

        # 预构建插值函数
        eps_interp_func = interpolate.interp1d(lambda_grid, eps_interp, bounds_error=False, fill_value='extrapolate')
        tau_atm_interp_func = interpolate.interp1d(lambda_grid, tau_atm_interp, bounds_error=False, fill_value='extrapolate')

        # 批量计算
        result_list = []
        # 注意：tamb_list现在只有一个值
        for tamb in tamb_list:
            for trad in trad_list:
                for q in q_list:
                    # 1. P_rad
                    def p_rad_integrand(lmbda_μm):
                        lmbda_m = lmbda_μm * 1e-6
                        L_λ = planck_law(trad, lmbda_m)
                        eps = eps_interp_func(lmbda_μm)
                        return L_λ * eps * cos_theta * 1e-6

                    try:
                        p_rad_integral, _ = integrate.quad(p_rad_integrand, lambda_min, lambda_max, limit=200)
                        p_rad = p_rad_integral * 2 * np.pi
                    except:
                        p_rad = 0.0

                    # 2. P_atm
                    def p_atm_integrand(lmbda_μm):
                        lmbda_m = lmbda_μm * 1e-6
                        L_λ = planck_law(tamb, lmbda_m)
                        eps = eps_interp_func(lmbda_μm)
                        tau_atm = tau_atm_interp_func(lmbda_μm)
                        
                        if cos_theta < 1e-6:
                            eps_atm = 0.9
                        else:
                            tau_atm = max(tau_atm, 1e-8)
                            eps_atm = 1 - (tau_atm ** (1 / cos_theta))
                        return L_λ * eps * eps_atm * cos_theta * 1e-6

                    try:
                        p_atm_integral, _ = integrate.quad(p_atm_integrand, lambda_min, lambda_max, limit=200)
                        p_atm = p_atm_integral * 2 * np.pi
                    except:
                        p_atm = 0.0

                    # 3. P_sun
                    p_sun = 0.0
                    if is_day:
                        try:
                            p_sun = trapezoid(sun_interp * eps_interp, lambda_grid)
                        except:
                            p_sun = 0.0

                    # 4. P_cond_conv
                    p_cond_conv = q * (tamb - trad)

                    # 5. P_net
                    p_net = p_rad - p_atm - p_sun - p_cond_conv

                    result_list.append({
                        "昼夜模式": day_night,
                        "环境温度Tamb(K)": tamb,
                        "环境温度Tamb(°C)": round(k_to_c(tamb), 2),
                        "冷却器温度Trad(K)": trad,
                        "冷却器温度Trad(°C)": round(k_to_c(trad), 2),
                        "对流换热系数q(W/(m²·K))": q,
                        "材料辐射功率P_rad(W/m²)": round(p_rad, 2),
                        "大气逆辐射P_atm(W/m²)": round(p_atm, 2),
                        "太阳辐射吸收P_sun(W/m²)": round(p_sun, 2) if is_day else 0.0,
                        "非辐射损失P_cond+conv(W/m²)": round(p_cond_conv, 2),
                        "净制冷功率P_net(W/m²)": round(p_net, 2),
                        "制冷状态": "✅ 制冷" if p_net > 0 else "❌ 不制冷"
                    })

        # 结果展示
        result_df = pd.DataFrame(result_list)
        st.markdown("### 📈 计算结果")
        with st.expander("查看完整数据表格", expanded=True):
            st.dataframe(result_df, use_container_width=True, height=400)

        # --- 修改点1：图表乱码终极修复，增加英文备选方案 ---
        st.markdown("### 📊 不同q值净功率对比曲线")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        color_cycle = plt.get_cmap('tab10', len(q_list))
        
        for idx, q in enumerate(q_list):
            plot_data = result_df[result_df["对流换热系数q(W/(m²·K))"] == q].sort_values("冷却器温度Trad(K)")
            if len(plot_data) > 0:
                ax.plot(
                    plot_data["冷却器温度Trad(K)"], 
                    plot_data["净制冷功率P_net(W/m²)"],
                    'o-', color=color_cycle(idx), linewidth=2, markersize=6,
                    label=f"q={q} $W/(m^2 \cdot K)$"  # 使用LaTeX或纯英文避免乱码
                )
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1.5, label="Cooling Threshold")
        
        # 图表标签：如果中文字体不可用，自动降级为英文
        try:
            ax.set_xlabel("Radiative Cooler Temperature Trad (K)", fontsize=12)
            ax.set_ylabel("Net Cooling Power P_net (W/m²)", fontsize=12)
            ax.set_title(f"{day_night.split('（')[0]}: P_net vs Trad (Tamb={tamb_k:.2f}K)", fontsize=14, fontweight='bold')
        except:
            pass # 如果设置失败也不报错
            
        ax.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        # 结果下载
        st.markdown("### 📥 结果下载")
        excel_file = "辐射制冷功率计算结果.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            result_df.to_excel(writer, sheet_name=day_night.split('（')[0], index=False)
            pd.DataFrame(cond_data).to_excel(writer, sheet_name="计算条件", index=False)
        
        with open(excel_file, 'rb') as f:
            st.download_button(
                label="📥 下载计算结果Excel",
                data=f,
                file_name=f"辐射制冷功率计算结果_{day_night.split('（')[0]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
