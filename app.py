import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
import chardet  # 核心：自动检测编码的库
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 全局配置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 权威物理常数（CODATA 2018）
H_PLANCK = 6.62607015e-34    # 普朗克常数 (J·s)
C_LIGHT = 299792458          # 光速 (m/s)
K_BOLTZMANN = 1.380649e-23   # 玻尔兹曼常数 (J/K)

# -------------------------- 核心函数：自动检测编码并读取文件（根本解决编码问题） --------------------------
def auto_read_file(uploaded_file):
    """
    根本解法：自动检测文件编码，然后读取
    步骤：1. 读取文件二进制内容 2. 检测编码 3. 用检测到的编码读取文件
    """
    if not uploaded_file:
        return pd.DataFrame(), "❌ 未上传文件"
    
    # 步骤1：读取二进制内容，检测编码（chardet是行业标准的编码检测库）
    file_bytes = uploaded_file.read()
    encoding_result = chardet.detect(file_bytes)
    file_encoding = encoding_result['encoding']  # 自动识别的编码（如gb18030/utf-8）
    file_confidence = encoding_result['confidence']  # 识别置信度（0-1）
    
    # 步骤2：处理特殊情况（编码检测失败时用兜底编码）
    if file_encoding is None:
        file_encoding = 'gb18030'  # 兜底：适配Windows绝大多数情况
        file_confidence = 0.8
    
    # 步骤3：用检测到的编码读取文件
    try:
        # 重置文件指针（避免读取空内容）
        uploaded_file.seek(0)
        df = pd.read_csv(
            uploaded_file,
            encoding=file_encoding,
            sep=None,  # 自动检测分隔符（逗号/制表符都兼容）
            engine='python'
        )
        # 清理列名（去除空格/特殊字符，提升兼容性）
        df.columns = [col.strip() for col in df.columns]
        return df, f"✅ 读取成功（编码：{file_encoding}，置信度：{file_confidence:.2f}）"
    except Exception as e:
        return pd.DataFrame(), f"❌ 读取失败：{str(e)}（尝试编码：{file_encoding}）"

# -------------------------- 辅助计算函数 --------------------------
def planck_law(T_rad, lmbda_m):
    """普朗克定律：计算黑体辐射光谱辐照度（W/(m²·sr·m)）"""
    numerator = 2 * H_PLANCK * C_LIGHT**2 / (lmbda_m**5)
    denominator = np.exp(H_PLANCK * C_LIGHT / (lmbda_m * K_BOLTZMANN * T_rad)) - 1
    return numerator / denominator

def interpolate_curve(x_target, x_source, y_source, desc):
    """线性插值：将源曲线插值到目标波长网格"""
    if len(x_source) < 2 or len(y_source) < 2:
        st.error(f"{desc}数据不足，无法插值")
        return np.zeros_like(x_target)
    f = interpolate.interp1d(x_source, y_source, bounds_error=False, fill_value='extrapolate')
    return f(x_target)

# -------------------------- UI界面（简洁、聚焦核心功能） --------------------------
st.title("🌡️ 辐射制冷净功率自动计算系统")
st.markdown("---")

# ================================= 侧边栏：参数输入 =================================
st.sidebar.title("🔧 计算参数配置")

# 1. 基础固定参数
st.sidebar.markdown("### 1. 基础参数")
theta_deg = st.sidebar.number_input(
    "入射角 θ（度）", value=0.0, step=1.0, min_value=0.0, max_value=90.0,
    help="默认0°（正入射），范围0-90°"
)
theta_rad = np.radians(theta_deg)
lambda_min = st.sidebar.number_input(
    "波长下限（μm）", value=0.25, step=0.1, min_value=0.25, max_value=5.0,
    help="默认0.25μm（覆盖太阳辐射起始）"
)
lambda_max = st.sidebar.number_input(
    "波长上限（μm）", value=25.0, step=1.0, min_value=10.0, max_value=25.0,
    help="默认25μm（覆盖热辐射全范围）"
)

# 2. 必需数据文件上传（核心：自动检测编码）
st.sidebar.markdown("### 2. 数据文件上传（自动适配编码）")
uploaded_sun = st.sidebar.file_uploader(
    "📁 太阳辐射数据（AM1.5）", type=["csv", "txt"],
    help="格式：两列，列名含「波长」「辐射强度」（支持任意编码）"
)
uploaded_atm = st.sidebar.file_uploader(
    "📁 大气透过率数据（τatm）", type=["csv", "txt"],
    help="格式：两列，列名含「波长」「透过率」（支持任意编码）"
)
uploaded_eps = st.sidebar.file_uploader(
    "📁 冷却器发射率数据", type=["csv", "txt"],
    help="格式：两列，列名含「波长」「发射率」（支持任意编码）"
)

# 3. 批量计算参数
st.sidebar.markdown("### 3. 批量计算参数")
day_night = st.sidebar.radio("计算模式", ["白天（含太阳辐射）", "夜晚（无太阳辐射）"], index=0)
is_day = (day_night == "白天（含太阳辐射）")

# 环境温度Tamb
tamb_min = st.sidebar.number_input("Tamb最小值（K）", value=290.0, step=1.0, min_value=250.0)
tamb_max = st.sidebar.number_input("Tamb最大值（K）", value=300.0, step=1.0, min_value=tamb_min)
tamb_step = st.sidebar.number_input("Tamb步长（K）", value=5.0, step=1.0, min_value=0.5)
tamb_list = np.arange(tamb_min, tamb_max + tamb_step/2, tamb_step).round(2)

# 冷却器温度Trad
trad_min = st.sidebar.number_input("Trad最小值（K）", value=270.0, step=1.0, min_value=250.0)
trad_max = st.sidebar.number_input("Trad最大值（K）", value=285.0, step=1.0, min_value=trad_min)
trad_step = st.sidebar.number_input("Trad步长（K）", value=2.0, step=0.5, min_value=0.5)
trad_list = np.arange(trad_min, trad_max + trad_step/2, trad_step).round(2)

# 对流换热系数q
q_min = st.sidebar.number_input("q最小值（W/(m²·K)）", value=3.0, step=0.5, min_value=0.5)
q_max = st.sidebar.number_input("q最大值（W/(m²·K)）", value=8.0, step=0.5, min_value=q_min)
q_step = st.sidebar.number_input("q步长（W/(m²·K)）", value=1.0, step=0.5, min_value=0.5)
q_list = np.arange(q_min, q_max + q_step/2, q_step).round(2)

# ================================= 主界面：数据验证 + 计算 =================================
st.markdown("### 📋 数据文件验证（自动检测编码）")
col1, col2, col3 = st.columns(3)

# 验证太阳辐射文件
with col1:
    st.subheader("太阳辐射数据")
    sun_df, sun_msg = auto_read_file(uploaded_sun)
    st.write(sun_msg)
    if not sun_df.empty:
        st.dataframe(sun_df.head(3), use_container_width=True)

# 验证大气透过率文件
with col2:
    st.subheader("大气透过率数据")
    atm_df, atm_msg = auto_read_file(uploaded_atm)
    st.write(atm_msg)
    if not atm_df.empty:
        st.dataframe(atm_df.head(3), use_container_width=True)

# 验证发射率文件
with col3:
    st.subheader("发射率数据")
    eps_df, eps_msg = auto_read_file(uploaded_eps)
    st.write(eps_msg)
    if not eps_df.empty:
        st.dataframe(eps_df.head(3), use_container_width=True)

# 检查是否所有文件都加载成功
all_files_ready = not (sun_df.empty or atm_df.empty or eps_df.empty)
calculate_btn = st.button("🚀 开始计算", disabled=not all_files_ready)

if not all_files_ready:
    st.warning("请先上传并验证所有数据文件（太阳辐射、大气透过率、发射率）")

# ================================= 计算逻辑（原有逻辑不变） =================================
if calculate_btn:
    with st.spinner("正在计算，请稍候..."):
        # 1. 提取核心列（兼容不同列名，只要含关键词）
        # 太阳辐射：提取波长和强度列
        sun_wl_col = [col for col in sun_df.columns if "波长" in col][0]
        sun_val_col = [col for col in sun_df.columns if "强度" in col or "辐射" in col][0]
        sun_wl = sun_df[sun_wl_col].values
        sun_val = sun_df[sun_val_col].values

        # 大气透过率：提取波长和透过率列
        atm_wl_col = [col for col in atm_df.columns if "波长" in col][0]
        atm_val_col = [col for col in atm_df.columns if "透过率" in col or "τ" in col][0]
        atm_wl = atm_df[atm_wl_col].values
        atm_val = atm_df[atm_val_col].values

        # 发射率：提取波长和发射率列
        eps_wl_col = [col for col in eps_df.columns if "波长" in col][0]
        eps_val_col = [col for col in eps_df.columns if "发射率" in col or "ε" in col][0]
        eps_wl = eps_df[eps_wl_col].values
        eps_val = eps_df[eps_val_col].values

        # 2. 生成统一波长网格
        lambda_grid = np.arange(lambda_min, lambda_max + 0.005, 0.01).round(2)
        
        # 3. 插值所有曲线到统一网格
        sun_interp = interpolate_curve(lambda_grid, sun_wl, sun_val, "太阳辐射")
        atm_interp = interpolate_curve(lambda_grid, atm_wl, atm_val, "大气透过率")
        eps_interp = interpolate_curve(lambda_grid, eps_wl, eps_val, "发射率")
        sun_interp = sun_interp if is_day else np.zeros_like(lambda_grid)

        # 4. 批量计算所有参数组合
        result_list = []
        for tamb in tamb_list:
            for trad in trad_list:
                for q in q_list:
                    # 计算P_rad（材料自身辐射）
                    def p_rad_integrand(lmbda_μm):
                        lmbda_m = lmbda_μm * 1e-6
                        ibb = planck_law(trad, lmbda_m)
                        eps = interpolate_curve([lmbda_μm], lambda_grid, eps_interp, "发射率")[0]
                        return ibb * eps * np.cos(theta_rad) * 1e6

                    p_rad = integrate.quad(p_rad_integrand, lambda_min, lambda_max)[0] * 2 * np.pi

                    # 计算P_atm（大气逆辐射）
                    def p_atm_integrand(lmbda_μm):
                        lmbda_m = lmbda_μm * 1e-6
                        ibb = planck_law(tamb, lmbda_m)
                        eps = interpolate_curve([lmbda_μm], lambda_grid, eps_interp, "发射率")[0]
                        tau_atm = interpolate_curve([lmbda_μm], lambda_grid, atm_interp, "大气透过率")[0]
                        eps_atm = 1 - (tau_atm ** (1 / np.cos(theta_rad))) if np.cos(theta_rad) > 1e-6 else 0.9
                        return ibb * eps * eps_atm * np.cos(theta_rad) * 1e6

                    p_atm = integrate.quad(p_atm_integrand, lambda_min, lambda_max)[0] * 2 * np.pi

                    # 计算P_sun（太阳辐射）
                    p_sun = integrate.trapz(sun_interp * eps_interp, lambda_grid) if is_day else 0.0

                    # 计算P_cond_conv（非辐射损失）
                    p_cond_conv = q * (tamb - trad)

                    # 计算净功率P_net
                    p_net = p_rad - p_atm - p_sun - p_cond_conv

                    result_list.append({
                        "昼夜模式": day_night,
                        "Tamb（K）": tamb,
                        "Trad（K）": trad,
                        "q（W/(m²·K)）": q,
                        "P_rad（W/m²）": round(p_rad, 2),
                        "P_atm（W/m²）": round(p_atm, 2),
                        "P_sun（W/m²）": round(p_sun, 2),
                        "P_cond+conv（W/m²）": round(p_cond_conv, 2),
                        "P_net（W/m²）": round(p_net, 2),
                        "制冷状态": "✅ 制冷" if p_net > 0 else "❌ 不制冷"
                    })

        # 展示结果
        result_df = pd.DataFrame(result_list)
        st.markdown("### 📊 计算结果")
        st.dataframe(result_df, use_container_width=True)

        # 可视化
        st.markdown("### 📈 P_net随Trad变化曲线")
        tamb_mid = tamb_list[len(tamb_list)//2]
        q_mid = q_list[len(q_list)//2]
        plot_df = result_df[(result_df["Tamb（K）"] == tamb_mid) & (result_df["q（W/(m²·K)）"] == q_mid)]
        
        if len(plot_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(plot_df["Trad（K）"], plot_df["P_net（W/m²）"], 'o-', color='darkred', linewidth=2)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            ax.set_xlabel("Trad（K）")
            ax.set_ylabel("P_net（W/m²）")
            ax.set_title(f"{day_night}（Tamb={tamb_mid}K, q={q_mid}W/(m²·K)）")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        # 下载结果
        st.markdown("### 📥 结果下载")
        with pd.ExcelWriter("辐射制冷计算结果.xlsx", engine="openpyxl") as writer:
            result_df.to_excel(writer, sheet_name=day_night, index=False)
        
        with open("辐射制冷计算结果.xlsx", "rb") as f:
            st.download_button(
                label="下载Excel结果",
                data=f,
                file_name=f"辐射制冷计算结果_{day_night}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
