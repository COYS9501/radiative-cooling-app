import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chardet
from scipy import integrate
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="辐射制冷计算系统",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 全局样式设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 6)

# 加载默认数据（本地文件，原逻辑保留）
def load_default_data():
    """加载默认的太阳辐射和大气透过率数据"""
    try:
        # 太阳辐射默认数据（波长μm，辐照度W/m²·μm）
        sun_wavelengths = np.linspace(0.3, 2.5, 1000)
        sun_irradiance = np.zeros_like(sun_wavelengths)
        
        # 模拟太阳辐射光谱分布
        for i, λ in enumerate(sun_wavelengths):
            if 0.3 <= λ < 0.7:  # 可见光
                sun_irradiance[i] = 1000 * np.exp(-((λ - 0.5) / 0.2) **2)
            elif 0.7 <= λ < 2.5:  # 近红外
                sun_irradiance[i] = 500 * np.exp(-((λ - 1.0) / 0.5)** 2)
        
        sun_df_default = pd.DataFrame({
            "波长_μm": sun_wavelengths,
            "太阳辐照度_Wm2μm": sun_irradiance
        })
        
        # 大气透过率默认数据
        atm_wavelengths = np.linspace(0.3, 50, 1000)
        atm_transmittance = np.ones_like(atm_wavelengths)
        
        # 模拟大气窗口（8-13μm高透过率）
        for i, λ in enumerate(atm_wavelengths):
            if λ < 8 or λ > 13:
                atm_transmittance[i] = 0.2 + 0.8 * np.exp(-((λ - 10) / 5)** 2)
        
        atm_df_default = pd.DataFrame({
            "波长_μm": atm_wavelengths,
            "大气透过率": atm_transmittance
        })
        
        return sun_df_default, atm_df_default
    
    except Exception as e:
        st.error(f"默认数据加载失败：{str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# 初始化默认数据
sun_df_default, atm_df_default = load_default_data()

# 侧边栏参数设置
st.sidebar.title("🌡️ 辐射制冷计算参数")

# 1. 基本环境参数
st.sidebar.markdown("### 1. 环境参数")
T_amb = st.sidebar.number_input("环境温度 (K)", value=300.0, min_value=273.15, max_value=350.0, step=1.0)
T_sky = st.sidebar.number_input("天空温度 (K)", value=280.0, min_value=200.0, max_value=300.0, step=1.0)
latitude = st.sidebar.number_input("纬度 (°)", value=30.0, min_value=-90.0, max_value=90.0, step=1.0)
altitude = st.sidebar.number_input("海拔 (m)", value=0.0, min_value=0.0, max_value=5000.0, step=100.0)

# 2. 时间参数
st.sidebar.markdown("### 2. 时间参数")
month = st.sidebar.slider("月份", 1, 12, 7)
hour = st.sidebar.slider("小时", 0, 23, 12)

# 3. 自定义数据上传（太阳辐射和大气透过率）
st.sidebar.markdown("### 3. 自定义光谱数据（可选）")
uploaded_sun = st.sidebar.file_uploader("上传太阳辐射CSV（波长_μm, 太阳辐照度_Wm2μm）", type="csv")
uploaded_atm = st.sidebar.file_uploader("上传大气透过率CSV（波长_μm, 大气透过率）", type="csv")

# 4. 发射率数据上传（核心修改1：修复UploadedFile读取逻辑）
st.sidebar.markdown("### 4. 辐射冷却器发射率数据（必需）")
uploaded_eps = st.sidebar.file_uploader("上传发射率CSV（格式：波长_μm, 发射率ε）", type="csv", accept_multiple_files=False)
if uploaded_eps:
    try:
        # 直接读取UploadedFile二进制内容，无需open
        file_content = uploaded_eps.getvalue()
        result = chardet.detect(file_content)
        encoding = result['encoding'] or 'utf-8'  # 编码为空时默认utf-8
        # 用pd.read_csv直接读取二进制内容
        eps_df = pd.read_csv(pd.io.common.BytesIO(file_content), encoding=encoding)
        if not all(col in eps_df.columns for col in ["波长_μm", "发射率ε"]):
            st.sidebar.error("发射率CSV需包含列：波长_μm、发射率ε")
            eps_df = pd.DataFrame()
        else:
            st.sidebar.success(f"发射率数据加载成功（{len(eps_df)}行，波长{eps_df['波长_μm'].min():.2f}-{eps_df['波长_μm'].max():.2f}μm）")
    except Exception as e:
        st.sidebar.error(f"发射率数据加载失败：{str(e)}")
        eps_df = pd.DataFrame()
else:
    st.sidebar.warning("请上传发射率CSV文件（示例格式：波长_μm=0.3, 发射率ε=0.1；波长_μm=8, 发射率ε=0.95）")
    eps_df = pd.DataFrame()

# 主页面
st.title("🌍 辐射制冷性能计算系统")
st.markdown("### 计算说明")
st.markdown("""
该系统用于计算辐射冷却器的制冷功率，核心参数包括：
1. 环境/天空温度、地理位置（纬度/海拔）
2. 太阳辐射、大气透过率光谱数据
3. 冷却器发射率光谱数据
计算结果包含：净制冷功率、各波段贡献占比、光谱分布图
""")

# 计算按钮
if st.button("🚀 开始批量计算", type="primary"):
    # 数据校验
    if eps_df.empty:
        st.error("请先上传有效的发射率数据！")
        st.stop()
    
    if sun_df_default.empty or atm_df_default.empty:
        st.error("默认光谱数据加载失败，请检查程序！")
        st.stop()
    
    # 加载太阳辐射数据（核心修改2：修复自定义太阳辐射读取）
    if uploaded_sun:
        try:
            file_content = uploaded_sun.getvalue()
            result = chardet.detect(file_content)
            encoding = result['encoding'] or 'utf-8'
            sun_df = pd.read_csv(pd.io.common.BytesIO(file_content), encoding=encoding)
        except Exception as e:
            st.error(f"自定义太阳辐射文件加载失败：{str(e)}")
            st.stop()
    else:
        sun_df = sun_df_default if not sun_df_default.empty else st.stop()
    
    # 加载大气透过率数据（核心修改3：修复自定义大气透过率读取）
    if uploaded_atm:
        try:
            file_content = uploaded_atm.getvalue()
            result = chardet.detect(file_content)
            encoding = result['encoding'] or 'utf-8'
            atm_df = pd.read_csv(pd.io.common.BytesIO(file_content), encoding=encoding)
        except Exception as e:
            st.error(f"自定义大气透过率文件加载失败：{str(e)}")
            st.stop()
    else:
        atm_df = atm_df_default if not atm_df_default.empty else st.stop()
    
    # 数据预处理：插值到统一波长网格
    min_wl = max(eps_df['波长_μm'].min(), sun_df['波长_μm'].min(), atm_df['波长_μm'].min())
    max_wl = min(eps_df['波长_μm'].max(), sun_df['波长_μm'].max(), atm_df['波长_μm'].max())
    common_wavelengths = np.linspace(min_wl, max_wl, 1000)
    
    # 插值发射率
    eps_interp = np.interp(common_wavelengths, eps_df['波长_μm'], eps_df['发射率ε'])
    # 插值太阳辐射
    sun_interp = np.interp(common_wavelengths, sun_df['波长_μm'], sun_df['太阳辐照度_Wm2μm'])
    # 插值大气透过率
    atm_interp = np.interp(common_wavelengths, atm_df['波长_μm'], atm_df['大气透过率'])
    
    # 物理常数
    h = 6.626e-34  # 普朗克常数
    c = 3.0e8       # 光速
    k = 1.38e-23    # 玻尔兹曼常数
    σ = 5.67e-8     # 斯特藩-玻尔兹曼常数
    
    # 计算普朗克黑体辐射
    def planck(λ, T):
        """普朗克黑体辐射公式 (W/m²·μm)"""
        λ_m = λ * 1e-6  # 转换为米
        numerator = 2 * h * c**2 / (λ_m**5)
        denominator = np.exp(h * c / (λ_m * k * T)) - 1
        return numerator / denominator / 1e6  # 转换为W/m²·μm
    
    # 计算各波段辐射
    # 1. 冷却器发射的辐射
    emitter_radiation = eps_interp * planck(common_wavelengths, T_amb)
    emitter_total = integrate.simpson(emitter_radiation, common_wavelengths)
    
    # 2. 天空入射辐射
    sky_radiation = eps_interp * atm_interp * planck(common_wavelengths, T_sky)
    sky_total = integrate.simpson(sky_radiation, common_wavelengths)
    
    # 3. 太阳入射辐射
    solar_radiation = eps_interp * atm_interp * sun_interp
    solar_total = integrate.simpson(solar_radiation, common_wavelengths)
    
    # 净制冷功率
    net_power = emitter_total - sky_total - solar_total
    
    # 结果展示
    st.markdown("## 📊 计算结果")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("冷却器发射辐射 (W/m²)", f"{emitter_total:.2f}")
        st.metric("天空入射辐射 (W/m²)", f"{sky_total:.2f}")
    
    with col2:
        st.metric("太阳入射辐射 (W/m²)", f"{solar_total:.2f}")
        st.metric("净制冷功率 (W/m²)", f"{net_power:.2f}", 
                 delta=f"{net_power/emitter_total*100:.1f}% 能效",
                 delta_color="normal" if net_power > 0 else "inverse")
    
    with col3:
        st.metric("有效波长范围 (μm)", f"{min_wl:.2f} - {max_wl:.2f}")
        st.metric("光谱采样点数", len(common_wavelengths))
    
    # 光谱分布图
    st.markdown("## 📈 光谱分布分析")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：发射率和大气透过率
    ax1.plot(common_wavelengths, eps_interp, label='发射率 ε', color='red', linewidth=2)
    ax1.plot(common_wavelengths, atm_interp, label='大气透过率', color='blue', linewidth=2, linestyle='--')
    ax1.set_xlabel('波长 (μm)')
    ax1.set_ylabel('数值')
    ax1.set_title('发射率与大气透过率光谱')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：各辐射分量
    ax2.plot(common_wavelengths, emitter_radiation, label='冷却器发射', color='green')
    ax2.plot(common_wavelengths, sky_radiation, label='天空入射', color='orange')
    ax2.plot(common_wavelengths, solar_radiation, label='太阳入射', color='purple')
    ax2.set_xlabel('波长 (μm)')
    ax2.set_ylabel('辐射强度 (W/m²·μm)')
    ax2.set_title('各波段辐射强度分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # 数据导出
    st.markdown("## 📥 数据导出")
    result_df = pd.DataFrame({
        "波长_μm": common_wavelengths,
        "发射率ε": eps_interp,
        "大气透过率": atm_interp,
        "冷却器发射辐射": emitter_radiation,
        "天空入射辐射": sky_radiation,
        "太阳入射辐射": solar_radiation
    })
    
    csv_data = result_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="下载完整光谱数据 (CSV)",
        data=csv_data,
        file_name=f"辐射制冷计算结果_纬度{latitude}_温度{T_amb}K.csv",
        mime="text/csv"
    )
    
    # 计算总结
    st.markdown("## 📋 计算总结")
    st.markdown(f"""
    - 计算条件：纬度 {latitude}°，海拔 {altitude}m，环境温度 {T_amb}K，天空温度 {T_sky}K
    - 时间：{month}月 {hour}时
    - 净制冷功率：{net_power:.2f} W/m²
    - 主要贡献：
      - 冷却器发射：{emitter_total:.2f} W/m² ({emitter_total/(emitter_total+sky_total+solar_total)*100:.1f}%)
      - 天空入射损失：{sky_total:.2f} W/m² ({sky_total/(emitter_total+sky_total+solar_total)*100:.1f}%)
      - 太阳入射损失：{solar_total:.2f} W/m² ({solar_total/(emitter_total+sky_total+solar_total)*100:.1f}%)
    """)

# 页脚
st.markdown("---")
st.markdown("© 2025 辐射制冷计算系统 | 技术支持：Streamlit + Python")
