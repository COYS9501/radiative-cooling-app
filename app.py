import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
import warnings
import os
warnings.filterwarnings('ignore')
from io import BytesIO

# -------------------------- 全局配置 & 中文乱码终极修复 --------------------------
# 全环境兼容的中文字体配置，彻底解决乱码
plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei',  # Streamlit Cloud/Linux环境优先
    'SimHei',                # Windows环境
    'PingFang SC',           # Mac环境
    'DejaVu Sans'            # 兜底通用字体
]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.rcParams['figure.dpi'] = 100  # 图表清晰度优化

# 物理常数（固定不变，行业标准值）
H_PLANCK = 6.62607015e-34  # J·s
C_LIGHT = 299792458        # m/s
K_BOLTZMANN = 1.380649e-23 # J/K
SIGMA_STEFAN = 5.670374419e-8 # W/(m²·K^4)

# scipy版本全兼容（解决trapz/trapezoid问题）
try:
    from scipy.integrate import trapezoid
except ImportError:
    from scipy.integrate import trapz as trapezoid

# 默认数据文件路径
DEFAULT_SUN_FILE = 'AM15太阳辐射_处理后.csv'
DEFAULT_ATM_FILE = '大气透过率_处理后.csv'

# -------------------------- 核心函数（全异常兜底） --------------------------
import chardet

def load_and_clean_csv(file_path_or_buffer, desc, required_cols=2):
    """
    通用CSV加载&清洗函数（核心兜底，解决所有数据脏问题）
    输入：文件路径/文件buffer，描述，要求列数
    输出：清洗后的DataFrame，状态信息
    """
    try:
        # 读取文件&检测编码
        if isinstance(file_path_or_buffer, str):
            # 本地文件路径
            if not os.path.exists(file_path_or_buffer):
                return pd.DataFrame(), f"❌ {desc}文件不存在：{file_path_or_buffer}"
            with open(file_path_or_buffer, 'rb') as f:
                file_content = f.read()
        else:
            # 上传的文件buffer
            file_content = file_path_or_buffer.getvalue()
        
        # 自动检测编码
        result = chardet.detect(file_content)
        encoding = result['encoding'] or 'utf-8'
        # 读取CSV
        df = pd.read_csv(BytesIO(file_content), encoding=encoding)
        
        # 列数校验
        if len(df.columns) != required_cols:
            return pd.DataFrame(), f"❌ {desc}必须为{required_cols}列数据，当前列数：{len(df.columns)}"
        
        # 强制重命名列（第一列=波长，第二列=数值）
        df.columns = ["波长_μm", "数值"]
        # 强制转换为数值，非数值转为NaN
        df["波长_μm"] = pd.to_numeric(df["波长_μm"], errors='coerce')
        df["数值"] = pd.to_numeric(df["数值"], errors='coerce')
        # 过滤空值
        df_clean = df.dropna().reset_index(drop=True)
        
        # 有效数据校验
        if len(df_clean) < 2:
            return pd.DataFrame(), f"❌ {desc}清洗后有效数据不足2行，无法插值"
        
        # 数值范围校验
        if df_clean["波长_μm"].min() < 0 or df_clean["数值"].min() < 0:
            return pd.DataFrame(), f"❌ {desc}包含负数，数据无效"
        
        return df_clean, f"✅ {desc}加载成功，有效数据{len(df_clean)}行"
    
    except Exception as e:
        return pd.DataFrame(), f"❌ {desc}加载失败：{str(e)}"

def planck_law(T_rad, lmbda_m):
    """
    普朗克黑体辐射定律（全异常保护）
    输入：T_rad-温度(K)，lmbda_m-波长(m)
    输出：光谱辐射亮度 W/(m²·sr·m)
    """
    lmbda_m = np.maximum(lmbda_m, 1e-20)  # 避免分母为0
    exponent = H_PLANCK * C_LIGHT / (lmbda_m * K_BOLTZMANN * np.maximum(T_rad, 1e-10))
    exponent = np.minimum(exponent, 700)  # 避免指数溢出
    numerator = 2 * H_PLANCK * C_LIGHT**2 / (lmbda_m**5)
    denominator = np.exp(exponent) - 1
    denominator = np.maximum(denominator, 1e-10)  # 避免分母为0
    return numerator / denominator

def interpolate_curve(x_target, x_source, y_source, desc):
    """
    终极鲁棒插值函数（100%不报错）
    输入：目标网格、源x、源y、描述
    输出：插值后的一维numpy数组，异常时返回全零数组
    """
    try:
        # 强制转换为一维数值数组，所有异常都捕获
        x_target = np.asarray(x_target, dtype=np.float64).flatten()
        x_source = np.asarray(x_source, dtype=np.float64).flatten()
        y_source = np.asarray(y_source, dtype=np.float64).flatten()
    except Exception as e:
        st.error(f"{desc}数据转换失败：{str(e)}，返回全零数组")
        return np.zeros(len(x_target), dtype=np.float64)
    
    # 基础数据量校验
    if len(x_source) < 2 or len(y_source) < 2 or len(x_target) < 1:
        st.error(f"{desc}有效数据不足2个点，返回全零数组")
        return np.zeros(len(x_target), dtype=np.float64)
    
    # 过滤NaN、无穷值
    valid_mask = ~(np.isnan(x_source) | np.isnan(y_source) | np.isinf(x_source) | np.isinf(y_source))
    x_valid = x_source[valid_mask]
    y_valid = y_source[valid_mask]
    
    if len(x_valid) < 2:
        st.error(f"{desc}清洗后有效数据不足2个点，返回全零数组")
        return np.zeros(len(x_target), dtype=np.float64)
    
    # 执行插值，全异常捕获
    try:
        f = interpolate.interp1d(x_valid, y_valid, bounds_error=False, fill_value='extrapolate')
        result = f(x_target).flatten()
        return np.asarray(result, dtype=np.float64)
    except Exception as e:
        st.error(f"{desc}插值失败：{str(e)}，返回全零数组")
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
st.sidebar.caption(f"cosθ = {cos_theta:.4f}")

# 波长范围（固定0.25-25μm，不建议修改）
lambda_min = st.sidebar.number_input(
    "波长下限（μm）", value=0.25, step=0.1, min_value=0.2, max_value=5.0,
    help="建议固定0.25μm，覆盖太阳辐射+大气窗口"
)
lambda_max = st.sidebar.number_input(
    "波长上限（μm）", value=25.0, step=1.0, min_value=10.0, max_value=30.0,
    help="建议固定25μm，覆盖完整红外大气窗口"
)
st.sidebar.caption(f"计算波长范围：{lambda_min:.2f}-{lambda_max:.2f} μm")

# 数据文件配置
st.sidebar.markdown("### 2. 数据文件（支持自定义上传）")

# 太阳辐射数据
st.sidebar.subheader("太阳辐射数据（AM1.5）")
uploaded_sun = st.sidebar.file_uploader(
    "上传自定义太阳辐射CSV（两列：波长μm、辐照度W/(m²·μm)）",
    type="csv", key="sun_upload"
)

# 大气透过率数据
st.sidebar.subheader("大气透过率数据")
uploaded_atm = st.sidebar.file_uploader(
    "上传自定义大气透过率CSV（两列：波长μm、透过率0-1）",
    type="csv", key="atm_upload"
)

# 计算模式与批量参数
st.sidebar.markdown("### 3. 计算模式与批量参数")
day_night = st.sidebar.radio("计算模式", ["白天（含太阳辐射）", "夜晚（无太阳辐射）"], index=0)
is_day = (day_night == "白天（含太阳辐射）")

# 环境温度Tamb
st.sidebar.subheader("环境温度 Tamb（K）")
tamb_min = st.sidebar.number_input("Tamb最小值", value=280.0 if not is_day else 290.0, step=1.0, min_value=250.0, max_value=330.0)
tamb_max = st.sidebar.number_input("Tamb最大值", value=290.0 if not is_day else 300.0, step=1.0, min_value=tamb_min, max_value=330.0)
tamb_step = st.sidebar.number_input("Tamb步长", value=5.0, step=1.0, min_value=0.5, max_value=10.0)
tamb_list = np.arange(tamb_min, tamb_max + tamb_step/2, tamb_step).round(2)
st.sidebar.caption(f"Tamb计算列表：{tamb_list} K")

# 冷却器温度Trad
st.sidebar.subheader("冷却器温度 Trad（K）")
trad_min = st.sidebar.number_input("Trad最小值", value=270.0, step=1.0, min_value=250.0, max_value=tamb_max)
trad_max = st.sidebar.number_input("Trad最大值", value=285.0, step=1.0, min_value=trad_min, max_value=tamb_max)
trad_step = st.sidebar.number_input("Trad步长", value=2.0, step=0.5, min_value=0.5, max_value=5.0)
trad_list = np.arange(trad_min, trad_max + trad_step/2, trad_step).round(2)
st.sidebar.caption(f"Trad计算列表：{trad_list} K")

# 对流换热系数q
st.sidebar.subheader("对流换热系数 q（W/(m²·K)）")
q_min = st.sidebar.number_input("q最小值", value=3.0, step=0.5, min_value=0.0, max_value=20.0)
q_max = st.sidebar.number_input("q最大值", value=8.0, step=0.5, min_value=q_min, max_value=20.0)
q_step = st.sidebar.number_input("q步长", value=1.0, step=0.5, min_value=0.5, max_value=5.0)
q_list = np.arange(q_min, q_max + q_step/2, q_step).round(2)
st.sidebar.caption(f"q计算列表：{q_list} W/(m²·K)")

# 发射率数据（必需）
st.sidebar.markdown("### 4. 冷却器发射率数据（必需）")
uploaded_eps = st.sidebar.file_uploader(
    "上传发射率CSV（两列：波长μm、发射率0-1）",
    type="csv", key="eps_upload"
)

# 处理发射率数据
eps_df = pd.DataFrame()
eps_status = ""
if uploaded_eps:
    eps_df, eps_status = load_and_clean_csv(uploaded_eps, "发射率数据", required_cols=2)
    if len(eps_df) > 0:
        # 发射率限制在0-1
        eps_df["数值"] = eps_df["数值"].clip(0.0, 1.0)
        st.sidebar.success(
            f"{eps_status}\n"
            f"波长范围：{eps_df['波长_μm'].min():.2f}-{eps_df['波长_μm'].max():.2f}μm\n"
            f"发射率范围：{eps_df['数值'].min():.3f}-{eps_df['数值'].max():.3f}"
        )
    else:
        st.sidebar.error(eps_status)
else:
    st.sidebar.warning(
        "请上传发射率CSV，示例格式：\n"
        "Wavelength,Emissivity\n"
        "0.3,0.1\n"
        "8.0,0.95\n"
        "15.0,0.98"
    )

# ================================= 主页面计算逻辑 =================================
st.markdown("### 📊 计算条件汇总")
with st.expander("点击查看当前计算参数", expanded=True):
    cond_data = {
        "参数名称": [
            "入射角θ", "计算波长范围", "计算模式",
            "太阳辐射文件", "大气透过率文件", "发射率文件",
            "Tamb计算列表", "Trad计算列表", "q计算列表"
        ],
        "当前值": [
            f"{theta_deg:.1f}°（cosθ={cos_theta:.4f}）",
            f"{lambda_min:.2f}-{lambda_max:.2f} μm",
            day_night,
            uploaded_sun.name if uploaded_sun else "默认文件",
            uploaded_atm.name if uploaded_atm else "默认文件",
            uploaded_eps.name if uploaded_eps else "未上传",
            f"{tamb_list}（共{len(tamb_list)}个点）",
            f"{trad_list}（共{len(trad_list)}个点）",
            f"{q_list}（共{len(q_list)}个点）"
        ]
    }
    st.dataframe(pd.DataFrame(cond_data), use_container_width=True)

# 计算权限校验
can_calculate = (len(eps_df) > 0) and (len(tamb_list) > 0) and (len(trad_list) > 0) and (len(q_list) > 0)
if not can_calculate:
    st.warning("请完成必需输入：1. 上传有效的发射率CSV；2. 确认Tamb/Trad/q的范围和步长")

# 计算按钮
calculate_btn = st.button("🚀 开始批量计算辐射制冷净功率", disabled=not can_calculate)

if calculate_btn:
    with st.spinner("正在计算中...（批量计算约10-20秒）"):
        # -------------------------- 1. 加载&清洗所有数据（核心兜底） --------------------------
        # 加载太阳辐射数据
        if uploaded_sun:
            sun_df, sun_status = load_and_clean_csv(uploaded_sun, "太阳辐射数据", required_cols=2)
        else:
            sun_df, sun_status = load_and_clean_csv(DEFAULT_SUN_FILE, "默认太阳辐射数据", required_cols=2)
        
        if len(sun_df) == 0:
            st.error(f"太阳辐射数据加载失败：{sun_status}")
            st.stop()
        st.success(sun_status)

        # 加载大气透过率数据
        if uploaded_atm:
            atm_df, atm_status = load_and_clean_csv(uploaded_atm, "大气透过率数据", required_cols=2)
        else:
            atm_df, atm_status = load_and_clean_csv(DEFAULT_ATM_FILE, "默认大气透过率数据", required_cols=2)
        
        if len(atm_df) == 0:
            st.error(f"大气透过率数据加载失败：{atm_status}")
            st.stop()
        st.success(atm_status)

        # 大气透过率限制在0-1
        atm_df["数值"] = atm_df["数值"].clip(0.0, 1.0)

        # -------------------------- 2. 生成波长网格 & 插值 --------------------------
        # 生成统一波长网格
        lambda_grid = np.arange(lambda_min, lambda_max + 0.005, 0.01).round(2)
        lambda_grid = np.asarray(lambda_grid, dtype=np.float64).flatten()
        st.success(f"✅ 生成统一波长网格：{len(lambda_grid)}个点（{lambda_min:.2f}-{lambda_max:.2f}μm）")

        # 所有曲线插值到统一网格（100%兜底）
        eps_interp = interpolate_curve(lambda_grid, eps_df["波长_μm"], eps_df["数值"], "发射率")
        tau_atm_interp = interpolate_curve(lambda_grid, atm_df["波长_μm"], atm_df["数值"], "大气透过率")
        
        # 太阳辐射插值
        sun_interp = np.zeros(len(lambda_grid), dtype=np.float64)
        if is_day:
            sun_interp = interpolate_curve(lambda_grid, sun_df["波长_μm"], sun_df["数值"], "太阳辐射")

        # 预构建插值函数（用于积分）
        eps_interp_func = interpolate.interp1d(lambda_grid, eps_interp, bounds_error=False, fill_value='extrapolate')
        tau_atm_interp_func = interpolate.interp1d(lambda_grid, tau_atm_interp, bounds_error=False, fill_value='extrapolate')

        # -------------------------- 3. 批量计算净功率 --------------------------
        result_list = []
        for tamb in tamb_list:
            for trad in trad_list:
                for q in q_list:
                    # 1. 计算材料辐射功率 P_rad
                    def p_rad_integrand(lmbda_μm):
                        lmbda_m = lmbda_μm * 1e-6  # μm → m
                        L_λ = planck_law(trad, lmbda_m)
                        eps = eps_interp_func(lmbda_μm)
                        return L_λ * eps * cos_theta * 1e-6  # 单位转换系数

                    try:
                        p_rad_integral, _ = integrate.quad(p_rad_integrand, lambda_min, lambda_max, limit=200)
                        p_rad = p_rad_integral * 2 * np.pi  # 半球立体角
                    except Exception as e:
                        st.warning(f"P_rad计算异常（Tamb={tamb}, Trad={trad}）：{str(e)}，按0计算")
                        p_rad = 0.0

                    # 2. 计算大气逆辐射 P_atm
                    def p_atm_integrand(lmbda_μm):
                        lmbda_m = lmbda_μm * 1e-6
                        L_λ = planck_law(tamb, lmbda_m)
                        eps = eps_interp_func(lmbda_μm)
                        tau_atm = tau_atm_interp_func(lmbda_μm)
                        
                        # 大气发射率计算，避免除零
                        if cos_theta < 1e-6:
                            eps_atm = 0.9
                        else:
                            tau_atm = max(tau_atm, 1e-8)
                            eps_atm = 1 - (tau_atm ** (1 / cos_theta))
                        return L_λ * eps * eps_atm * cos_theta * 1e-6

                    try:
                        p_atm_integral, _ = integrate.quad(p_atm_integrand, lambda_min, lambda_max, limit=200)
                        p_atm = p_atm_integral * 2 * np.pi
                    except Exception as e:
                        st.warning(f"P_atm计算异常（Tamb={tamb}, Trad={trad}）：{str(e)}，按0计算")
                        p_atm = 0.0

                    # 3. 计算太阳辐射吸收 P_sun
                    p_sun = 0.0
                    if is_day:
                        try:
                            p_sun = trapezoid(sun_interp * eps_interp, lambda_grid)
                        except Exception as e:
                            st.warning(f"P_sun计算异常：{str(e)}，按0计算")
                            p_sun = 0.0

                    # 4. 计算非辐射损失 P_cond_conv
                    p_cond_conv = q * (tamb - trad)

                    # 5. 计算净制冷功率 P_net
                    p_net = p_rad - p_atm - p_sun - p_cond_conv

                    # 保存结果
                    result_list.append({
                        "昼夜模式": day_night,
                        "环境温度Tamb(K)": tamb,
                        "冷却器温度Trad(K)": trad,
                        "对流换热系数q(W/(m²·K))": q,
                        "材料辐射功率P_rad(W/m²)": round(p_rad, 2),
                        "大气逆辐射P_atm(W/m²)": round(p_atm, 2),
                        "太阳辐射吸收P_sun(W/m²)": round(p_sun, 2) if is_day else 0.0,
                        "非辐射损失P_cond+conv(W/m²)": round(p_cond_conv, 2),
                        "净制冷功率P_net(W/m²)": round(p_net, 2),
                        "制冷状态": "✅ 制冷" if p_net > 0 else "❌ 不制冷"
                    })

        # -------------------------- 4. 结果展示与下载 --------------------------
        result_df = pd.DataFrame(result_list)
        st.markdown("### 📈 批量计算结果（共{}组数据）".format(len(result_df)))

        # 完整结果表格
        with st.expander("查看完整结果表格", expanded=True):
            st.dataframe(result_df, use_container_width=True, height=500)

        # -------------------------- 核心优化：多q值同图绘制 --------------------------
        st.markdown("### 📊 不同对流换热系数q的净功率对比曲线（固定Tamb）")
        # 固定中间值Tamb，保证对比变量唯一
        tamb_mid = tamb_list[len(tamb_list)//2]
        # 颜色循环，区分不同q值
        color_cycle = plt.get_cmap('tab10', len(q_list))
        
        # 创建画布
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 遍历所有q值，绘制曲线
        for idx, q in enumerate(q_list):
            # 筛选当前Tamb和q的数据
            plot_data = result_df[
                (result_df["环境温度Tamb(K)"] == tamb_mid) & 
                (result_df["对流换热系数q(W/(m²·K))"] == q)
            ].sort_values("冷却器温度Trad(K)")
            
            if len(plot_data) > 0:
                ax.plot(
                    plot_data["冷却器温度Trad(K)"], 
                    plot_data["净制冷功率P_net(W/m²)"],
                    'o-', 
                    color=color_cycle(idx),
                    linewidth=2, 
                    markersize=6,
                    label=f"q={q} W/(m²·K)"
                )
        
        # 绘制制冷临界点
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1.5, label="制冷临界点（P_net=0）")
        
        # 图表美化
        ax.set_xlabel("辐射冷却器温度 Trad (K)", fontsize=13)
        ax.set_ylabel("净制冷功率 P_net (W/m²)", fontsize=13)
        ax.set_title(f"{day_night} 不同q值净功率对比（固定Tamb={tamb_mid}K）", fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, bbox_to_anchor=(1.02, 1), loc='upper left')  # 图例放图外，避免遮挡
        ax.grid(alpha=0.3, linestyle='-')
        plt.tight_layout()  # 自动调整布局，避免标签被截断
        
        # 展示图表
        st.pyplot(fig)

        # 结果下载
        st.markdown("### 📥 结果下载")
        excel_file = "辐射制冷功率计算结果.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            result_df.to_excel(writer, sheet_name=day_night, index=False)
            pd.DataFrame(cond_data).to_excel(writer, sheet_name="计算条件", index=False)
        
        with open(excel_file, 'rb') as f:
            st.download_button(
                label="📥 下载计算结果Excel",
                data=f,
                file_name=f"辐射制冷功率计算结果_{day_night.replace('（', '_').replace('）', '')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # 结果合理性校验
        st.markdown("### 📊 结果合理性校验")
        total_cooling = len(result_df[result_df["制冷状态"] == "✅ 制冷"])
        max_pnet = result_df["净制冷功率P_net(W/m²)"].max()
        st.info(f"""
        - 总计算组数：{len(result_df)} 组
        - 实现制冷的组数：{total_cooling} 组（占比 {total_cooling/len(result_df)*100:.1f}%）
        - 最大净制冷功率：{max_pnet:.2f} W/m²
        - 300K黑体极限辐射功率：{round(SIGMA_STEFAN * 300**4, 2)} W/m²（结果应小于此值）
        """)
