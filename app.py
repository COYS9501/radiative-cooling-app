import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 全局配置 --------------------------
# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 权威物理常数（CODATA 2018，不可修改）
H_PLANCK = 6.62607015e-34    # 普朗克常数 (J·s)
C_LIGHT = 299792458          # 光速 (m/s)
K_BOLTZMANN = 1.380649e-23   # 玻尔兹曼常数 (J/K)

# 内置文件路径（默认值，用户可替换）
DEFAULT_SUN_FILE = 'AM15太阳辐射_处理后.csv'
DEFAULT_ATM_FILE = '大气透过率_处理后.csv'

# -------------------------- 基础函数（计算逻辑封装，UI不显示） --------------------------
def load_default_data(file_path, desc):
    """加载默认数据，返回DataFrame和基本信息"""
    try:
        df = pd.read_csv(file_path)
        return df, f"✅ 加载成功：{desc}（{len(df)}行数据，波长{df['波长_μm'].min():.2f}-{df['波长_μm'].max():.2f}μm）"
    except Exception as e:
        return pd.DataFrame(), f"❌ 加载失败：{str(e)}"

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
    # 超出源范围的部分用边缘值填充
    f = interpolate.interp1d(x_source, y_source, bounds_error=False, fill_value='extrapolate')
    return f(x_target)

# -------------------------- UI页面开发 --------------------------
st.title("🌞 辐射制冷净功率自动计算系统")
st.markdown("---")

# ================================= 输入区（侧边栏） =================================
st.sidebar.title("🔧 计算参数输入（默认值可修改）")
st.sidebar.markdown("### 1. 基础固定参数（默认值适配常规场景）")

# 1.1 入射角（默认0°，正入射）
theta_deg = st.sidebar.number_input(
    "入射角 θ（度）",
    value=0.0,
    step=1.0,
    min_value=0.0,
    max_value=90.0,
    help="默认0°（正入射），范围0-90°，角度越大余弦权重越小"
)
theta_rad = np.radians(theta_deg)
st.sidebar.caption(f"当前θ（弧度）：{theta_rad:.4f} rad | cosθ：{np.cos(theta_rad):.4f}")

# 1.2 计算波长范围（默认0.3-20μm）
# 波长下限默认0.25μm
lambda_min = st.sidebar.number_input(
    "波长下限（μm）",
    value=0.25,
    step=0.1,
    min_value=0.25,
    max_value=5.0,
    help="默认0.25μm（覆盖太阳辐射起始）"
)
# 波长上限默认25μm
lambda_max = st.sidebar.number_input(
    "波长上限（μm）",
    value=25.0,
    step=1.0,
    min_value=10.0,
    max_value=25.0,
    help="默认25μm（覆盖热辐射全范围）"
)
st.sidebar.caption(f"最终计算波长范围：{lambda_min:.1f}-{lambda_max:.1f} μm")

# 1.3 内置文件显示与替换（太阳辐射+大气透过率）
st.sidebar.markdown("### 2. 内置数据文件（支持自定义替换）")

# 太阳辐射文件
st.sidebar.subheader("太阳辐射数据（AM1.5）")
sun_df_default, sun_msg_default = load_default_data(DEFAULT_SUN_FILE, "AM1.5太阳辐射")
st.sidebar.caption(f"默认文件：{DEFAULT_SUN_FILE.split('/')[-1]} | {sun_msg_default}")
uploaded_sun = st.sidebar.file_uploader("上传自定义太阳辐射CSV（波长_μm, 太阳辐射强度_Wm-2μm-1）", type="csv")

# 大气透过率文件
st.sidebar.subheader("大气透过率数据（τatm）")
atm_df_default, atm_msg_default = load_default_data(DEFAULT_ATM_FILE, "大气透过率")
st.sidebar.caption(f"默认文件：{DEFAULT_ATM_FILE.split('/')[-1]} | {atm_msg_default}")
uploaded_atm = st.sidebar.file_uploader("上传自定义大气透过率CSV（波长_μm, 大气透过率_τatm）", type="csv")

# 1.4 昼夜模式与批量参数
st.sidebar.markdown("### 3. 昼夜模式与批量计算参数")
day_night = st.sidebar.radio("计算模式", ["白天（含太阳辐射）", "夜晚（无太阳辐射）"], index=0)
is_day = (day_night == "白天（含太阳辐射）")

# 环境温度Tamb（范围+步长）
st.sidebar.subheader("环境温度 Tamb（K）")
tamb_min = st.sidebar.number_input("Tamb最小值", value=280.0 if not is_day else 290.0, step=1.0, min_value=250.0, max_value=330.0)
tamb_max = st.sidebar.number_input("Tamb最大值", value=290.0 if not is_day else 300.0, step=1.0, min_value=tamb_min, max_value=330.0)
tamb_step = st.sidebar.number_input("Tamb步长", value=5.0, step=1.0, min_value=0.5, max_value=10.0)
tamb_list = np.arange(tamb_min, tamb_max + tamb_step/2, tamb_step).round(2)
st.sidebar.caption(f"Tamb计算列表：{tamb_list} K")

# 冷却器温度Trad（范围+步长）
st.sidebar.subheader("辐射冷却器温度 Trad（K）")
trad_min = st.sidebar.number_input("Trad最小值", value=270.0, step=1.0, min_value=250.0, max_value=tamb_max)
trad_max = st.sidebar.number_input("Trad最大值", value=285.0, step=1.0, min_value=trad_min, max_value=tamb_max)
trad_step = st.sidebar.number_input("Trad步长", value=2.0, step=0.5, min_value=0.5, max_value=5.0)
trad_list = np.arange(trad_min, trad_max + trad_step/2, trad_step).round(2)
st.sidebar.caption(f"Trad计算列表：{trad_list} K")

# 对流换热系数q（范围+步长）
st.sidebar.subheader("对流换热系数 q（W/(m²·K)）")
q_min = st.sidebar.number_input("q最小值", value=3.0, step=0.5, min_value=0.5, max_value=20.0)
q_max = st.sidebar.number_input("q最大值", value=8.0, step=0.5, min_value=q_min, max_value=20.0)
q_step = st.sidebar.number_input("q步长", value=1.0, step=0.5, min_value=0.5, max_value=5.0)
q_list = np.arange(q_min, q_max + q_step/2, q_step).round(2)
st.sidebar.caption(f"q计算列表：{q_list} W/(m²·K)")

# 1.5 发射率数据上传
st.sidebar.markdown("### 4. 辐射冷却器发射率数据（必需）")
uploaded_eps = st.sidebar.file_uploader("上传发射率CSV（格式：波长_μm, 发射率ε）", type="csv", accept_multiple_files=False)
if uploaded_eps:
    try:
        eps_df = pd.read_csv(uploaded_eps)
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

# 1.6 动态调整波长范围
if uploaded_eps:
    try:
        eps_df = pd.read_csv(uploaded_eps)
        if not all(col in eps_df.columns for col in ["波长_μm", "发射率ε"]):
            st.sidebar.error("发射率CSV需包含列：波长_μm、发射率ε")
            eps_df = pd.DataFrame()
        else:
            st.sidebar.success(f"发射率数据加载成功（{len(eps_df)}行，波长{eps_df['波长_μm'].min():.2f}-{eps_df['波长_μm'].max():.2f}μm）")
            # 动态调整波长范围（取默认范围和发射率范围的交集）
            eps_lambda_min = eps_df["波长_μm"].min()
            eps_lambda_max = eps_df["波长_μm"].max()
            final_lambda_min = max(lambda_min, eps_lambda_min)
            final_lambda_max = min(lambda_max, eps_lambda_max)
            st.sidebar.info(f"自动调整波长范围：{final_lambda_min:.2f}-{final_lambda_max:.2f}μm（匹配发射率数据）")
    except Exception as e:
        st.sidebar.error(f"发射率数据加载失败：{str(e)}")
        eps_df = pd.DataFrame()

# ================================= 输出区（主页面） =================================
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
            f"{lambda_min:.1f}-{lambda_max:.1f} μm",
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

# 计算按钮（所有输入验证通过后才允许点击）
can_calculate = (len(eps_df) > 0) and (len(tamb_list) > 0) and (len(trad_list) > 0) and (len(q_list) > 0)
if not can_calculate:
    st.warning("请完成必需输入：1. 上传发射率CSV；2. 确认Tamb/Trad/q的范围和步长（确保列表非空）")

calculate_btn = st.button("🚀 开始批量计算辐射制冷净功率", disabled=not can_calculate)

# 计算结果展示
if calculate_btn:
    with st.spinner("正在计算...（批量计算可能需要10-30秒，请耐心等待）"):
        # 1. 加载最终使用的数据（优先用户上传，其次默认）
        # 太阳辐射
        if uploaded_sun:
           sun_df = pd.read_csv(uploaded_sun)
        else:
            sun_df = sun_df_default if not sun_df_default.empty else st.stop()
        # 大气透过率
        if uploaded_atm:
            atm_df = pd.read_csv(uploaded_atm)
        else:
            atm_df = atm_df_default if not atm_df_default.empty else st.stop()

        # 2. 生成统一波长网格（0.3-20μm，间隔0.01μm，确保插值精度）
        lambda_grid = np.arange(lambda_min, lambda_max + 0.005, 0.01).round(2)  # 0.01μm间隔
        st.success(f"生成统一波长网格：{len(lambda_grid)}个点（{lambda_min:.1f}-{lambda_max:.1f}μm，间隔0.01μm）")

        # 3. 所有曲线插值到统一网格
        # 发射率插值
        eps_interp = interpolate_curve(lambda_grid, eps_df["波长_μm"], eps_df["发射率ε"], "发射率")
        # 大气透过率插值
        tau_atm_interp = interpolate_curve(lambda_grid, atm_df["波长_μm"], atm_df["大气透过率_τatm"], "大气透过率")
        # 太阳辐射插值（仅白天用）
        sun_interp = interpolate_curve(lambda_grid, sun_df["波长_μm"], sun_df["太阳辐射强度_Wm-2μm-1"], "太阳辐射") if is_day else np.zeros_like(lambda_grid)

        # 4. 批量计算所有参数组合
        result_list = []
        for tamb in tamb_list:
            for trad in trad_list:
                for q in q_list:
                    # 4.1 计算P_rad（材料自身辐射，仅用Trad）
                    def p_rad_integrand(lmbda_μm):
                        lmbda_m = lmbda_μm * 1e-6  # 转换为米
                        ibb = planck_law(trad, lmbda_m)  # W/(m²·sr·m)
                        eps = interpolate_curve([lmbda_μm], lambda_grid, eps_interp, "P_rad发射率")[0]
                        return ibb * eps * np.cos(theta_rad) * 1e6  # 1e6：m→μm转换

                    p_rad, _ = integrate.quad(p_rad_integrand, lambda_min, lambda_max)
                    p_rad *= 2 * np.pi  # 半球积分（2π立体角）

                    # 4.2 计算P_atm（大气逆辐射，用Tamb）
                    def p_atm_integrand(lmbda_μm):
                        lmbda_m = lmbda_μm * 1e-6
                        ibb = planck_law(tamb, lmbda_m)
                        eps = interpolate_curve([lmbda_μm], lambda_grid, eps_interp, "P_atm发射率")[0]
                        tau_atm = interpolate_curve([lmbda_μm], lambda_grid, tau_atm_interp, "P_atm透过率")[0]
                        eps_atm = 1 - (tau_atm ** (1 / np.cos(theta_rad))) if np.cos(theta_rad) > 1e-6 else 0.9
                        return ibb * eps * eps_atm * np.cos(theta_rad) * 1e6

                    p_atm, _ = integrate.quad(p_atm_integrand, lambda_min, lambda_max)
                    p_atm *= 2 * np.pi

                    # 4.3 计算P_sun（太阳辐射，仅白天）
                    p_sun = integrate.trapz(sun_interp * eps_interp, lambda_grid) if is_day else 0.0

                    # 4.4 计算P_cond_conv（非辐射损失）
                    p_cond_conv = q * (tamb - trad)

                    # 4.5 计算净功率P_net
                    p_net = p_rad - p_atm - p_sun - p_cond_conv

                    # 4.6 保存结果
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

        # 5. 结果整理与展示
        result_df = pd.DataFrame(result_list)
        st.markdown("### 📈 批量计算结果（共{}组数据）".format(len(result_df)))

        # 5.1 结果表格（分昼夜筛选）
        with st.expander("查看完整结果表格", expanded=False):
            st.dataframe(result_df, use_container_width=True, height=400)

        # 5.2 可视化：Trad-P_net曲线（固定Tamb和q，取中间值）
        st.markdown("### 📊 净功率P_net随Trad变化曲线（固定中间Tamb和q）")
        # 取中间值（避免极端值）
        tamb_mid = tamb_list[len(tamb_list)//2]
        q_mid = q_list[len(q_list)//2]
        plot_df = result_df[(result_df["环境温度Tamb（K）"] == tamb_mid) & (result_df["对流换热系数q（W/(m²·K)）"] == q_mid)]
        
        if len(plot_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(plot_df["冷却器温度Trad（K）"], plot_df["净制冷功率P_net（W/m²）"], 
                    'o-', color='darkred', linewidth=2, markersize=6, label=f"Tamb={tamb_mid}K, q={q_mid}W/(m²·K)")
            # 标记制冷/不制冷分界线
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, label="P_net=0（制冷临界点）")
            # 标记最大P_net
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

        # 5.3 结果下载（Excel分sheet）
        st.markdown("### 📥 结果下载")
        # 按昼夜模式分sheet（若有白天和夜晚数据）
        with pd.ExcelWriter('/mnt/辐射制冷功率计算结果.xlsx', engine='openpyxl') as writer:
            result_df.to_excel(writer, sheet_name=day_night, index=False)
            # 若有需要，可添加其他sheet（如参数汇总）
            cond_df = pd.DataFrame(cond_data)
            cond_df.to_excel(writer, sheet_name="计算条件", index=False)
        
        # 提供下载按钮
        with open('/mnt/辐射制冷功率计算结果.xlsx', 'rb') as f:
            st.download_button(
                label=f"下载{day_night}计算结果（Excel，含{len(result_df)}组数据）",
                data=f,
                file_name=f"辐射制冷功率计算结果_{day_night.replace('（', '_').replace('）', '')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # 5.4 关键统计信息
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




