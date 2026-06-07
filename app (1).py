import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")

# Load dataset file
df_wide = pd.read_csv("df_wide_final.csv")
y_var = "SP.POP.65UP.TO"

# Auto filter columns: remove categorical grouping columns, leave all numeric independent variables
drop_cols = ["country_code","year","region","incomegroup"]
all_numeric_cols = [col for col in df_wide.columns if col not in drop_cols]
full_independent_list = [x for x in all_numeric_cols if x != y_var]

# Complete full variable dictionary covering all WB indicators in dataset
full_var_map = {
    # Dependent variable
    "SP.POP.65UP.TO": "Share of population aged 65 and above (%) | Dependent Variable",
    # Demographic, birth & death indicators
    "SP.DYN.CBRT.IN": "Crude birth rate (per 1,000 population)",
    "SP.DYN.CDRT.IN": "Crude death rate (per 1,000 population)",
    "SP.POP.BRTH.MF": "Sex ratio at birth (male per 100 female newborns)",
    "SP.DYN.LE00.IN": "Life expectancy at birth, total (years)",
    "SP.DYN.LE00.FE.IN": "Life expectancy at birth, female (years)",
    "SP.DYN.LE00.MA.IN": "Life expectancy at birth, male (years)",
    "SP.DYN.TFRT.IN": "Total fertility rate (births per woman)",
    "SP.ADO.TFRT": "Adolescent fertility rate (births per 1,000 women ages 15-19)",
    "SP.DYN.AMRT.FE": "Adult mortality rate, female (per 1,000)",
    "SP.DYN.AMRT.MA": "Adult mortality rate, male (per 1,000)",
    "SP.DYN.IMRT.IN": "Infant mortality rate (per 1,000 live births)",
    "SP.DYN.IMRT.FE.IN": "Infant mortality rate, female",
    "SP.DYN.IMRT.MA.IN": "Infant mortality rate, male",
    "SP.DYN.TO65.FE.ZS": "Female population aged 65 and above (%)",
    "SP.DYN.TO65.MA.ZS": "Male population aged 65 and above (%)",
    "SP.POP.GROW": "Annual population growth rate (%)",
    # Maternal health and nutrition indicators
    "SH.ANM.ALLW.ZS": "Anemia prevalence among women of reproductive age (%)",
    "SH.ANM.NPRG.ZS": "Anemia prevalence in pregnant women (%)",
    "SH.PRG.ANEM": "Prevalence of anemia during pregnancy",
    # Child mortality indicators
    "SH.DTH.IMRT": "Infant mortality (per 1000 live births)",
    "SH.DTH.MORT": "Under-five mortality rate",
    "SH.DTH.NMRT": "Neonatal mortality rate",
    "SH.DYN.MORT": "Under-five child mortality rate",
    "SH.DYN.MORT.FE": "Under-five mortality, female",
    "SH.DYN.MORT.MA": "Under-five mortality, male",
    "SH.DYN.NMRT": "Neonatal mortality rate (per 1000 live births)",
    # Vaccination & immunization data
    "SH.IMM.IDPT": "DPT immunization coverage (% of children under 1)",
    "SH.IMM.MEAS": "Measles immunization coverage (% of children under 1)",
    # Maternal mortality related metrics
    "SH.MMR.DTHS": "Number of maternal deaths",
    "SH.MMR.RISK": "Lifetime risk of maternal death",
    "SH.MMR.RISK.ZS": "Maternal death risk ratio (%)",
    "SH.STA.MMRT": "Maternal mortality ratio (per 100,000 live births)",
    # Sanitation, drinking water and chronic disease
    "SH.H2O.BASW.ZS": "Access to basic drinking water services (%)",
    "SH.STA.BASS.ZS": "Access to basic sanitation services (%)",
    "SH.STA.BRTC.ZS": "Low-birthweight babies (% of births)",
    "SH.STA.DIAB.ZS": "Diabetes prevalence (% of population aged 20+)",
    "SH.STA.ODFC.ZS": "Prevalence of overweight (% of adults)",
    # Medical workforce and hospital resource
    "SH.MED.BEDS.ZS": "Hospital beds per 1,000 population",
    "SH.MED.NUMW.P3": "Nursing and midwifery personnel per 1,000",
    "SH.MED.PHYS.ZS": "Physicians per 1,000 population",
    # Various health expenditure indicators
    "SH.XPD.CHEX.GD.ZS": "Total health expenditure (% of GDP)",
    "SH.XPD.CHEX.PC.CD": "Total health expenditure per capita (current USD)",
    "SH.XPD.CHEX.PP.CD": "Total health expenditure per capita (PPP USD)",
    "SH.XPD.GHED.CH.ZS": "Gov health expenditure (% of total health expenditure)",
    "SH.XPD.GHED.GD.ZS": "Government health spending (% GDP)",
    "SH.XPD.GHED.PC.CD": "Government health expenditure per capita(current USD)",
    "SH.XPD.GHED.PP.CD": "Government health expenditure per capita(PPP USD)",
    "SH.XPD.OOPC.CH.ZS": "Out-of-pocket health spending (% total health exp)",
    "SH.XPD.OOPC.PC.CD": "Out-of-pocket per capita (current USD)",
    "SH.XPD.OOPC.PP.CD": "Out-of-pocket per capita (PPP USD)",
    "SH.XPD.PVTD.CH.ZS": "Private domestic health spending (% total health)",
    "SH.XPD.PVTD.PC.CD": "Private health expenditure per capita(current USD)",
    "SH.XPD.PVTD.PP.CD": "Private health expenditure per capita(PPP USD)"
}
# Auto match variables existing in current dataset only
exist_var_list = all_numeric_cols
final_dict = {k:v for k,v in full_var_map.items() if k in exist_var_list or k == y_var}

# ===================== Page Layout Setting =====================
st.set_page_config(page_title="Population Aging Analysis Dashboard", layout="wide")
# Leave one blank line below main title
st.markdown("### Interactive Dashboard for Population Aging Empirical Research")
st.markdown("&nbsp;")

# Tab order:1 Dataset,2 Correlation,3 Trend,4 Regression,5 ARIMA,6 Variable Dictionary,7 About
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Dataset Overview",
    "🔥 Correlation Heatmap",
    "📈 Trend Visualization",
    "📉 Fixed-Effect Regression",
    "🔮 ARIMA Forecast",
    "📋 Variable Dictionary",
    "📖 About This App"
])

# Tab1: Dataset filtering module
with tab1:
    col_left, col_right = st.columns([1,3])
    with col_left:
        income_opt = ["All"] + sorted(df_wide["incomegroup"].unique().tolist())
        sel_income = st.selectbox("Select Income Group", income_opt)
        region_opt = ["All"] + sorted(df_wide["region"].unique().tolist())
        sel_region = st.selectbox("Select Region", region_opt)
        min_y, max_y = int(df_wide["year"].min()), int(df_wide["year"].max())
        yr_start, yr_end = st.slider("Select Year Range", min_y, max_y, (min_y, max_y))

        df_filter = df_wide.copy()
        if sel_income != "All":
            df_filter = df_filter[df_filter["incomegroup"] == sel_income]
        if sel_region != "All":
            df_filter = df_filter[df_filter["region"] == sel_region]
        df_filter = df_filter[(df_filter["year"]>=yr_start) & (df_filter["year"]<=yr_end)]

        csv_data = df_filter.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Filtered CSV", csv_data, "Filtered_Data.csv")

    with col_right:
        st.subheader("Descriptive Statistics")
        desc_df = df_filter[[y_var]+full_independent_list].describe()
        st.dataframe(desc_df.round(3))
        st.subheader("Raw Data Preview")
        st.dataframe(df_filter.head(15))

# Tab2: Correlation heatmap, no pre-selected variables by default
with tab2:
    col_side, col_plot = st.columns([1,3])
    with col_side:
        pick_vars = st.multiselect("Choose Variables for Correlation", full_independent_list, default=[])
        show_annot = st.checkbox("Show Correlation Value", value=True)
    with col_plot:
        if len(pick_vars)>=2:
            corr_data = df_filter[pick_vars].dropna()
            corr_mat = corr_data.corr()
            fig, ax = plt.subplots(figsize=(12,9))
            sns.heatmap(corr_mat, annot=show_annot, cmap="coolwarm", fmt=".2f", linewidths=0.3, ax=ax)
            plt.title("Pearson Correlation Heatmap", fontsize=15)
            st.pyplot(fig)
        elif len(pick_vars) < 2:
            st.info("Please select at least two indicators to generate heatmap.")

# Tab3: Trend plot, freely select dependent or independent indicators
with tab3:
    col_ctrl, col_chart = st.columns([1,3])
    with col_ctrl:
        group_type = st.radio("Group by", ["Single Country Code","Income Group","Region"])
        trend_pool = [y_var] + full_independent_list
        trend_ind = st.selectbox("Select Indicator", trend_pool, index=0)
        t_start, t_end = st.slider("Year Filter", min_y, max_y, (min_y, max_y))
        df_trend = df_filter[(df_filter["year"]>=t_start)&(df_filter["year"]<=t_end)].dropna(subset=[trend_ind])
    with col_chart:
        fig,ax = plt.subplots(figsize=(12,6))
        if group_type=="Single Country Code":
            country_list = sorted(df_trend["country_code"].unique())
            sel_ctry = st.selectbox("Pick Country Code", country_list)
            df_p = df_trend[df_trend["country_code"]==sel_ctry]
            sns.lineplot(data=df_p, x="year", y=trend_ind, ax=ax, linewidth=2, errorbar=None)
        elif group_type=="Income Group":
            sns.lineplot(data=df_trend, x="year", y=trend_ind, hue="incomegroup", ax=ax, linewidth=2, errorbar=None)
        else:
            sns.lineplot(data=df_trend, x="year", y=trend_ind, hue="region", ax=ax, linewidth=2, errorbar=None)
        plt.grid(alpha=0.3)
        plt.ylabel(trend_ind)
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True)
        plt.tight_layout()
        st.pyplot(fig)

# Tab4: Fixed effect panel regression
with tab4:
    st.markdown(f"Dependent Variable: {y_var}")
    x_pick = st.multiselect("Select Independent Variables", full_independent_list)
    fe_mode = st.radio("Fixed Effect Type", ["Individual FE","Two-way FE(Country+Year)"])
    run_reg = st.button("▶️ Run Regression", type="primary")
    if run_reg and len(x_pick)>=1:
        reg_df = df_filter[["country_code","year",y_var]+x_pick].dropna().set_index(["country_code","year"])
        dep = reg_df[y_var]
        exog = reg_df[x_pick]
        if fe_mode == "Individual FE":
            mod = PanelOLS(dependent=dep, exog=exog, entity_effects=True)
        else:
            mod = PanelOLS(dependent=dep, exog=exog, entity_effects=True, time_effects=True)
        res = mod.fit()

        # ---------------------- Coefficient Bar Chart (Display First) ----------------------
        st.subheader("Coefficient Distribution Bar Chart")

        # Extract coefficient, p-value and confidence interval from regression results
        coef_df = pd.DataFrame({
            "Variable": res.params.index,
            "Coefficient": res.params.values,
            "P_Value": res.pvalues.values
        })

        # Remove constant term, only keep selected independent variables
        coef_df = coef_df[coef_df["Variable"] != "constant"].reset_index(drop=True)

        # Sort data by coefficient value for better visual comparison
        coef_df = coef_df.sort_values("Coefficient").reset_index(drop=True)

        # Create figure, adjust height dynamically based on variable quantity
        fig, ax = plt.subplots(figsize=(11, len(coef_df) * 0.55))

        # Set different colors for positive and negative coefficients
        bar_color = ["#2E86AB" if val > 0 else "#F24236" for val in coef_df["Coefficient"]]

        # Draw horizontal bar chart for regression coefficients
        bars = ax.barh(coef_df["Variable"], coef_df["Coefficient"], color=bar_color, alpha=0.8)

        # Add vertical reference line at 0 to distinguish positive/negative effects
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2)

        # Add coefficient value labels next to each bar
        for bar in bars:
            width = bar.get_width()
            # Adjust label position for positive and negative values
            if width > 0:
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va="center", fontsize=9)
            else:
                ax.text(width - 0.01, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va="center", fontsize=9)

        # Set chart title and axis labels
        ax.set_title("Regression Coefficients of Independent Variables", fontsize=14, pad=15)
        ax.set_xlabel("Coefficient Value", fontsize=12)
        ax.set_ylabel("Independent Variable", fontsize=12)

        # Optimize layout to avoid label overlap
        plt.tight_layout()

        # Render chart in Streamlit
        st.pyplot(fig)
        
        # -------------------------------------------------------------------
        # Display original regression summary table (Display After Chart)
        st.subheader("Regression Result")
        st.write(res.summary)   


# Tab5: ARIMA univariate forecast only for aging rate
with tab5:
    col_cfg, col_out = st.columns([1,3])
    with col_cfg:
        ctry_list = sorted(df_wide["country_code"].unique())
        sel_c = st.selectbox("Select Country Code", ctry_list)
        pred_horizon = st.number_input("Forecast Years", min_value=3, max_value=30, value=10)
        run_ar = st.button("🔍 Run Auto ARIMA", type="primary")
    with col_out:
        if run_ar:
            ts_df = df_wide[df_wide["country_code"]==sel_c].sort_values("year").dropna(subset=[y_var])
            ts_series = ts_df[y_var].values
            model = auto_arima(ts_series, trace=False, suppress_warnings=True)
            fc_res = model.predict(n_periods=pred_horizon, return_conf_int=True)
            fc_vals, ci_low, ci_high = fc_res[0], fc_res[1][:,0], fc_res[1][:,1]
            future_year = np.arange(max(ts_df["year"])+1, max(ts_df["year"])+1+pred_horizon)

            fig,ax = plt.subplots(figsize=(12,5))
            ax.plot(ts_df["year"], ts_series, label="Historical", lw=2)
            ax.plot(future_year, fc_vals, color="red", label="Forecast", lw=2)
            ax.fill_between(future_year, ci_low, ci_high, alpha=0.2, color="gray", label="95% Confidence Interval")
            plt.legend()
            plt.title(f"ARIMA Forecast: {sel_c} Aging Rate")
            st.pyplot(fig)

            pred_table = pd.DataFrame({
                "Future_Year":future_year,
                "Forecast":np.round(fc_vals,3),
                "CI_Lower95":np.round(ci_low,3),
                "CI_Upper95":np.round(ci_high,3)
            })
            st.dataframe(pred_table)

# Tab6: Variable definition dictionary page
with tab6:
    st.markdown("### 📋 Variable Name Dictionary & Full Description")
    st.divider()
    dict_df = pd.DataFrame(list(final_dict.items()),columns=["WB Raw Code","Detailed Indicator Definition"])
    st.dataframe(dict_df,use_container_width=True,height=600)
    st.info("All indicators sourced from World Bank World Development Indicators(WDI) Database.")

# Tab7: About page, one blank line below title
with tab7:
    st.markdown("### 📖 About This Application")
    st.markdown("&nbsp;")

    st.markdown("**Course: WQD7003 – Data Analytics**")
    st.markdown("**Dataset: World Bank Open Data (1960–2025)**")
    st.markdown("**Core Models: Fixed-Effect Panel Regression + ARIMA Time Series Forecasting**")

    st.markdown("#### Purpose")
    st.markdown("This dashboard supports empirical research on global population ageing: (1) explore how health, medical resource, fertility and socioeconomic indicators affect national elderly population (65+) proportion via panel regression; (2) implement country-level future ageing rate prediction using ARIMA time-series model.")

    st.markdown("#### Methodology")
    st.markdown("""
- Panel Fixed-Effect regression for coefficient estimation to identify key driving factors of population ageing
- Iterative VIF screening (VIF<10) to eliminate severe multicollinearity and filter valid explanatory variables
- ARIMA univariate time-series model for out-of-sample forecast of national ageing ratio
- Grouped time-series visualization by geographic region & national income group for descriptive analysis
""")
    st.markdown("*Note: All raw indicators are log-transformed and within-group de-meaned before fixed-effect modeling.*")
