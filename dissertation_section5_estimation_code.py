#######################
# Estimation of Average Treatment Effects and Average Causal Responses
#######################

#######################
# 1 Pre-processing
#######################

# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import linearmodels as lm
import statsmodels as statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load cleaned dataset
jobs_master_estimation = pd.read_csv(r"C:\Users\sofiy\OneDrive\Рабочий стол\Uni\Study\Y3\Diss\final_data.csv")

# Create variables to use in estimation
    # list of occupations
occuptitle_list = list(jobs_master_estimation["Clean Occupation Title"].unique())
    # clusters for standard errors
clusters = jobs_master_estimation['Clean Occupation Title']
    # list of quarters
qlist = list(jobs_master_estimation["q"].unique())
    # posting website controls (src_eures_lu default):      
controls = [
    "src_careerjet_lu", "src_efinancialcareers_lu", "src_indeed_lu", "src_linkedin_lu",
    "src_monster2_lu", "src_monster_lu", "src_reed_lu", "src_totaljobs_lu", "src_xing_lu"]


#######################
# 2 Defining model specifications
####################### 

# Key to specification names below:
    # s - static (1 dummy for post-treatment), d - dynamic with discrete treatment, dc - dynamic with continuous treatment
    # n - no controls, c - including source controls
    # e.g. d_c: dynamic model with a discrete treatment variable and source controls included

# Static specifications:
    # no controls
s_n = ["exposure_bool", "post_bool", "interaction_simple"]
    # with controls
s_c = ["exposure_bool", "post_bool", "interaction_simple"]
for control in controls:
    s_c.append(control)

# Dynamic specifications with discrete treatment:
    # no controls
d_n = ["exposure_bool", 
        "q_1", "q_2", "q_3", "q_4", "q_5", "q_6", "q_7", "q_8", "q_9", "q_10", "q_11", "q_13", "q_14", "q_15", "q_16", "q_17", 
        "interact1_1", "interact1_2", "interact1_3", "interact1_4", "interact1_5", "interact1_6", "interact1_7", "interact1_8", "interact1_9", "interact1_10", "interact1_12", "interact1_13", "interact1_14", "interact1_15", "interact1_16", "interact1_17"]
    # with controls
d_c = ["exposure_bool", 
        "q_1", "q_2", "q_3", "q_4", "q_5", "q_6", "q_7", "q_8", "q_9", "q_10", "q_11", "q_13", "q_14", "q_15", "q_16", "q_17", 
        "interact1_1", "interact1_2", "interact1_3", "interact1_4", "interact1_5", "interact1_6", "interact1_7", "interact1_8", "interact1_9", "interact1_10", "interact1_12", "interact1_13", "interact1_14", "interact1_15", "interact1_16", "interact1_17"]
for control in controls:
    d_c.append(control)

# Dynamic specifications with continuous treatment:
    # no controls
dc_n = ["q_1", "q_2", "q_3", "q_4", "q_5", "q_6", "q_7", "q_8", "q_9", "q_10", "q_11", "q_13", "q_14", "q_15", "q_16", "q_17", 
        "interact_1", "interact_2", "interact_3", "interact_4", "interact_5", "interact_6", "interact_7", "interact_8", "interact_9", "interact_10", "interact_12", "interact_13", "interact_14", "interact_15", "interact_16", "interact_17"]
for occup in occuptitle_list:
    dc_n.append(occup)
    # with controls
dc_c = ["q_1", "q_2", "q_3", "q_4", "q_5", "q_6", "q_7", "q_8", "q_9", "q_10", "q_11", "q_13", "q_14", "q_15", "q_16", "q_17", 
        "interact_1", "interact_2", "interact_3", "interact_4", "interact_5", "interact_6", "interact_7", "interact_8", "interact_9", "interact_10", "interact_12", "interact_13", "interact_14", "interact_15", "interact_16", "interact_17"]
for occup in occuptitle_list:
    dc_c.append(occup)
for control in controls:
    dc_c.append(control)
 

#######################
# 3 Defining functions for model estimation
####################### 

# Function to estimate difference-in-differences results, plot treatment effects and print out estimations:
def diff_in_diff(x_variables, y_variable, plot):
    X = jobs_master_estimation[x_variables]
    if "interact1_1" in x_variables:
        X = sm.add_constant(X) # only add constant fot variation where AIOE is discrete
    Y = jobs_master_estimation[y_variable]
    spec_gen = sm.OLS(Y, X.astype(float)).fit(cov_type="cluster", cov_kwds={"groups": clusters})

    coeffs = spec_gen.params
    ci = spec_gen.conf_int()
    se = spec_gen.bse

    if plot == "continuous":
        toplot = ["interact_1", "interact_2", "interact_3", "interact_4", "interact_5", "interact_6", "interact_7", "interact_8", "interact_9", 
                                    "interact_10", "interact_12", "interact_13", "interact_14", "interact_15", "interact_16", "interact_17"]
        sel_coeffs = coeffs[toplot]
        sel_coeffs = pd.concat([sel_coeffs.loc[:"interact_10"], pd.Series([0.00], index=['default']), sel_coeffs.loc['interact_12':]])
        sel_se = se[toplot]
        sel_se = pd.concat([sel_se.loc[:"interact_10"], pd.Series([0.00], index=['default']), sel_se.loc['interact_12':]])
        sel_ci = ci.loc[toplot]
        sel_ci = pd.concat([sel_ci.loc[:"interact_10"], pd.DataFrame([{0:0.00,1:0.00}], index=['default']), sel_ci.loc['interact_12':]])
        error_bars = (sel_ci[1]-sel_ci[0])/2
    
    elif plot=="discrete":
        toplot = ["interact1_1", "interact1_2", "interact1_3", "interact1_4", "interact1_5", "interact1_6", "interact1_7", "interact1_8", "interact1_9", 
                                    "interact1_10", "interact1_12", "interact1_13", "interact1_14", "interact1_15", "interact1_16", "interact1_17"]
        sel_coeffs = coeffs[toplot]
        sel_coeffs = pd.concat([sel_coeffs.loc[:"interact1_10"], pd.Series([0.00], index=['default']), sel_coeffs.loc['interact1_12':]])
        sel_se = se[toplot]
        sel_se = pd.concat([sel_se.loc[:"interact1_10"], pd.Series([0.00], index=['default']), sel_se.loc['interact1_12':]])
        sel_ci = ci.loc[toplot]
        sel_ci = pd.concat([sel_ci.loc[:"interact1_10"], pd.DataFrame([{0:0.00,1:0.00}], index=['default']), sel_ci.loc['interact1_12':]])
        error_bars = (sel_ci[1]-sel_ci[0])/2
   
    plt.figure(figsize=(10,5))
    plt.errorbar(x=range(len(sel_coeffs)),y=sel_coeffs, yerr=error_bars, fmt="o", capsize=5, color="blue")
    plt.plot(range(len(sel_coeffs)), sel_coeffs, '-o', color="blue") 
    plt.axhline(y=0, color='red', linestyle='-')
    plt.axvline(x=10, color='red', linestyle='-')
    plt.xticks(range(len(qlist)), qlist, rotation=45)
    plt.grid(False)
    plt.xlabel("Quarter")
    if plot=="discrete":
        plt.title("Dynamic treatment effect: binary dummy "+"["+y_variable+"]")
        plt.ylabel("Treatment Effect")
        plt.show()
    elif plot=="continuous":
        plt.title("Average causal effect (ACE): AIOE as continuous treatment "+"["+y_variable+"]")
        plt.ylabel("ACE")
        plt.show()
    else:
        print("no plot")
    
    print("R2:", spec_gen.rsquared)

    for var in toplot: 
        print(f"Results for {var}:")
        print(f"  Coefficient: {spec_gen.params[var]}")
        print(f"  Std Error: {spec_gen.bse[var]}")
        print(f"  T-statistic: {spec_gen.tvalues[var]}")
        print(f"  P-value: {spec_gen.pvalues[var]}")
    print("95% CI", sel_ci)


# Function to set the linear transformation of Felten's LLM exposure score
    # This is useful for estimations with continuous treatment
def set_aioe_lintransf(lintransf):
    for j in range(1,18):
        jobs_master_estimation[f"interact_{j}"] = jobs_master_estimation[lintransf] * jobs_master_estimation[f"q_{j}"]
    print(jobs_master_estimation[[lintransf]].quantile([.01, .25, .33, .5, .67, .75, .99], axis = 0))


# Function to set the LLM exposure cutoff score that defines the treatment and the control in the discrete treatment variable case:
def set_aioe_thresh(thresh):
    jobs_master_estimation["exposure_bool"] = jobs_master_estimation["AIOE "].map(lambda x: x >= thresh)
    jobs_master_estimation["interaction_simple"] = jobs_master_estimation["exposure_bool"] * jobs_master_estimation["post_bool"]
    for j in range(1,18):
        jobs_master_estimation[f"interact1_{j}"] = jobs_master_estimation["exposure_bool"] * jobs_master_estimation[f"q_{j}"]

    # Possible thresh values:
        # mean: 0.936
        # 25th percentile: 0.678
        # 50th percentile: 1.106
        # 75th percentile: 1.295
        # 33th percentile: 0.882
        # 67th percentile: 1.253
    

#######################
# 4 Estimating models
####################### 

# Estimation of ATE (discrete treatment specification):
set_aioe_lintransf("AIOE ")
    # with no controls:
set_aioe_thresh(1.106)
diff_in_diff(d_n, "Experience needed float", "discrete")
diff_in_diff(d_n, "degree req float", "discrete")
diff_in_diff(d_n, "interpersonal int", "discrete")
diff_in_diff(d_n, "leadership int", "discrete")
set_aioe_thresh(0.936)
diff_in_diff(d_n, "cognitive int", "discrete")
diff_in_diff(d_n, "time management int", "discrete")
set_aioe_thresh(0.882)
diff_in_diff(d_n, "creativity int", "discrete")
diff_in_diff(d_n, "motivation int", "discrete")
set_aioe_thresh(1.106)
diff_in_diff(d_n, "flexibility int", "discrete")
set_aioe_thresh(1.253)
diff_in_diff(d_n, "independence ext", "discrete")
set_aioe_thresh(1.106)
diff_in_diff(d_n, "interpersonal ext", "discrete")
diff_in_diff(d_n, "leadership ext", "discrete")
set_aioe_thresh(0.936)
diff_in_diff(d_n, "cognitive ext", "discrete")
set_aioe_thresh(1.106)
diff_in_diff(d_n, "time management ext", "discrete")
set_aioe_thresh(0.882)
diff_in_diff(d_n, "creativity ext", "discrete")
diff_in_diff(d_n, "motivation ext", "discrete")
set_aioe_thresh(1.106)
diff_in_diff(d_n, "flexibility ext", "discrete")
set_aioe_thresh(1.253)
diff_in_diff(d_n, "independence ext", "discrete")
    # with controls:
set_aioe_thresh(1.106)
diff_in_diff(d_c, "Experience needed float", "discrete")
diff_in_diff(d_c, "degree req float", "discrete")
diff_in_diff(d_c, "interpersonal int", "discrete")
diff_in_diff(d_c, "leadership int", "discrete")
set_aioe_thresh(0.936)
diff_in_diff(d_c, "cognitive int", "discrete")
diff_in_diff(d_c, "time management int", "discrete")
set_aioe_thresh(0.882)
diff_in_diff(d_c, "creativity int", "discrete")
diff_in_diff(d_c, "motivation int", "discrete")
set_aioe_thresh(1.106)
diff_in_diff(d_c, "flexibility int", "discrete")
set_aioe_thresh(1.253)
diff_in_diff(d_c, "independence ext", "discrete")
set_aioe_thresh(1.106)
diff_in_diff(d_c, "interpersonal ext", "discrete")
diff_in_diff(d_c, "leadership ext", "discrete")
set_aioe_thresh(0.936)
diff_in_diff(d_c, "cognitive ext", "discrete")
set_aioe_thresh(1.106)
diff_in_diff(d_c, "time management ext", "discrete")
set_aioe_thresh(0.882)
diff_in_diff(d_c, "creativity ext", "discrete")
diff_in_diff(d_c, "motivation ext", "discrete")
set_aioe_thresh(1.106)
diff_in_diff(d_c, "flexibility ext", "discrete")
set_aioe_thresh(1.253)
diff_in_diff(d_c, "independence ext", "discrete")

# Estimation of ACR (continuous treatment specification):

# Setting parameters:
        # The specification used is where Felten's LLM exposure score is cubed, as it produces the most well-behaved ATE/ACR estimates
        # Other options include:  AIOE (original score), aioe_2 (squared), aioe_sqrt (square root), aioe_ln (ln), aioe_quad (to 4th power)
set_aioe_lintransf("aioe_3")

    # with no controls:
diff_in_diff(dc_n, "Experience needed float", "continuous")
diff_in_diff(dc_n, "degree req float", "continuous")
diff_in_diff(dc_n, "interpersonal int", "continuous")
diff_in_diff(dc_n, "leadership int", "continuous")
diff_in_diff(dc_n, "cognitive int", "continuous")
diff_in_diff(dc_n, "time management int", "continuous")
diff_in_diff(dc_n, "creativity int", "continuous")
diff_in_diff(dc_n, "motivation int", "continuous")
diff_in_diff(dc_n, "flexibility int", "continuous")
diff_in_diff(dc_n, "independence ext", "continuous")
diff_in_diff(dc_n, "interpersonal ext", "continuous")
diff_in_diff(dc_n, "leadership ext", "continuous")
diff_in_diff(dc_n, "cognitive ext", "continuous")
diff_in_diff(dc_n, "time management ext", "continuous")
diff_in_diff(dc_n, "creativity ext", "continuous")
diff_in_diff(dc_n, "motivation ext", "continuous")
diff_in_diff(dc_n, "flexibility ext", "continuous")
diff_in_diff(dc_n, "independence ext", "continuous")
    # with controls:
diff_in_diff(dc_c, "Experience needed float", "continuous")
diff_in_diff(dc_c, "degree req float", "continuous")
diff_in_diff(dc_c, "interpersonal int", "continuous")
diff_in_diff(dc_c, "leadership int", "continuous")
diff_in_diff(dc_c, "cognitive int", "continuous")
diff_in_diff(dc_c, "time management int", "continuous")
diff_in_diff(dc_c, "creativity int", "continuous")
diff_in_diff(dc_c, "motivation int", "continuous")
diff_in_diff(dc_c, "flexibility int", "continuous")
diff_in_diff(dc_c, "independence ext", "continuous")
diff_in_diff(dc_c, "interpersonal ext", "continuous")
diff_in_diff(dc_c, "leadership ext", "continuous")
diff_in_diff(dc_c, "cognitive ext", "continuous")
diff_in_diff(dc_c, "time management ext", "continuous")
diff_in_diff(dc_c, "creativity ext", "continuous")
diff_in_diff(dc_c, "motivation ext", "continuous")
diff_in_diff(dc_c, "flexibility ext", "continuous")
diff_in_diff(dc_c, "independence ext", "continuous")

# Seperate estimation and plots for Experience along the intensive margin

# Filter dataframes for estimation to exclude postings not mentioning experience
flt1 = jobs_master_estimation["experience"]!=0
flt2 = pd.notnull(jobs_master_estimation["experience"])
flt3 = jobs_master_estimation["Experience needed"]==True
flt = flt1 & flt2 & flt3
jobs_master_ei = jobs_master_estimation[flt]
clustx = jobs_master_ei["Clean Occupation Title"]

# Fitting static, 1-dummy DiD
X = jobs_master_ei[["exposure_bool", "post_bool", "interaction_simple"]]
X = sm.add_constant(X)
Y = jobs_master_ei["experience"]
spec21 = sm.OLS(Y, X.astype(float)).fit(cov_type="cluster", cov_kwds={"groups": clustx})
print(spec21.summary())

# Fitting dynamic, 1-dummy DiD
set_aioe_thresh(1.106)
X = jobs_master_ei[["exposure_bool", 
                           "q_1", "q_2", "q_3", "q_4", "q_5", "q_6", "q_7", "q_8", "q_9", "q_10", "q_12", "q_13", "q_14", "q_15", "q_16", "q_17", 
                           "interact1_1", "interact1_2", "interact1_3", "interact1_4", "interact1_5", "interact1_6", "interact1_7", "interact1_8", "interact1_9", 
                            "interact1_10", "interact1_12", "interact1_13", "interact1_14", "interact1_15", "interact1_16", "interact1_17",
                            "src_careerjet_lu", "src_efinancialcareers_lu", "src_indeed_lu", "src_linkedin_lu", # posting website controls, src_eures_lu default
    "src_monster2_lu", "src_monster_lu", "src_reed_lu", "src_totaljobs_lu", "src_xing_lu"]]

X = sm.add_constant(X)
Y = jobs_master_ei["experience"]
spec22 = sm.OLS(Y, X.astype(float)).fit(cov_type="cluster", cov_kwds={"groups": clustx})
print(spec22.summary())

# Plot
coeffs = spec22.params
ci = spec22.conf_int()
se = spec22.bse

toplot = ["interact1_1", "interact1_2", "interact1_3", "interact1_4", "interact1_5", "interact1_6", "interact1_7", "interact1_8", "interact1_9", 
                            "interact1_10", "interact1_12", "interact1_13", "interact1_14", "interact1_15", "interact1_16", "interact1_17"]
sel_coeffs = coeffs[toplot]
sel_coeffs = pd.concat([sel_coeffs.loc[:"interact1_10"], pd.Series([0.00], index=['default']), sel_coeffs.loc['interact1_12':]])
sel_se = se[toplot]
sel_se = pd.concat([sel_se.loc[:"interact1_10"], pd.Series([0.00], index=['default']), sel_se.loc['interact1_12':]])
sel_ci = ci.loc[toplot]
sel_ci = pd.concat([sel_ci.loc[:"interact1_10"], pd.DataFrame([{0:0.00,1:0.00}], index=['default']), sel_ci.loc['interact1_12':]])
error_bars = (sel_ci[1]-sel_ci[0])/2

plt.figure(figsize=(10,5))
plt.errorbar(x=range(len(sel_coeffs)),y=sel_coeffs, yerr=error_bars, fmt="o", capsize=5, color="blue")
plt.plot(range(len(sel_coeffs)), sel_coeffs, '-o', color="blue") 
plt.axhline(y=0, color='red', linestyle='-')
plt.axvline(x=10, color='red', linestyle='-')
plt.xticks(range(len(qlist)), qlist, rotation=45)
plt.grid(False)
plt.title("Dynamic treatment effect: binary dummy [Years of experience]")
plt.xlabel("Quarter")
plt.ylabel("Treatment Effect")
plt.show()


# Fitting dynamic DiD with continuous treatment:
set_aioe_lintransf("aioe_3")
xvarlist = ["q_1", "q_2", "q_3", "q_4", "q_5", "q_6", "q_7", "q_8", "q_9", "q_10", "q_12", "q_13", "q_14", "q_15", "q_16", "q_17", 
                           "interact_1", "interact_2", "interact_3", "interact_4", "interact_5", "interact_6", "interact_7", "interact_8", "interact_9", 
                            "interact_10", "interact_12", "interact_13", "interact_14", "interact_15", "interact_16", "interact_17"]
for x in occuptitle_list:
    xvarlist.append(x)
for x in controls:
    xvarlist.append(x)

X = jobs_master_ei[xvarlist]
Y = jobs_master_ei["experience"]
spec23 = sm.OLS(Y, X.astype(float)).fit(cov_type="cluster", cov_kwds={"groups": clustx})
print(spec23.summary())

# Plot
coeffs = spec23.params
ci = spec23.conf_int()
se = spec23.bse

toplot = ["interact_1", "interact_2", "interact_3", "interact_4", "interact_5", "interact_6", "interact_7", "interact_8", "interact_9", 
                            "interact_10", "interact_12", "interact_13", "interact_14", "interact_15", "interact_16", "interact_17"]
sel_coeffs = coeffs[toplot]
sel_coeffs = pd.concat([sel_coeffs.loc[:"interact_10"], pd.Series([0.00], index=['default']), sel_coeffs.loc['interact_12':]])
sel_se = se[toplot]
sel_se = pd.concat([sel_se.loc[:"interact_10"], pd.Series([0.00], index=['default']), sel_se.loc['interact_12':]])
sel_ci = ci.loc[toplot]
sel_ci = pd.concat([sel_ci.loc[:"interact_10"], pd.DataFrame([{0:0.00,1:0.00}], index=['default']), sel_ci.loc['interact_12':]])
error_bars = (sel_ci[1]-sel_ci[0])/2

plt.figure(figsize=(10,5))
plt.errorbar(x=range(len(sel_coeffs)),y=sel_coeffs, yerr=error_bars, fmt="o", capsize=5, color="blue")
plt.plot(range(len(sel_coeffs)), sel_coeffs, '-o', color="blue") 
plt.axhline(y=0, color='red', linestyle='-')
plt.axvline(x=10, color='red', linestyle='-')
plt.xticks(range(len(qlist)), qlist, rotation=45)
plt.grid(False)
plt.title("Average causal effect (ACE): AIOE as continuous treatment [Years of experience]")
plt.xlabel("Quarter")
plt.ylabel("ACE")
plt.show()

print("R2:", spec23.rsquared)

for var in toplot: 
    print(f"Results for {var}:")
    print(f"  Coefficient: {spec23.params[var]}")
    print(f"  Std Error: {spec23.bse[var]}")
    print(f"  T-statistic: {spec23.tvalues[var]}")
    print(f"  P-value: {spec23.pvalues[var]}")
    print("95% CI", sel_ci)

# End