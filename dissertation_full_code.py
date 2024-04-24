# EC331: Research in Applies Economics. Final Project code.


############################################################################################################################
# Contents:
    # 1. Loading data from AWS
        # A section dedicated to fetching the data from AWS and basic cleaning
    # 2. Assigning SOC to vacancy name
        # A section dedicated to the matching algorithm, where word vecorization is used to match a standardised occupation code to every vacancy in the posting
        # Parts of this section take several hours to run
    # 3. AIOE descriptives
        # A section creating descriptive statistics and graphs that describe the to trends in LLM exposure in the sample
    # 4. Text processing
        # A section dedicated to extracting the outcome variables from the job posting text
        # Parts of this section take several hours to run
        # [!] PLEASE DISREGARD SECTION 4.3.3, as it containts redundant code
    # 5. Estimation
        # The fitting of difference-in-difference models and creation of dynamic treatment effect plots
############################################################################################################################


# Importing necessary libraries:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import boto3
import os
import requests

from langdetect import detect, LangDetectException
import re
import itertools
from itertools import filterfalse
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from copy import deepcopy

import linearmodels as lm
import statsmodels as statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

###################################################################################################################################
# 1: Loading data from AWS
###################################################################################################################################

# Preliminary analysis
print(os.environ.get('AWS_ACCESS_KEY_ID'))
print(os.environ.get('AWS_SECRET_ACCESS_KEY'))

# Call to get filenames
s3 = boto3.client('s3', region_name='eu-central-1')

dataset_arn = 'arn:aws:s3:eu-central-1:291973504423:accesspoint/df394cdc-ff14-4ac5-bda4-5e431d02b47f'
prefix = 'lu/'
destination_folder = r"C:\Users\sofiy\OneDrive\Рабочий стол\Uni\Study\Y3\Diss\Data"
dataset_apc = '95a2f00f7b43e653136ada33cf5f014e'

filenames = []
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=dataset_arn,
                               Prefix=prefix,
                               RequestPayer='requester'):
    filenames.extend(obj['Key'] for obj in page.get('Contents', []))

# Download
for file in filenames:
    s3.download_file(
        dataset_arn,
        file,
        os.path.join(destination_folder, os.path.basename(file)),
        ExtraArgs={'RequestPayer': 'requester'}
        )
    print(f'Downloaded {file}')

# Load into a DataFrame 
jobs = pd.DataFrame()
n = 1
for filename in filenames:
    jobs = pd.concat([jobs, pd.read_json(os.path.join(destination_folder, os.path.basename(filename)), lines=True)], ignore_index=True)
    print(n)
    n+=1

# To print some observations:
for i in range(100):
    print(jobs['company'].iloc[i])

# Detecting the string language:
def safe_detect(text):
    try:
        return detect(text)
    except LangDetectException:
        return None 

# Adding a language column to the dataframe
lang_list = []
i = 0
while i <= 212249:
    newseries = jobs['text'][i:i+500].apply(lambda text: safe_detect(text) if isinstance(text, str) and text.strip() != "" else None)
    print(newseries.head())
    lang_list.append(pd.DataFrame(newseries))
    i+=500
    print(f"{i-1} processed")
lang_data = pd.concat(lang_list, ignore_index = True)
lang_data.value_counts() # en 138704, fr 64408, de 8797
lang_data.value_counts(normalize=True) # en 65%, fr 30%, de 4%, other 1%
lang_data.isna().sum() #8
print(len(lang_data))
jobs["posting_lang"] = lang_data

# Checking if language recognition is correct
jobs[["posting_lang", "text"]].head(10)

# Creating a dataframe with postings in english only:
english_filter = jobs["posting_lang"]=="en"
jobs_english = jobs[english_filter] # 138704 obs.

# Dataframe with selected columns:
jobs_sel = jobs_english[["source", "name", "text", "position", "dateCreated", "location"]]

# Working with dates 
dates = jobs_sel[["dateCreated"]] #this is a DataFrame
dates_list = list(dates["dateCreated"]) #length 138704

# Isolating lists into years_list
years_list = []
a = 0 
while a < len(dates_list):
    years_list.append(dates_list[a][:4])
    a+=1
    print(a)
print(dates_list[:20])

# Isolating months into months_list
months_list = []
b = 0 
while b < len(dates_list):
    months_list.append(dates_list[b][5:7])
    b+=1
    print(b)
print(months_list[:10])

# Isolating days into days_list
days_list = []
c = 0 
while c < len(dates_list):
    days_list.append(dates_list[c][8:10])
    c+=1
    print(c)
print(days_list[:10])

# Adding formatted dates to jobs_sel -> jobs_main is the main dataframe
jobs_sel["year"] = years_list
jobs_sel[["year"]].value_counts()

jobs_sel["month"] = months_list
jobs_sel[["month"]].value_counts()

jobs_sel["day"] = days_list
jobs_sel[["day"]].value_counts()

jobs_sel1 = jobs_sel[["source", "name", "text", "position", "location", "year", "month", "day"]]

# "Opening up" the dictionary in the "position" variable
dict_df = pd.DataFrame(jobs_sel['position'].tolist())
worktype = dict_df['workType']
wt_list = worktype.tolist()

jobs_sel1['worktype'] = wt_list

# Creating main dataframe
jobs_main = jobs_sel1[["source", "name", "text", "worktype", "location", "year", "month", "day"]]




###################################################################################################################################
# 2: Assigning SOC to vacancy name
###################################################################################################################################

#######################
# 2.1 Preprocessing - Job Data
#######################

# Source: https://github.com/nestauk/ojo_daps_mirror/blob/main/ojd_daps/flows/enrich/labs/soc/common.py

# Creating list of vacancy names from the Techmap.io dataset:
names = jobs_main[["name"]]

# Working with stopwords
RE_TERMS = re.compile(r"(\w+)")
RE_SPACES = re.compile(" +")
stopwords_unsorted = [
    "month", "months", "part time", "full time", "part", "contract", "ftc", "fte",
    "full time equivalent", "per annum", "per day", "per hour", "ft ", " ft", "pt ", " pt", "work from home",
    "remote working", "remote work", "working from home", "working remotely", "work remotely",
    "experience needed", "excellent opportunity", "urgently required", "wanted", "filling quickly",
    "apply now", "senior", "trainee", "immediate start", "newly qualified", "experience new",
    "start new", "openings for", "explore far", "away places", "now hiring", "experiences start",
    "your new", "flexible working", "new year", "new start", "new job", "positions filling",
    "bsc req", "any major", "and free", "fixed term", "german speaking", "french speaking",
    "days nights", "up to", "to hours", "to talk", "home working", "night shift", "new job",
    "degree required", "all majors accepted", "no ", "experienced", "experience", 
    "great package", "positions", "full arrival support", "free flights",
    "great package", "full arrival support", "limited openings", "early", " ba ", " bsc ", " req ",
    "any major positions filling quickly", "full support", "no experience necessary",
    " in ", "limited", "flights", "bachelor", "new job", "from ", "to ", "firm", "fixed", "term", " for "
    "now interviewing", "new year new start new job", "days", "nights", " k ", "bonus", "amp",
    "currently", "seeking", "looking", "competitive salary", "benefits", "based", "location", "industry",
    "temporary basis", "gain experience", "opportunity", "pay ", "pay rate", "m f", "f m", "f m d", "luxembourg", "europe",
    "internship", "intern ", " or ", "of ", "amazon", "deloitte", "deloitte solutions", "vienna", "paris", "london", "berlin", "munich",
    "belgium", "belgian", "work-life balance", "&amp", "fr ", " eng ", "french", "english", "german", " uk ", "lux ", "united kingdom", "emea"
]
stopwords_mid = sorted(stopwords_unsorted, key=lambda x: x.count(" "), reverse=True)
stopwords = sorted(stopwords_mid, key=lambda y: len(y), reverse=True)

prefixes = [
    "apprentice", "assistant", "chief", "departmental", "deputy",
    "head", "principal", "senior", "trainee", "under", "junior", "vice", "vice president"
]

# Defininf functions to clean vacancy names
def remove_digits(text):
    return "".join(filterfalse(str.isdigit, text))

def remove_punctuation(text):
    return " ".join(RE_TERMS.findall(text))

def standardise(text):
    text = remove_digits(text)
    text = remove_punctuation(text)
    text = text.strip().lower()
    return RE_SPACES.sub(" ", text)

def remove_stopwords(text):
# stopwords - stopword list defined above
    for word in stopwords:
        if word not in text:
            continue
        text = text.replace(word, "")
    return text

def remove_prefix(text):
# prefixes - prefix list defined above
    try:
        first, rest = text.split(" ", 1)
    except ValueError:
        first, rest = text, ""
    for pref in prefixes:
        if first == pref:
            return rest
    return text

def clean(text):
    text = standardise(text)
    text = remove_stopwords(text)
    text = remove_prefix(text)
    return standardise(text)

# This is specific to Felten's list of AI exposure:
def and_to_comma(text):
    if " and " in text:
        return text.replace(" and ", ", ")
    return text

# Testing the text cleaning functions
test_jobname = "Assistant Sales & Administrative Assistant (m/f) Now Interviewing 15 months"
a = standardise(test_jobname)
b = remove_stopwords(a)
c = remove_prefix(b)

names_clean = names.map(lambda x: clean(x))
posting_list = list(names_clean["name"])

#######################
# 2.2 Preprocessing - SOCs from Felten et al., 2023
#######################

# Loading the data from the pdf
felten_ai_exp = pd.read_csv(r"C:\Users\sofiy\OneDrive\Рабочий стол\Uni\Study\Y3\Diss\Felten_lang_AI_exp.csv")
occupation_title = felten_ai_exp["Occupation Title "]
occup_title_clean = occupation_title.map(lambda x: and_to_comma(standardise(x)))
occup_title_list = list(occup_title_clean) # what we work with subsequently

felten_ai_exp["Clean Occupation Title"] = occup_title_clean # will be used for matching

#######################
# 2.3 Capturing meaning via DistilBert
#######################

# Creating model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encoding Felten AI exposure scores:
felten_embeddings = model.encode(occup_title_list) 

# Encoding job posting list:
posting_embeddings = model.encode(posting_list)

#######################
# 2.4 Calculating cosine similarity measure
#######################

# Function to assign a felten classification with max. cosine similarity:
def max_cos_sim(posting_emb):
    soc_row = 0
    largest_soc_similarity = 0
    largest_soc_index = 0

    while soc_row <= 773:
        cossim = util.cos_sim(felten_embeddings[soc_row,:], posting_emb)
        if cossim > largest_soc_similarity:
            largest_soc_similarity = cossim
            largest_soc_index = soc_row
        soc_row +=1
    return largest_soc_index

# Function to loop through vacancy names and assign all of them similarity indices:
# Returns a list of matched felten_embeddings indices
def soc_match(vacancy_embeddings_array):
    soc_matches_list = []
    list_length = 138703
    i = 0

    while i <= list_length:
        soc_matches_list.append(max_cos_sim(vacancy_embeddings_array[i,:]))
        i += 1
        print(i)
    
    return soc_matches_list

# Applying soc_match to posting_embeddings
soc_matched = soc_match(posting_embeddings)

# List of 138704 ordered Felten SOC matches:
soc_matched_string = [occup_title_list[x] for x in soc_matched]

# Adding matched SOC string to the job postings dataframe jobs_main
jobs_main = jobs_main.reset_index()

jobs_main["clean name"] = pd.Series(posting_list)
jobs_main["Clean Occupation Title"] = pd.Series(soc_matched_string)
print(jobs_main[["name", "clean name", "Clean Occupation Title"]].iloc[20:30])

# Adding Occupational AI exposure to jobs_main dataframe - creating a jobs_master dataframe
jobs_master = pd.merge(jobs_main, felten_ai_exp, how = "left", on = ["Clean Occupation Title"])
jobs_master = jobs_master.iloc[0:138703]




###################################################################################################################################
# 3: AIOE descriptives
###################################################################################################################################

    # Note: "AIOE" = Artificial Intelligence Occupational Exposure. This is the same as LLM exposure.

# Some more data cleaning

# Creating dataframe of US SOC major groups
major_group_code = [11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55]
major_group_title = [
    "Management",
    "Business and Finance",
    "Computer and Mathematical",
    "Architecture and Engineering",
    "Life, Physical and Social Sciences",
    "Community and Social Service",
    "Legal",
    "Education and Library Occupations",
    "Arts, Design, Sports, Media and Entertainment",
    "Healthcare",
    "Healthcare Support",
    "Protective Services",
    "Food Preparation & Related",
    "Building, Cleaning and Maintenance",
    "Personal Care",
    "Sales",
    "Office and Administrative Support",
    "Farming, Fishing and Forestry",
    "Construction and Extraction",
    "Installation, Maintenance and Repair",
    "Production",
    "Transportation",
    "Military"
]

major_group_code = pd.Series(major_group_code)
major_group_title = pd.Series(major_group_title)

us_2018_soc = pd.DataFrame()
us_2018_soc["major code"] = major_group_code
us_2018_soc["major title"] = major_group_title

# Isolating the first 2 numbers from AIOE:
def soc_clean(soc):
    if " " in soc:
        soc.replace(" ", "")
    if "-" in soc:
        soc.replace("-", "")
    return str(soc)

def soc_to_major(soc):
    return int(soc[0:2])

jobs_master["major code"] = jobs_master[["Code "]].iloc[1:138704].map(lambda code: soc_to_major(soc_clean(code)))

jobs_nani = pd.merge(jobs_master, us_2018_soc, how = "left", on = ["major code"])
jobs_master = jobs_nani

jm_noduplicates = jobs_master.drop_duplicates(subset=["text"])

# (1) Summary statistics for LLM exposure:
print("AIOE Mean:", jobs_master[["AIOE "]].mean())
print("AIOE s.d:", jobs_master[["AIOE "]].std())
print("AIOE min:", jobs_master[["AIOE "]].min())
print("AIOE max:", jobs_master[["AIOE "]].max())
print("AIOE bottom 25:", jobs_master[["AIOE "]].quantile(0.25))
print("AIOE top 75:", jobs_master[["AIOE "]].quantile(0.75))
print("AIOE skewness:", jobs_master[["AIOE "]].skew())
print("AIOE kurtosis:", jobs_master[["AIOE "]].kurtosis())

# (2) AIOE histogram:
aioe_hist = jm_noduplicates["AIOE "].hist(bins=50)
aioe_hist.set_title("LLM exposure distribution in sample")
aioe_hist.set_xlabel("LLM exposure score")
aioe_hist.set_ylabel("frequency")
plt.show()

# (3) Major SOC title bar:
occupations_summary = jm_noduplicates[["major title"]].value_counts()
occupations_summary = occupations_summary.to_frame().reset_index(level=["major title"])

occupations_bar = occupations_summary.plot.bar(
    x = "major title",
    y = "count"
    )
plt.xticks(rotation=45, ha="right")
occupations_bar.set_xlabel("major occupation")
occupations_bar.set_ylabel("frequency")
occupations_bar.set_title("Major occupation groups in sample")
occupations_bar.legend([])
plt.xticks(rot=35)
plt.show()

# (4) Mean AIOE by major SOC title bar:
aioe_means = jobs_master.groupby(by="major title")[["AIOE "]].mean()
aioe_means = aioe_means.reset_index(level=["major title"]).sort_values(by="AIOE ", ascending=False)

aioe_occup_mean_bar = aioe_means.plot.bar(
    x = "major title",
    y = "AIOE ",
    legend=False
)
aioe_occup_mean_bar.bar_label(aioe_occup_mean_bar.containers[0], fmt="%.2f")
aioe_occup_mean_bar.set_title("GenAI exposure per broad occupation")
aioe_occup_mean_bar.set_xlabel("broad occupation")
aioe_occup_mean_bar.set_ylabel("mean GenAI exposure")
plt.xticks(rotation=45, ha="right")
plt.show()

# (5) Mean AIOE by year line:
aioe_years = jm_noduplicates.groupby(by = "year")[["AIOE "]].mean()
aioe_years = aioe_years.reset_index(level=["year"])

aioe_year_line = aioe_years.plot.line(
    x = "year",
    y = "AIOE "
)
plt.show()

# (6) Mean AIOE by month line
aioe_months = jm_noduplicates.groupby(by = "month")[["AIOE "]].mean()
aioe_months = aioe_months.reset_index(level=["month"])

aioe_month_line = aioe_months.plot.line(
    x = "month",
    y = "AIOE "
)
plt.show()

# (7) Evolution in posting numbers
jm_noduplicates["full date"] = pd.to_datetime(jm_noduplicates[["year","month","day"]])
jobs_master_dates = jm_noduplicates
jobs_master_dates.set_index("full date", inplace = True)
monthly_count = jobs_master_dates.resample("M").size()
monthly_jobs_plot = monthly_count.plot(kind="line")
monthly_jobs_plot.set_title("Total monthly number of postings")
monthly_jobs_plot.set_xlabel("month")
monthly_jobs_plot.set_ylabel("number of vacancies")
plt.show()

# (8) Evolution in AIOE
monthly_mean_aioe = jobs_master_dates["AIOE "].resample("M").mean()
mm_aioe_plot = monthly_mean_aioe.plot(kind="line")
mm_aioe_plot.set_title("Monthly mean of GenAI exposure")
mm_aioe_plot.set_xlabel("month")
mm_aioe_plot.set_ylabel("average GenAI exposure")
plt.axvline(x = "2022-11-30", color="red", linewidth=2)
plt.show()

monthly_mean_aioeW = jobs_master_dates["AIOE "].resample("W").mean()
mm_aioe_plotW = monthly_mean_aioeW.plot(kind="line")
mm_aioe_plotW.set_title("Weekly mean of LLM exposure")
mm_aioe_plotW.set_xlabel("week")
mm_aioe_plotW.set_ylabel("average LLM exposure")
plt.axvline(x = "2022-11-30", color="red", linewidth=2)
plt.show()

# (9) Mean AIOE by job site
aioe_source = jm_noduplicates.groupby(by = "source")[["AIOE "]].mean()
aioe_source = aioe_source.reset_index(level=["source"]).sort_values(by="AIOE ", ascending = False)

aioe_source_bar = aioe_source.plot.bar(
    x = "source",
    y = "AIOE ",
    legend = False
)
aioe_source_bar.bar_label(aioe_source_bar.containers[0], fmt="%.2f")
aioe_source_bar.set_title("Mean LLM exposure per job website")
aioe_source_bar.set_xlabel("job website")
aioe_source_bar.set_ylabel("mean LLM exposure")
plt.xticks(rotation=45, ha="right")
plt.show()

# (10) AIOE evolution: for a specific job site
jobs_linkedin = jobs_master_dates[jobs_master_dates["source"]=="xing_lu"]
monthly_mean_aioeWL = jobs_linkedin["AIOE "].resample("W").mean()
mm_aioe_plotWL = monthly_mean_aioeWL.plot(kind="line")
plt.axvline(x = "2022-11-30", color="red", linewidth=2)
plt.show()

# (11) Daily evolution in breakdown of job sites
site_breakdown = jm_noduplicates["source"].groupby(by="full date").value_counts()
site_breakdown_df = pd.DataFrame(site_breakdown).reset_index(level=["full date","source"])
site_breakdown_pivot = site_breakdown_df.pivot(index = "full date", columns = "source", values = "count")
site_breakdown_pivot.index = pd.to_datetime(site_breakdown_pivot.index)
sites_line = site_breakdown_pivot.plot(kind = "line",
                                        subplots = True,
                                        ylim  = (0, 500))
plt.show()
sites_linex = site_breakdown_pivot.plot(kind = "line",
                                        y = site_breakdown_pivot.columns)
plt.show()

# (12) Montly evolution in breakdown of job sites !!
sites_monthly = site_breakdown_pivot.resample("M").sum()
sites_lineY = sites_monthly.plot(kind = "line",
                                        y = site_breakdown_pivot.columns)
sites_lineY.set_title("Monthly number of postings per website")
sites_lineY.set_xlabel("month")
sites_lineY.set_ylabel("number of postings")
plt.show()




###################################################################################################################################
# 4: Text processing
###################################################################################################################################

#######################
# 4.1 Additional text work
#######################

text_data = jobs_master["text"]
text_data_list = list(text_data)

#######################
# 4.2 Preprocessing
#######################

nltk.download("stopwords")
nltk.download("punkt")
print(stopwords.words("english"))

text_tok_lower = [word_tokenize(pstng) for pstng in text_data_list] # this is taking long
text_tok_lower_backup = deepcopy(text_tok_lower)

#######################
# 4.3 Experience
#######################

#######################
# 4.3.1 Experience - extensive margin
#######################

# Counting experience requirements
experience_in = [("experience" in i) | ("Experience" in i) for i in text_data_list]
print(experience_in.count(True)) # 128439 - almost all

# Defining regex patterns to be searched for (example of text suiting these patterns in comments on the right):

regex1 = r"(?<!-)\d\syears" # 5 years
regex1_matches = [re.search(regex1, pstng) for pstng in text_data_list]
regex1_matches_list = [match.group() if match else "" for match in regex1_matches]

regex2 = r"1\syear\s" # 1 year
regex2_matches = [re.search(regex2, pstng) for pstng in text_data_list]
regex2_matches_list = [match.group() if match else "" for match in regex2_matches]

regex3 = r"\d\-\d\syears|\d\-\dyears|\d\-\d\d\syears|\d\d\-\d\d\syears" # 3-5 years
regex3_matches = [re.search(regex3, pstng) for pstng in text_data_list]
regex3_matches_list = [match.group() if match else "" for match in regex3_matches]
regex3_1_matches_list = [i if " y" in i else i.replace("y", " y") for i in regex3_matches_list]

regex4 = r"(?<!\d)1\d\syears" # 11 years
regex4_matches = [re.search(regex4, pstng) for pstng in text_data_list]
regex4_matches_list = [match.group() if match else "" for match in regex4_matches]

regex5 = r"\d\sto\s\d\syears|\d\sto\s\d\d\syears|\d\d\sto\s\d\d\syears" # 5 to 6 years
regex5_matches = [re.search(regex5, pstng) for pstng in text_data_list]
regex5_matches_list = [match.group() if match else "" for match in regex5_matches]

regex6 = r"1\d\+years|\d\+\syears" # 5+ years
regex6_matches = [re.search(regex6, pstng) for pstng in text_data_list]
regex6_matches_list = [match.group() if match else "" for match in regex6_matches]

regex7 = r"1\d\+\syears" # 11+ years
regex7_matches = [re.search(regex7, pstng) for pstng in text_data_list]
regex7_matches_list = [match.group() if match else "" for match in regex7_matches]

regex8 = r"\d\syrs|\dyrs" # 5 yrs, 5yrs
regex8_matches = [re.search(regex8, pstng) for pstng in text_data_list]
regex8_matches_list = [match.group() if match else "" for match in regex8_matches]

regex9 = r"(one|two|three|four|five|six|seven|eight|nine|ten)\sto\s(two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen)\syears" # "five to seven years"
regex9_matches = [re.search(regex9, pstng) for pstng in text_data_list]
regex9_matches_list = [match.group() if match else "" for match in regex9_matches]

# Transforming some into numerical values
expl1 = [int(exp[0]) if exp!='' else np.nan for exp in regex1_matches_list]
expl2 = [1 if exp!='' else np.nan for exp in regex2_matches_list]
expl3 = [(int(exp[0])+int(exp[2]))/2 if len(exp)==9 else (int(exp[0])+int(exp[2]+exp[3]))/2 if len(exp)==10 else (int(exp[0]+exp[1])+int(exp[3]+exp[4]))/2 if len(exp)==11 else np.nan for exp in regex3_1_matches_list]
expl4 = [int(exp[0]+exp[1]) if exp!='' else np.nan for exp in regex4_matches_list]
expl5 = [(int(exp[0])+int(exp[5]))/2 if len(exp)==12 else (int(exp[0])+int(exp[5]+exp[6]))/2 if len(exp)==13 else (int(exp[0]+exp[1])+int(exp[6]+exp[7]))/2 if len(exp)==14 else np.nan for exp in regex5_matches_list]
expl6 = [int(exp[0]) if exp!='' else np.nan for exp in regex6_matches_list]
expl7 = [int(exp[0]+exp[1]) if exp!='' else np.nan for exp in regex7_matches_list]
expl8 = [int(exp[0]) if exp!='' else np.nan for exp in regex8_matches_list]

def clean_ntn(regex):
    if "one" in regex:
        regex = regex.replace("one","1")
    if "two" in regex:
        regex = regex.replace("two","2")
    if "three" in regex:
        regex = regex.replace("three","3")
    if "four" in regex:
        regex = regex.replace("four","4")
    if "five" in regex:
        regex = regex.replace("five","5")
    if "six" in regex:
        regex = regex.replace("six","6")
    if "seven" in regex:
        regex = regex.replace("seven","7")
    if "eight" in regex:
        regex = regex.replace("eight","8")
    if "nine" in regex:
        regex = regex.replace("nine","9")
    if "ten" in regex:
        regex = regex.replace("ten","10")
    if "twelve" in regex:
        regex = regex.replace("twelve","12")
    if "fifteen" in regex:
        regex = regex.replace("fifteen","15")
    return regex

expl9_prelim = [clean_ntn(exp) if exp!='' else exp for exp in regex9_matches_list]
expl9 = [(int(exp[0])+int(exp[5]))/2 if len(exp)==12 else (int(exp[0])+int(exp[5]+exp[6]))/2 if len(exp)==13 else (int(exp[0]+exp[1])+int(exp[6]+exp[7]))/2 if len(exp)==14 else np.nan for exp in expl9_prelim]

# Creating a list indicating average experience required in a job posting:
experience_master = [np.nan for i in range(138703)]

for n in range(len(experience_master)):
    if not np.isnan(expl3[n]):
        experience_master[n] = expl3[n]
    elif not np.isnan(expl5[n]):
        experience_master[n] = expl5[n]
    elif not np.isnan(expl9[n]):
        experience_master[n] = expl9[n]
    elif not np.isnan(expl6[n]):
        experience_master[n] = expl6[n]
    elif not np.isnan(expl7[n]):
        experience_master[n] = expl7[n]
    elif not np.isnan(expl8[n]):
        experience_master[n] = expl8[n]
    elif not np.isnan(expl4[n]):
        experience_master[n] = expl4[n]
    elif not np.isnan(expl1[n]):
        experience_master[n] = expl1[n]
    elif not np.isnan(expl2[n]):
        experience_master[n] = expl2[n]
    else:
        experience_master[n] = experience_master[n]

# experience - column detailing the experience required in a job posting (numerical)
jobs_master["experience"] = pd.Series(experience_master)

#######################
# 4.3.2 Experience - visualisation, descriptives
#######################

jobs_master_exp = jobs_master

# Mean experience required
jobs_master_exp["experience"].mean() #4.19

# Histogram
exp_hist = jobs_master_exp["experience"].hist(bins=20, grid=False)
exp_hist.set_xlim(0,20)
exp_hist.set_title("Years of experience required: frequency")
plt.show()

# Mean experience by occupation
exp_occup = jobs_master_exp.groupby(by="major title")[["experience"]].mean()
exp_occup = exp_occup.reset_index(level=["major title"])

exp_occup_mean_bar = exp_occup.plot.bar(
    x = "major title",
    y = "experience"
)
exp_occup_mean_bar.xticks(rot=45)
exp_occup_mean_bar.bar_label(exp_occup_mean_bar.containers[0])
plt.show()

# Mean experience by year
exp_occupY = jobs_master_exp.groupby(by="year")[["experience"]].mean()
exp_occupY = exp_occupY.reset_index(level=["year"])

exp_occupY_mean_bar = exp_occupY.plot.bar(
    x = "year",
    y = "experience",
    rot=45
)
exp_occupY_mean_bar.bar_label(exp_occupY_mean_bar.containers[0])
plt.show()

# Mean experience by month
exp_occupM = jobs_master_exp.groupby(by="month")[["experience"]].mean()
exp_occupM = exp_occupM.reset_index(level=["month"])

exp_occupM_mean_bar = exp_occupM.plot.bar(
    x = "month",
    y = "experience",
    rot=45
)
exp_occupM_mean_bar.bar_label(exp_occupM_mean_bar.containers[0])
plt.show()

# Mean experience by site
exp_occupS = jobs_master_exp.groupby(by="source")[["experience"]].mean()
exp_occupS = exp_occupS.reset_index(level=["source"])

exp_occupS_mean_bar = exp_occupS.plot.bar(
    x = "source",
    y = "experience",
    rot=45
)
exp_occupS_mean_bar.bar_label(exp_occupS_mean_bar.containers[0])
plt.show()

# Mean experience by occupation and year
exp_occup_year = jobs_master_exp.groupby(by = ["source", "year"])[["experience"]].mean()
exp_occup_year_1 = exp_occup_year.reset_index(level=["source", "year"])
exp_occup_year_pivot  = exp_occup_year_1.pivot(index = "source", columns = "year", values = "experience")

exp_o_y_bar = exp_occup_year_pivot.plot.bar(rot=45)
plt.show() 

#######################
# 4.3.3 Experience - keyword search
#######################

experience_extracts = dict()

text_tok_lower_backup = text_tok_lower[0:1001]
text_tok_lower_testing = deepcopy(text_tok_lower_backup)

def extract_experience(tokenized_posting):
    exp_list=[]
    while "experience" in tokenized_posting:
        idx = tokenized_posting.index("experience")
        idx_start = idx - 6
        idx_end = idx + 10
        exp_list.extend(tokenized_posting[idx_start:idx_end])
        del tokenized_posting[idx]
    while "Experience" in tokenized_posting:
        idx = tokenized_posting.index("Experience")
        idx_start = idx +1
        idx_end = idx + 7
        exp_list.extend(tokenized_posting[idx_start:idx_end])
        del tokenized_posting[idx]
    while "experience" in exp_list:
        exp_list.remove("experience")
    return exp_list

exper_list = [extract_experience(j) for j in text_tok_lower_testing]
exper_list[:10]

exper_df = pd.DataFrame()
exper_df["Experience extract"] = pd.Series(exper_list)

experience_preferred = set(["plus", "preferably", "preferable", "preferred", "advantage"])

exper_df["Experience preferred"] = exper_df["Experience extract"].map(lambda extract: True if set(extract).intersection(experience_preferred)!=set([]) else False)


#######################
# 4.4 Education
#######################

text_tok_lower_backup = deepcopy(text_tok_lower)

def lowercasing(alist):
    return [i.lower() if i.isalnum() else i for i in alist]

text_tok_lower_backup1 = [lowercasing(j) for j in text_tok_lower_backup]

jobs_main["text"].iloc[135007]

def needs_degree(tokenlist):
    return ("degree" in tokenlist) | ("educated" in tokenlist) | ("university" in tokenlist) | ("academic" in tokenlist) | ("educational" in tokenlist)
needs_degree_list = [needs_degree(i) for i in text_tok_lower_backup1]

# Type of degree identifier:
def degree_type(tokenlist):
    degree = ""
    if ("bachelor" in tokenlist) |("bachelors" in tokenlist) | ("bsc" in tokenlist) | ("ba" in tokenlist) | ("undergraduate" in tokenlist):
        degree += "B"
    if ("master" in tokenlist) | ("masters" in tokenlist) | ("postgraduate" in tokenlist) | ("msc" in tokenlist):
        degree += "M"
    if ("postgraduate" in tokenlist) | ("phd" in tokenlist) | ("doctorate" in tokenlist):
        degree += "P"
    return degree

degree_type_list = [degree_type(i) for i in text_tok_lower_backup1]

# More accurate degree identifier - in case the type of degree is mentioned, but not the degree itself:
degree_req_list = [(needs_degree_list[i] | (degree_type_list[i]!="")) for i in range(138703)]

# Including "Any degree" type into degree type identifier:
degree_type_list_v2 = ["Any" if (needs_degree_list[i] & (degree_type_list[i]=="")) else degree_type_list[i] for i in range(138703)]

# Adding columns to the main dataframe
jobs_master["Degree required"] = pd.Series(degree_req_list)
jobs_master["Degree type"] = pd.Series(degree_type_list_v2)

jobs_master["Degree required"].value_counts(normalize=True)
jobs_master["Degree type"].value_counts(normalize=True)

jobs_master.groupby("year")["Degree required"].mean()
jobs_master.groupby("year")["Degree type"].value_counts(normalize=True)

# Making time series plots:

jmaster_edu = deepcopy(jobs_master)

jmed = jmaster_edu.groupby(["high_exposure", "year", "month"])[["Degree required"]].mean()

jmed1 = jmed.reset_index(level=["high_exposure", "year", "month"])
jmed1["day"] = pd.Series([15 for i in range(len(jme1))])
jmed1["ymd"] = pd.to_datetime(jme1[["year", "month", "day"]])
jmed2 = jmed1.set_index("ymd").drop(labels=["year", "month", "day"], axis=1)

jmed3 = jmed2.pivot(columns="high_exposure", values=["Degree required"]).resample("QS-NOV").mean()

# (Plot for degree requirement:
ax = jmed3.plot(
    kind = "line",
    color=["turquoise","blue"]
)
plt.axvline(pd.Timestamp('2022-11-15'), color='r')
ax.set_xlabel("quarter")
ax.set_ylabel("proportion requiring degree")
ax.set_title("Evolution in proportion of postings requiring university education\nfor high vs low GenAI exposure occupations")
ax.set_ylim(0.3, 0.8)
plt.show()

# Plot for degree type:

jmedx = jmaster_edu.groupby(["high_exposure", "year", "month"])[["Degree type"]].value_counts(normalize=True)
jmedx1 = jmedx.reset_index(level=["high_exposure", "year", "month", "Degree type"])
jmedx1["day"] = pd.Series([15 for i in range(len(jmedx1))])
jmedx1["ymd"] = pd.to_datetime(jmedx1[["year", "month", "day"]])
jmedx2 = jmedx1.set_index("ymd").drop(labels=["year", "month", "day"], axis=1)

jmedx3 = jmedx2.pivot(columns=["high_exposure", "Degree type"], values=["proportion"]).resample("QS-NOV").mean()

ax = jmedx3.plot(
    kind="line",
    y="proportion"
)
plt.show()

#######################
# 4.5 Skills
#######################

text_tok_lower_backup = deepcopy(text_tok_lower)

# Defining soft skill dictionaries

language_req = ["english", "french", "german", "nordic"]
language_fluency_req = ["fluent", "proficient", "fluency", "proficiency"]

interpersonal_req = ["communication", "communicate", "communicating",
                     "interpersonal", "presentation", "presentations", "present", "presenting", "influence", "persuade", "relationships",
                     "collaborate", "collaborative", "collaboration",
                     "team", "teamwork", "part of team", "part of a team", "team spirit", "team player"]

leadership_req = ["lead", "leadership",
                 "project management", "manage teams", "manage a team", "managing a team",
                 "motivate", "coordinate", "coordination"]

cognitive_req = ["analysis", "analytical", "analyse", "analyze", "analyzing", "analysing"
                 "problem", "problems", "problem solving", "problem-solving", "problem resolution", "solution", "solution-oriented", "solution oriented", "solutions", "problem solver"
                 "decision-making", "decision making", "decisions", "judgment", "judgements", 
                 "diagnose", "suggest", "suggestions", "propose", "summarise", "summarize", "summarizing", "summarising", "interpret", "interpreting", "structurings"]

timemgmt_req = ["plan", "planning", "organize", "organise", "organization", "organizational", "organisation", "organisational", "organized", "well-organized", "organised",
                "deadlines", "workload", "priorities", "prioritise", "prioritize", "timekeeping", "time management", "time-management", "simultaneous", "multiple", "timely"]

creativity_req = ["open-mindedness", "open-minded", "open minded",
                  "creativity", "creative", "create", "creating",
                  "innovative", "innovation", "innovate", "entrepreneurial", "ideas"]

motivation_req = ["motivated", "interest", "interested", "committed", "ambitious",
                  "initiative", "initiatives", "proactive", "proactively", "proactivity", "pro-active",
                  "passion", "passionate", "hard working", "hard-working", "hardworking"]

flexibility_req = ["adapt", "ambigous", "adaptable", "adaptibility", "versatile"] # "flexible/flexibility" not included - too much noise

independence_req = ["independent", "independently", "self-starter", "autonomous", "autonomy", "autonomously",
                    "responsibility", "responsible", "ownership"]

# Function to assign a "skill score"
def assign_skill_score(job_string, req_list):
    count = 0
    for req in req_list:
        count += job_string.count(req)
    return count

# Lowercasing string for easier analysis
text_data_list_lower = [" ".join(i) for i in text_tok_lower_backup]

# Creating lists of skill scores
interpersonal_score = [assign_skill_score(i, interpersonal_req) for i in text_data_list_lower]
leadership_score = [assign_skill_score(i, leadership_req) for i in text_data_list_lower]
cognitive_score = [assign_skill_score(i, cognitive_req) for i in text_data_list_lower]
timemgmt_score = [assign_skill_score(i, timemgmt_req) for i in text_data_list_lower]
creativity_score = [assign_skill_score(i, creativity_req) for i in text_data_list_lower]
motivation_score = [assign_skill_score(i, motivation_req) for i in text_data_list_lower]
flexibility_score = [assign_skill_score(i, flexibility_req) for i in text_data_list_lower]
independence_score = [assign_skill_score(i, independence_req) for i in text_data_list_lower]

# Adding to jobs_master dataframe
jobs_master["interpersonal skills"] = pd.Series(interpersonal_score)
jobs_master["leadership skills"] = pd.Series(leadership_score)
jobs_master["cognitive skills"] = pd.Series(cognitive_score)
jobs_master["time management skills"] = pd.Series(timemgmt_score)
jobs_master["creativity"] = pd.Series(creativity_score)
jobs_master["motivation"] = pd.Series(motivation_score)
jobs_master["flexibility"] = pd.Series(flexibility_score)
jobs_master["independence"] = pd.Series(independence_score)


# Visual analysis

jm_skills = deepcopy(jobs_master)

jm_skills1 = jm_skills.groupby(["high_exposure", "year", "month"])[["interpersonal skills", 
                                                                             "leadership skills",
                                                                             "cognitive skills",
                                                                             "time management skills",
                                                                             "creativity",
                                                                             "motivation",
                                                                             "flexibility",
                                                                             "independence"]].mean() # This is almost suitable

jm_skills2 = jm_skills1.reset_index(level=["high_exposure", "year", "month"])
jm_skills2["day"] = pd.Series([15 for i in range(138704)])
jm_skills2["ymd"] = pd.to_datetime(jm_skills2[["year", "month", "day"]])
jm_skills2 = jm_skills2.set_index("ymd")
jm_skills2 = jm_skills2.drop(labels=["year", "month", "day"], axis=1)
jm_interpersonal = jm_skills2[["independence", "high_exposure"]]
jm_interpersonal_reshape = jm_interpersonal.pivot(columns="high_exposure", values="independence").resample("QS-NOV").mean()

# Plot for interpersonal:

ax = jm_interpersonal_reshape.plot(
    kind = "line",
    color=["turquoise","blue"]
)
plt.axvline(pd.Timestamp('2022-11-15'), color='r', label="ChatGPT shock")
ax.set_title("Evolution in independence demands\nfor high vs low GenAI exposure occupations")
ax.set_xlabel("quarter")
ax.set_ylabel("independence demands")
plt.show()


###################################################################################################################################
# 5: Estimation
###################################################################################################################################

#######################
# 5.1 Additional data work
#######################

# Creating a clean DF with only the columns needed
jobs_master = pd.read_csv(r"C:\Users\sofiy\OneDrive\Рабочий стол\Uni\Study\Y3\Diss\JobsMaster.csv")
jobs_master_estimation = deepcopy(jobs_master)

# Creating transformations of the AIOE score:
    # linear transformation to make the scores positive:
jobs_master_estimation["aioe_+"] = jobs_master_estimation["AIOE "].map(lambda x: x+1.793)
    # squared:
jobs_master_estimation["aioe_2"] = jobs_master_estimation["AIOE "].map(lambda x: (x+1.793)**2)
    # cubed:
jobs_master_estimation["aioe_3"] = jobs_master_estimation["AIOE "].map(lambda x: x**3)
    # square root:
jobs_master_estimation["aioe_sqrt"] = jobs_master_estimation["AIOE "].map(lambda x: (x+1.793)**0.5)
    # natural logarithm:
jobs_master_estimation["aioe_ln"] = jobs_master_estimation["AIOE "].map(lambda x: np.log(x+1.793001))
    # quadratic equation:
jobs_master_estimation["aioe_quad"] = jobs_master_estimation["aioe_+"] + jobs_master_estimation["aioe_2"]

# Creating dummies for every quarter
jobs_master_estimation["dmy"] = pd.to_datetime(jobs_master_estimation[["year", "month", "day"]])
jobs_master_estimation["q"] = pd.PeriodIndex(jobs_master_estimation.dmy, freq="Q-NOV")
jobs_master_estimation["q"].unique()

qlist = list(jobs_master_estimation["q"].unique())
qmatch = [qlist.index(i)+1 for i in qlist]
qmatch_df = pd.DataFrame(data = {"q": qlist, "q_idx": qmatch})
jobs_master_estimation = pd.merge(jobs_master_estimation, qmatch_df, how = "left", on = ["q"])

q_dummies = pd.get_dummies(jobs_master_estimation["q_idx"], prefix="q")
jobs_master_estimation = pd.concat([jobs_master_estimation, q_dummies], axis=1)

# Creating dummies for high exposure
aioe_lintransf = "AIOE "
jobs_master_estimation[aioe_lintransf].describe()
jobs_master_estimation[[aioe_lintransf]].quantile([.25, .33, .5, .67, .75], axis = 0)
aioe_threshold = 1.253
jobs_master_estimation["exposure_bool"] = jobs_master_estimation[aioe_lintransf].map(lambda x: x >= aioe_threshold)
jobs_master_estimation["exposure_bool"].mean()

# Creating dummies for post treatment
jobs_master_estimation["post_bool"] = jobs_master_estimation["q_idx"].map(lambda quart: quart >= 12)

# Creating interaction
jobs_master_estimation["interaction_simple"] = jobs_master_estimation["exposure_bool"] * jobs_master_estimation["post_bool"]

# Creating interaction variables - with AIOE (for continuous specification)
for j in range(1,18):
    jobs_master_estimation[f"interact_{j}"] = jobs_master_estimation[aioe_lintransf] * jobs_master_estimation[f"q_{j}"]

# Creating interaction variables - with exposure_bool (for discrete specification)
for j in range(1,18):
    jobs_master_estimation[f"interact1_{j}"] = jobs_master_estimation["exposure_bool"] * jobs_master_estimation[f"q_{j}"]

# Creating dummies for every occupation
occup_dummies = pd.get_dummies(jobs_master_estimation["Clean Occupation Title"])
jobs_master_estimation = pd.concat([jobs_master_estimation, occup_dummies], axis=1)
    # list for model purposes:
occuptitle_list = list(jobs_master_estimation["Clean Occupation Title"].unique())

# Creating dummies for source
source_dummies = pd.get_dummies(jobs_master_estimation["source"], prefix="src")
jobs_master_estimation = pd.concat([jobs_master_estimation, source_dummies], axis=1)

# Creating dummies for quarter (seasonality controls):
def assign_q_dummy(qidx):
    if qidx in (1,5,9,13,17):
        return "1"
    elif qidx in (2,6,10,14):
        return "2"
    elif qidx in (3,7,11,15):
        return "3"
    else:
        return "4"
jobs_master_estimation["QinY_dummy"] = jobs_master_estimation["q_idx"].map(lambda x: assign_q_dummy(x))
QinY_dummies = pd.get_dummies(jobs_master_estimation["QinY_dummy"], prefix = "QinY")
jobs_master_estimation = pd.concat([jobs_master_estimation, QinY_dummies], axis=1)

# Adding extensive and intensive margin skill dummies
jobs_master_estimation["interpersonal int"] = pd.Series(interpersonal_score)
jobs_master_estimation["leadership int"] = pd.Series(leadership_score)
jobs_master_estimation["cognitive int"] = pd.Series(cognitive_score)
jobs_master_estimation["time management int"] = pd.Series(timemgmt_score)
jobs_master_estimation["creativity int"] = pd.Series(creativity_score)
jobs_master_estimation["motivation int"] = pd.Series(motivation_score)
jobs_master_estimation["flexibility int"] = pd.Series(flexibility_score)
jobs_master_estimation["independence int"] = pd.Series(independence_score)

jobs_master_estimation["interpersonal ext"] = jobs_master_estimation["interpersonal int"].map(lambda x: x!=0)
jobs_master_estimation["leadership ext"] = jobs_master_estimation["leadership int"].map(lambda x: x!=0)
jobs_master_estimation["cognitive ext"] = jobs_master_estimation["cognitive int"].map(lambda x: x!=0)
jobs_master_estimation["time management ext"] = jobs_master_estimation["time management int"].map(lambda x: x!=0)
jobs_master_estimation["creativity ext"] = jobs_master_estimation["creativity int"].map(lambda x: x!=0)
jobs_master_estimation["motivation ext"] = jobs_master_estimation["motivation int"].map(lambda x: x!=0)
jobs_master_estimation["flexibility ext"] = jobs_master_estimation["flexibility int"].map(lambda x: x!=0)
jobs_master_estimation["independence ext"] = jobs_master_estimation["independence int"].map(lambda x: x!=0)

# Removing duplicates:
len(jobs_master_estimation)
jme_noduplicates = jobs_master_estimation.drop_duplicates(subset=["text"])
len(jme_noduplicates)
jobs_master_estimation=jme_noduplicates

# Creating clusters for standard errors
clusters = jobs_master_estimation['Clean Occupation Title']

# New variable
jobs_master_estimation["Experience needed float"] = jobs_master_estimation["Experience needed"].map(lambda x: float(x))
jobs_master_estimation["degree req float"] = jobs_master_estimation["Degree required"].map(lambda deg: float(deg))

jobs_master_estimation.to_csv(r"C:\Users\sofiy\OneDrive\Рабочий стол\Uni\Study\Y3\Diss\final_data.csv")


#######################
# 5.2 Estimation
#######################

#######################
# 5.2.1 Defining model specifications
####################### 

controls = [
    "src_careerjet_lu", "src_efinancialcareers_lu", "src_indeed_lu", "src_linkedin_lu", # posting website controls, src_eures_lu default
    "src_monster2_lu", "src_monster_lu", "src_reed_lu", "src_totaljobs_lu", "src_xing_lu"]

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
# 5.2.2 Defining functions for model estimation
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
# 5.2.3 Estimating models
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