# Basic Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

############### Data Manipulation and Data Cleaning ###############

# Load df
df = pd.read_csv('data/hospital_readmissions.csv')

df.info()
df.isna().sum()
df.duplicated().sum()

# Check for cat variables and and special characters
for column, values in df.iteritems():
    unique_values = values.sort_values().unique()
    print(f"Unique values in column '{column}': {unique_values}\n")

num = df.select_dtypes(exclude=['object'])

# Plot data distribution
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes[-1, -1].remove()
sns.set_style('dark')
sns.set_palette('Blues_r')

for ax, col in zip(axes.flatten(), num.columns):
    sns.kdeplot(num[col], ax=ax, fill=True)
    ax.set_title(col, fontsize=15)

plt.subplots_adjust(hspace=0.3, wspace=0.3)

plt.show()

# Basic statistic
df.describe()

# Plot Basic statistic and outliers
print('Figure 13. Subplot distribution and outliers.')
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes[-1, -1].remove()
sns.set_palette('bright')

# Iterate over the axes and the columns to fill the subplots with box-plots
for ax, col in zip(axes.flatten(), num.columns):
    ax.boxplot(num[col], medianprops={'color': 'mediumseagreen'})
    ax.set_title(col, fontsize=15)
    ax.set_ylabel('Value count')
    ax.set_xticks([])

plt.subplots_adjust(hspace=0.2, wspace=0.2)

plt.show()

# Inspecting outliers
Q1 = num.quantile(0.25)
Q3 = num.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = num[(num < lower_bound) | (num > upper_bound)]

print(f"The total number of outliers is: {outliers.count().sum()}\n"
      f"\nNumber of outliers in each columns:\n{outliers.count()}")

# Feature correlation
df['readmitted'] = df.readmitted.map({'yes': 1, 'no': 0})
corr = df.corr()

# Plot correlation
fig = plt.figure(figsize=(9, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap='Blues', mask=mask)

plt.show()

# Ranking table diagnosis by age group
age_group = df.groupby(['age', 'diag_1']).size().reset_index(name='counts')

pivot_table = pd.pivot_table(age_group,
                             index='diag_1',
                             columns='age',
                             values='counts').drop('Missing', axis=0).rank(ascending=False, axis=0)


def color_rank_one(val):
    """This function applies the orange background color to the first rank"""
    if val == 1:
        return 'background-color: mediumturquoise'
    else:
        return ''


pivot_table.style.format('{:,.0f}').background_gradient(cmap='Blues_r', axis=0).applymap(color_rank_one)

# Plot diagnosis by age group
fig, ax = plt.subplots(figsize=(10, 10))
sns.set_style('white')

diagnosis = df[df['diag_1'] != 'Missing']
unique_diags = diagnosis.diag_1.unique()

blues = sns.color_palette('Blues', n_colors=len(unique_diags))
custom_palette = ["mediumturquoise" if diag == "Circulatory"
                  else blues[len(unique_diags) - 1 - i]
                  for i, diag in enumerate(unique_diags)]

sns.countplot(y='age',
              hue='diag_1',
              order=diagnosis.age.sort_values().unique(),
              palette=custom_palette,
              data=diagnosis
              )

for bar in ax.patches:
    width = bar.get_width()
    x = width
    y = bar.get_y() + bar.get_height() / 2
    label = f"{width:.0f}"
    ax.annotate(label, (x, y),
                ha='left', va='center',
                xytext=(3, 0), fontsize=10,
                textcoords='offset points'
                )

sns.despine(bottom=True)
plt.ylabel('Age Group', fontsize=12)
plt.xlabel('')
plt.xticks([])
plt.title('Count primary diagnosis by age group')

plt.show()

# Maps the values in the 'readmitted' column
df['readmitted'] = df.readmitted.map({1: 'yes', 0: 'no'})

# Filtering diag_1 equal diabetes
df_diag_1 = df[df['diag_1'] == 'Diabetes']

# Table readmission rate by diag_1 equal diabetes
table_1 = (df_diag_1['readmitted']
           .value_counts(normalize=True)
           .mul(100)
           .round()
           .reset_index(name='readmission rate (%)')
           .rename(columns={'index': 'readmitted'})
           )

table_1

# Filtering diag_2 equal diabetes
df_diag_2 = df[df['diag_2'] == 'Diabetes']

# Table readmission rate by diag_2 equal diabetes
table_2 = (df_diag_2['readmitted']
           .value_counts(normalize=True)
           .mul(100).round()
           .reset_index(name='readmission rate (%)')
           .rename(columns={'index': 'readmitted'})
           )

table_2

# Filtering diag_1 equal diabetes
df_diag_3 = df[df['diag_3'] == 'Diabetes']

# Table readmission rate by diag_3 equal diabetes
table_3 = (df_diag_3['readmitted']
           .value_counts(normalize=True)
           .mul(100)
           .round()
           .reset_index(name='readmission rate (%)')
           .rename(columns={'index': 'readmitted'})
           )

table_3

# Filtering df different by 'Diabetes'
other_diag = df[df.apply(lambda x: 'Diabetes' not in x.values, axis=1)]

# Df other diagnosis
other_diag = (other_diag[['readmitted', 'diag_1', 'diag_2', 'diag_3']]
              .melt(id_vars='readmitted', var_name='diag', value_name='diseases')
              .drop('diag', axis=1)
              )

# Table readmission rate for each of the other diagnoses.
other_readmission = (other_diag.
                     groupby('diseases')
                     .value_counts(normalize=True)
                     .mul(100)
                     .round()
                     .reset_index(name='readmission rate (%)')
                     )

other_readmission = other_readmission[other_readmission['diseases'] != 'Missing']

# Table readmission by other diagnosis
toal_other_readmission = (other_readmission
                          .groupby('readmitted')['readmission rate (%)']
                          .mean()
                          .reset_index(name='readmission rate (%)')
                          )

toal_other_readmission

# Concat  table_1, table_2, table_3
diabetes_diag = pd.concat([table_1, table_2, table_3])

# Table diabetes_diag
diabetes_diag = pd.DataFrame(diabetes_diag
                             .groupby('readmitted')['readmission rate (%)']
                             .mean()
                             .reset_index()
                             )

# Table diabetes_other_diag
diabetes_other_diag = pd.concat([diabetes_diag, toal_other_readmission])

# Filtering for readmitted = yes
diabetes_other_diag = diabetes_other_diag[diabetes_other_diag['readmitted'] == 'yes']

diabetes_other_diag['diagnosis'] = ['Diabetes', 'Other']

# Sorting columns
diabetes_other_diag = diabetes_other_diag.reindex(columns=['diagnosis', 'readmitted', 'readmission rate (%)'])

diabetes_other_diag

# Plot Readmission Rate by Diabetes and Other Diagnosis
sns.set_style('whitegrid')
sns.set_palette('Paired')
fig, ax = plt.subplots(figsize=(8, 6))

sns.barplot(x='diagnosis', y='readmission rate (%)', data=diabetes_other_diag)

for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(f"{height:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 1), textcoords='offset points',
                ha='center', va='bottom', fontsize=12)

sns.despine(left=True)
plt.title('Readmission Rate by Diabetes and Other Diagnosis\nfor Patients Readmitted', fontsize=15)
plt.xlabel('Diagnosis', fontsize=12)
plt.ylabel('Readmission Rate (%)', fontsize=12)
plt.show()

############### Chi-squared test ###############
# Set the p-value threshold to 0.05
alpha = 0.05

# Create a new column 'has_diabetes' indicating whether the primary diagnosis is diabetes
df['has_diabetes'] = df.diag_1.str.contains('Diabetes')

# Perform a chi-squared independence test between 'has_diabetes' and 'readmitted' and obtain the p-value
expected, observed, stats = pg.chi2_independence(data=df, y='has_diabetes', x='readmitted', correction=False)

df.drop('has_diabetes', axis=1, inplace=True)

print(
    '---------------------------------------    Results Chi-squared test    ---------------------------------------\n')
print(f'The p_value is: {stats.pval.max()} \nIs p_value less than alpha?: {stats.pval.max() < alpha}')

##Other table
# table Readmission vs glucose test
table_4 = (df
           .groupby('glucose_test')['readmitted']
           .value_counts(normalize=True)
           .mul(100)
           .round()
           .reset_index(name='readmission rate (%)')
           )

table_4

# Table readmission vs AC1 test
table_5 = (df
           .groupby('A1Ctest')['readmitted']
           .value_counts(normalize=True)
           .mul(100)
           .round()
           .reset_index(name='readmission rate (%)')
           )

table_5

# Table Readmission vs diabetes medication
table_6 = (df
           .groupby('diabetes_med')['readmitted']
           .value_counts(normalize=True)
           .mul(100)
           .round()
           .reset_index(name='readmission rate (%)')
           )

table_6

# Table readmission vs change medication
table_7 = (df
           .groupby('change')['readmitted']
           .value_counts(normalize=True)
           .mul(100)
           .round()
           .reset_index(name='readmission rate (%)')
           )

table_7

# Table readmission vs time in hospital
table_8 = (df
           .groupby('time_in_hospital')['readmitted']
           .value_counts(normalize=True)
           .mul(100)
           .round()
           .reset_index(name='readmission rate (%)')
           )

table_8

# Plot Readmission Rates by Each Diabetes Diagnosis
sns.set_style('whitegrid')
sns.set_palette('Paired')

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 4))

for i, table in enumerate([table_1, table_2, table_3]):
    ax = axes[i]
    sns.barplot(x='readmitted', y='readmission rate (%)', data=table.sort_values('readmitted'), ax=ax)
    ax.set_title(f'diag_{i + 1}')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Annotate bar plot with percentage values
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(f"{height:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12)

fig.suptitle("Readmission Rates by Each Diabetes Diagnosis", fontsize=15)
fig.text(0.5, -0.02, 'Readmission', ha='center', fontsize=13)
fig.text(0, 0.5, 'Percentages', va='center', fontsize=13, rotation=90)
plt.tight_layout()
sns.despine(left=True)

plt.show()

# Subplots Other Tables
sns.set_style('whitegrid')
fig, axes = plt.subplots(5, 2, sharey='row', figsize=(20, 25))
axes[-1, -1].remove()

# Plot Readmission vs Glucose Test
ax = sns.barplot(x='glucose_test',
                 y='readmission rate (%)',
                 hue='readmitted',
                 order=['high', 'normal', 'no'],
                 hue_order=['no', 'yes'],
                 data=table_4,
                 ax=axes[0, 0]
                 )

for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.0f}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='center',
                xytext=(0, 6), fontsize=12,
                textcoords='offset points')

ax.legend(title='Readmission', loc='upper right', bbox_to_anchor=(1.16, 1))
ax.set_xlabel('Glucose Test', fontsize=13)
ax.set_ylabel('Readmission Rates (%)', fontsize=13)
ax.set_title('Readmission Rate by Patients with Glucose Test', fontsize=15)
sns.despine(left=True)

# Plot Readmission vs AC1 Test
ax = sns.barplot(x='A1Ctest',
                 y='readmission rate (%)',
                 hue='readmitted',
                 order=['high', 'normal', 'no'],
                 data=table_5,
                 ax=axes[0, 1])

for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.0f}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='center',
                xytext=(0, 6), fontsize=12,
                textcoords='offset points')

ax.legend().set_visible(False)
ax.set_xlabel('A1C Test', fontsize=13)
ax.set_ylabel('')
ax.set_title('Readmission Rate by Patients with A1C Test', fontsize=15)
sns.despine(left=True)

# Plot Readmission vs Medication
data_yes = table_6[table_6['diabetes_med'] == 'yes']
ax = sns.barplot(x='readmitted', y='readmission rate (%)', data=data_yes, ax=axes[1, 0])
ax.set_title('Readmission Rate by Patients \n With Diabetes Medication', fontsize=15)
ax.set_xlabel('Readmission', fontsize=13)
ax.set_ylabel('Readmission Rates (%)', fontsize=13)

for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.0f}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='center',
                xytext=(0, 6), fontsize=12,
                textcoords='offset points')

sns.despine(left=True)

# Plot Readmission vs No Medication
data_no = table_6[table_6['diabetes_med'] == 'no']
ax = sns.barplot(x='readmitted', y='readmission rate (%)', data=data_no, ax=axes[1, 1])
ax.set_title('Readmission Rate by Patients \n Without Diabetes Medication', fontsize=15)
ax.set_xlabel('Readmission', fontsize=13)
ax.set_ylabel('')

for bar in ax.patches:
    ax.annotate(f'{bar.get_height()}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='center',
                xytext=(0, 6), fontsize=12,
                textcoords='offset points')

# Plot Readmission vs Change Medication
data_yes = table_7[table_7['change'] == 'yes']
ax = sns.barplot(x='readmitted', y='readmission rate (%)', data=data_yes, ax=axes[2, 0])
ax.set_title('Readmission Rate by Patients \n With Change Diabetes Medication', fontsize=15)
ax.set_xlabel('Readmission', fontsize=13)
ax.set_ylabel('Readmission Rates (%)', fontsize=13)

for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.0f}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='center',
                xytext=(0, 6), fontsize=12,
                textcoords='offset points')

sns.despine(left=True)

# Plot Readmission vs No Change Medication
data_no = table_7[table_7['change'] == 'no']
ax = sns.barplot(x='readmitted', y='readmission rate (%)', data=data_no, ax=axes[2, 1])
ax.set_title('Readmission Rate by Patients \n Without Change Diabetes Medication', fontsize=15)
ax.set_xlabel('Readmission', fontsize=13)
ax.set_ylabel('')

for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.0f}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='center',
                xytext=(0, 6), fontsize=12,
                textcoords='offset points')

sns.despine(left=True)

# Plot readmission rate by other diagnosis
ax = sns.barplot(x='diseases',
                 y='readmission rate (%)',
                 hue='readmitted',
                 data=other_readmission,
                 ax=axes[3, 0])

for bar in ax.patches:
    ax.annotate(f'{bar.get_height():.0f}%',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='center',
                xytext=(2, 6), fontsize=12,
                textcoords='offset points')

ax.set_title('Readmission rate by other diagnosis', fontsize=15)
ax.set_ylabel('Readmission Rate (%)', fontsize=13)
ax.set_xlabel('')
ax.legend(title='Readmitted', loc='upper right', bbox_to_anchor=(1.15, 1))
sns.despine(left=True)

# Plot Readmission vs Time in hospital
ax = sns.barplot(x='time_in_hospital',
                 y='readmission rate (%)',
                 hue='readmitted',
                 data=table_8,
                 ax=axes[3, 1]
                 )
ax.axvspan(xmin=5.53,
           xmax=9.53,
           ymax=0.85,
           facecolor='mediumaquamarine',
           edgecolor='mediumaquamarine',
           alpha=0.4,
           lw=1.5,
           zorder=-1
           )
ax.axvspan(xmin=10.53,
           xmax=11.50,
           ymax=0.85,
           facecolor='mediumaquamarine',
           edgecolor='mediumaquamarine',
           alpha=0.4,
           lw=1.5,
           zorder=-1
           )

ax.set_ylabel('')
ax.set_xlabel('Days in Hospital', fontsize=13)
ax.set_title('Readmission Rates by Days in Hospital', fontsize=15)
ax.legend().set_visible(False)
sns.despine(left=True)

# correlation heatmap
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap='Blues', mask=mask, ax=axes[4, 0])

plt.subplots_adjust(hspace=0.4, wspace=0.18)
plt.title('Correlation heatmap', fontsize=15)

plt.show()

# Maps the values in the 'readmitted' column
df['readmitted'] = df['readmitted'].map({'yes': 1, 'no': 0})

# Finding unique age group, readmission reate and number of patients by age group
mean = df.groupby('age')['readmitted'].mean().mul(100).round(2)
patients = df.groupby('age').size()
age_group = df.age.sort_values().unique()

# Create a new df to plot
hp_by_age = pd.DataFrame({
    'age group': age_group,
    'num patients': patients,
    'readmission rate (%)': mean})

# Compute the weight of each age group based on the number of patients.
hp_by_age['weights'] = hp_by_age['num patients'] / hp_by_age['num patients'].sum()

# Compute the weighted average for each age group
hp_by_age['weighted readmission rate (%)'] = round(hp_by_age['readmission rate (%)'] * hp_by_age['weights'], 2)

# Table weighted readmission rate by age group
hp_by_age

# Compute average weighted readmission rate to compare
average_readmission = hp_by_age['weighted readmission rate (%)'].mean().round(2)

# Plot Mean readmission rate by age group
print('Figure 3: Weighted Readmission rate by Age Group.')
fig = plt.figure(figsize=(8, 6))
sns.set_style('white')
sns.set_palette('Paired')

sns.lineplot(x='age group', y='weighted readmission rate (%)', data=hp_by_age)

plt.axhline(y=average_readmission, color='mediumaquamarine', linestyle='--')

plt.text(6.5, average_readmission,
         f'Average Readmission Rate: {average_readmission}%',
         ha='center', va='center',
         color='mediumaquamarine', fontsize=13
         )

plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Weighted Readmission Rate (%)', fontsize=12)
sns.despine()

plt.show()

# Table readmission rate by diagnosis
hp_by_diag = (df[df.diag_1 != "Missing"]
              .groupby('diag_1')['readmitted']
              .mean()
              .mul(100)
              .round()
              .reset_index(name='readmission rate (%)')
              )

hp_by_diag

# Find the readmission_rate mean
tot_mean = round((df.readmitted.mean() * 100))

# Plot readmission rate by diagnosis
fig = plt.figure(figsize=(7, 6))
sns.set_style('white')

sns.barplot(
    y='readmission rate (%)',
    x='diag_1',
    palette='Blues_r',
    data=hp_by_diag.sort_values('readmission rate (%)', ascending=False)
)

plt.axhline(y=tot_mean, color='mediumaquamarine', linestyle='--')

plt.text(
    8, tot_mean,
    f'Average Readmission Rate: {tot_mean}%',
    ha='center', va='center', color='mediumaquamarine'
)

plt.title('Readmission Rate by Diagnosis')
plt.ylabel('Mean Readmission Rate (%)')
plt.xlabel('')
plt.xticks(rotation=90)
sns.despine(left=True)

plt.show()
