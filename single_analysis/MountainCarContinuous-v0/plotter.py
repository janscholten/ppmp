import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn')

ref = pd.read_csv('ref.csv', delim_whitespace=True).rename({'Control':'Algorithm'}, axis='columns', inplace=True)
A = pd.read_csv('file.csv', delim_whitespace=True).\
	append(ref)
plt.figure(figsize=(11, 8)); 
plt.title('Performance');
plt.cla(); 
plt.subplot(311)
plt.title('Feedback and Gain');
sns.lineplot(x='Episode', y='Reward', data=A, hue='Algorithm')
# lims = plt.xlim((0,200))
plt.ylim((-100,100))
plt.subplot(312)
sns.lineplot(x='Episode', y='Gain', data=A, hue='Algorithm')
# plt.xlim(lims)
plt.subplot(313)
sns.lineplot(x='Episode', y='Feedback', data=A, hue='Algorithm')
# plt.xlim(lims)
plt.tight_layout()
plt.savefig('live_view.pdf')