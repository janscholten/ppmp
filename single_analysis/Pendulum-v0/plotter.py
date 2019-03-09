import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn')

A = pd.read_csv('ref.csv', delim_whitespace=True).append(pd.read_csv('file.csv', delim_whitespace=True))
plt.figure(figsize=(11, 8)); 
plt.title('Performance');
plt.cla(); 
plt.subplot(311)
plt.title('Feedback and Gain');
sns.lineplot(x='Episode', y='Reward', data=A, hue='Control')
plt.xticks(range(0,100, 10))
lims = plt.xlim()
plt.subplot(312)
sns.lineplot(x='Episode', y='Gain', data=A, hue='Control')
plt.xticks(range(0,100, 10))
plt.xlim(lims)
plt.subplot(313)
sns.lineplot(x='Episode', y='Feedback', data=A, hue='Control')
plt.xticks(range(0,100, 10))
# sns.lineplot(x='Episode', y='EFB', data=A, hue='Control')
# sns.lineplot(x='Episode', y='Feedback', data=A, hue='Control').lineplot(x='Episode', y='EFB', data=A, hue='Control')
plt.xlim(lims)
# plt.ylim((0,1.5))
plt.tight_layout()

plt.savefig('live_view.pdf')