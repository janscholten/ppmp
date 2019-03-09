import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn')
problem = 'lunarlive'

A = pd.read_csv('ref.csv').append(pd.read_csv('file.csv', delim_whitespace=True))
plt.figure(figsize=(10,7)); 
plt.title('Performance');
plt.cla(); 
plt.subplot(211)
plt.title('Feedback and Gain');
sns.lineplot(x='Episode', y='Reward', data=A, hue='Control')
lims = plt.xlim()
plt.subplot(212)
# sns.lineplot(x='Episode', y='Gain', data=A.loc[A.Control=='kalman'])
# plt.xlim(lims)
# plt.subplot(313)
sns.lineplot(x='Episode', y='Feedback', data=A.loc[A.Control=='kalman'], hue='Control')
plt.xlim(lims)
plt.tight_layout()

plt.savefig(problem+'.pdf')