import numpy as np
import matplotlib.pyplot as plt

csvpath = '/Users/yuwahlim/Desktop/DataChallenge/b_tag_project/csv_data/'
figpath = '/Users/yuwahlim/Desktop/DataChallenge/b_tag_project/figure/'

tpr = np.genfromtxt(csvpath + 'tpr.csv', delimiter=',')
fpr = np.genfromtxt(csvpath + 'fpr.csv', delimiter=',')

AUC = np.trapz(tpr, x=fpr)
st_AUC = "AUC = " + str('%.2f' % round(AUC, 2))

pltRoc = plt.plot(fpr, tpr, linewidth=2, drawstyle='steps-post', color='blue')
pltDiag = plt.plot([0,1],[0,1], 'r--')
plt.text(0.7,0.1, st_AUC, fontsize=17, weight=550)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('false positive rate (fpr)', fontsize=15)
plt.ylabel('true positive rate (tpr)', fontsize=15)
plt.title('GRU Performance - Jet Level Variables Only', fontsize=19)
plt.savefig(figpath + 'test_ML_ROC.png')
