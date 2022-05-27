import pandas as pd
import matplotlib.pyplot as plt
from numpy import array, sign, zeros
import numpy as np
import seaborn as sns

myfile="Cab-o-jet_200nm_sample_B,_10mins.csv"
dfCab200 = pd.read_csv(myfile, sep=',')
myfile="Cab-o-jet_300nm_sample_D,_10mins.csv"
dfCab300 = pd.read_csv(myfile, sep=',')
myfile="Regal_Black_sampleH,_10mins.csv"
dfRB = pd.read_csv(myfile, sep=',')

dfCab200['Cab 200 area equivalent diameter']=dfCab200['areaeqd']
dfCab300['Cab 300 area equivalent diameter']=dfCab300['areaeqd']
dfRB['Regal Black area equivalent diameter']=dfRB['areaeqd']

dfCab200['Cab 200 solidity']=dfCab200['solidity']
dfCab300['Cab 300 solidity']=dfCab300['solidity']
dfRB['Regal solidity']=dfRB['solidity']


#fig, ax = plt.subplots(1,1,figsize=(10,10))
#dfCab200.plot(y='solidity',x='Cab 200 area equivalent diameter',color='blue',label='Cab-o-jet 200nm',style='o',ax=ax)
#dfCab300.plot(y='solidity',x='Cab 300 area equivalent diameter',color='red',label='Cab-o-jet 300nm',style='o',ax=ax)
#dfRB.plot(y='solidity',x='Regal Black area equivalent diameter',color='green',label='Regal black 300nm',style='o',ax=ax)

AreaTotal=[dfCab200['Cab 200 area equivalent diameter'],dfCab300['Cab 300 area equivalent diameter'],dfRB['Regal Black area equivalent diameter']]
fig, ax = plt.subplots(1,1,figsize=(10,10))
#hist=ax.hist(AreaTotal, bins='auto')


array=[dfCab200['Cab 200 area equivalent diameter'],dfCab300['Cab 300 area equivalent diameter'],dfRB['Regal Black area equivalent diameter']]
#sns.histplot(array,  kde=False, element="step")
#df =gapminder[gapminder.continent == 'Americas']
sns.histplot(dfCab300['Cab 300 area equivalent diameter'],  kde=False,label='Cab 300',bins=12,color='blue',multiple='dodge', element="step",ax=ax, alpha  = 0.4)
sns.histplot(dfRB['Regal Black area equivalent diameter'],  kde=False,label='Regal Black',bins=14,color='red',multiple='dodge', element="step",ax=ax, alpha  = 0.4)
sns.histplot(dfCab200['Cab 200 area equivalent diameter'],  kde=False, label='Cab 200',bins=9,color='green',multiple='dodge', element="step",ax=ax, alpha  = 0.4)

ax.grid( which='minor', axis='both')
ax.grid( which='major', axis='both')
ax.set_xlabel("Area equivalent Diameter (nm)", fontsize=22)
ax.set_ylabel("Count", fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=20)
# Plot formatting
ax.set_xlim([100,500])
plt.legend(prop={'size': 22})
plt.title('Area Equivalent diameter', fontsize=22)
#plt.xlabel('Equivalent diameter/DMA selected diameter')
#plt.ylabel('Solidity')
plt.show()


fig, ax = plt.subplots(1,1,figsize=(10,10))
#hist=ax.hist(AreaTotal, bins='auto')


array=[dfCab200['Cab 200 area equivalent diameter'],dfCab300['Cab 300 area equivalent diameter'],dfRB['Regal Black area equivalent diameter']]
#sns.histplot(array,  kde=False, element="step")
#df =gapminder[gapminder.continent == 'Americas']
sns.histplot(dfCab300['Cab 300 solidity'],  kde=False,label='Cab 300',bins=12,color='blue',multiple='dodge', element="step",ax=ax, alpha  = 0.4)
sns.histplot(dfRB['Regal solidity'],  kde=False,label='Regal Black',bins=14,color='red',multiple='dodge', element="step",ax=ax, alpha  = 0.4)
sns.histplot(dfCab200['Cab 200 solidity'],  kde=False, label='Cab 200',bins=9,color='green',multiple='dodge', element="step",ax=ax, alpha  = 0.4)


ax.grid( which='minor', axis='both')
ax.grid( which='major', axis='both')
ax.set_xlabel("Solidity", fontsize=22)
ax.set_ylabel("Count", fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=20)
# Plot formatting
plt.legend(prop={'size': 22})
plt.title('Solidity', fontsize=22)
#plt.xlabel('Equivalent diameter/DMA selected diameter')
plt.show()
