```
Devloperd by : Mohamed Aathil M
Register Number : 25008235
Dept :CSE
```
```
QU.NO-1

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FuelConsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission')
plt.show()

OUTPUT:

<img width="897" height="649" alt="Screenshot 2025-12-27 084619" src="https://github.com/user-attachments/assets/4bdc59dc-56e1-492d-9c15-c0d0f61fc721" />
```
