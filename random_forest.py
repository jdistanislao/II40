#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""random_Forest.py: Random Forest script."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":

        valid_cols = [
                'EmpID',
                'MarriedID',
                'MaritalStatusID',
                'GenderID',
                'EmpStatusID',
                'DeptID',
                'PerfScoreID',
                'FromDiversityJobFairID',
                'Salary',
                'PositionID',
                # # 'DOB',
                # 'Sex',
                # # 'DateofHire',
                # # 'DateofTermination',
                # # 'TermReason',
                # 'EmploymentStatus',
                # 'PerformanceScore',
                'EmpSatisfaction',
                'SpecialProjectsCount',
                # # 'LastPerformanceReview_Date',
                'DaysLateLast30',
                'Absences',
                'Termd'
        ]

        data = pd.read_csv('./data/hr_data.csv', usecols=valid_cols)
        print(data)

        x= data.iloc [:, : -1]
        y= data.iloc [:, -1 :]

        # create regressor object
        regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

        # fit the regressor with x and y data
        regressor.fit(x, y)

        qqq = [6.5,6.5,6.5,6.5,6.5,6.5,6.5,6.5,6.5,6.5,6.5,6.5,6.5,6.5]
        Y_pred = regressor.predict(np.array(qqq).reshape(1, len(valid_cols)-1))  # test the output by changing values
        X_grid = np.arange(min(x), max(x), 0.01) # <---problem here!!!

        # reshape for reshaping the data into a len(X_grid)*1 array,
        # i.e. to make a column out of the X_grid value
        X_grid = X_grid.reshape((len(X_grid), 1))

        # Scatter plot for original data
        plt.scatter(x, y, color = 'blue')

        # plot predicted data
        plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
        plt.title('Random Forest Regression')
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        plt.show()