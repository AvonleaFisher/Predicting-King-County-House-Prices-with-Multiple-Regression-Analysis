# For Bigger Bucks, Build a Bigger House
## Predicting King County House Prices with Multiple Regression  Analysis
<b>Author:</b> Avonlea Fisher
### Problem:
The aim of this analysis is to build a multiple regression model that can predict house prices with the greatest accuracy possible. The results can inform home owners interested in selling their homes about the most important factors to consider for improving sale prices.
### Data
The King County Housing Data Set contains information about the size, location, condition, and other features of houses in Washington's King County. The dataset and variable descriptions can be found on <a href ="https://www.kaggle.com/harlfoxem/housesalesprediction">Kaggle</a>.
## Methods
After exploring and preprocessing the data, simple and multiple linear regression models were built in OLS statsmodels, with price as the dependent variable. Sqft_living, grade, bathrooms and sqft_living15 were chosen for the first model with a .80 correlation cutoff for multicollinearity. The second model included only sqft_living and grade with a .75 correlation cutoff. 
## Results
Both models satisfied all multiple regression assumptions and p-values for each predictor variable were below .05. The r-squared value of the first model (.53) was higher than that of the second model (.51). The first model was chosen for final validation because of its greater predictive power.
## Recommendations 
Homeowners interested in selling their homes should focus on improving the design and quality of construction of their homes, which may in turn improve their home grade. If possible, they should also expand the square footage of living space on the lot, perhaps by building additional bathrooms. The square footage of neighbors' living space is also a strong positive predictor of price, but homeowners likely have less control over this factor. Homeowners may further increase the sale price of their homes by encouraging neighbors to also expand the square footage of their living space. 
## Limitations and Next Steps
Given that some of the variables needed to be log-transformed to satisfy regression assumptions, any new data used with the model would have to undergo similar preprocessing. Additionally, given regional differences in housing prices, the model's applicability to data from other counties may be limited. Further analysis is needed to develop models that are applicable to other regions.
### For further information
<a href ="https://info.kingcounty.gov/assessor/esales/Glossary.aspx?type=r">King County Residential Glossary of Terms</a>
