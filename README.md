# Sales-Forecasting-and-Optimization-for-Retail-and-E-commerce

The “Sales Forecasting and Optimization” project aims to leverage historical sales data to build a predictive model for retail and e-commerce businesses. The project will focus on forecasting future sales based on trends, seasonality, promotions, holidays, and other influential factors.

One of the leading retail stores in the US, **Walmart**, would like to predict sales and demand accurately. There are certain events and holidays which impact sales on each day. Sales data are available for 45 Walmart stores.

The business is facing a challenge due to unforeseen demands and stockouts, often caused by inappropriate machine learning algorithms. An ideal ML algorithm will predict demand accurately and consider economic conditions such as the **Consumer Price Index (CPI)** and **Unemployment Index**.

Walmart runs several **promotional markdown events** throughout the year, especially before major holidays such as:

- Super Bowl  
- Labour Day  
- Thanksgiving  
- Christmas  

Weeks that include these holidays are **weighted five times higher** in evaluation than non-holiday weeks. A major challenge in this task is modeling the effect of markdowns on these holidays, especially when complete historical data is unavailable.

---

## Dataset Information

This dataset contains historical sales data from **2010-02-05 to 2012-11-01**. The data is stored in the file `Walmart_Store_sales.csv` and includes the following fields:

- `Store` — Store number  
- `Date` — The week of sales  
- `Weekly_Sales` — Sales for the given store  
- `Holiday_Flag` — Indicates whether the week is a special holiday week (`1` = Holiday, `0` = Non-holiday)  
- `Temperature` — Temperature on the day of sale  
- `Fuel_Price` — Cost of fuel in the region  
- `CPI` — Consumer Price Index  
- `Unemployment` — Unemployment rate  

### Holiday Events

- **Super Bowl**:  
  - 12-Feb-10  
  - 11-Feb-11  
  - 10-Feb-12  
  - 8-Feb-13

- **Labour Day**:  
  - 10-Sep-10  
  - 9-Sep-11  
  - 7-Sep-12  
  - 6-Sep-13

- **Thanksgiving**:  
  - 26-Nov-10  
  - 25-Nov-11  
  - 23-Nov-12  
  - 29-Nov-13

- **Christmas**:  
  - 31-Dec-10  
  - 30-Dec-11  
  - 28-Dec-12  
  - 27-Dec-13

---
## Files:

Sales_Forecasting.ipynb — Notebook code file

app.py — Streamlit interface for the prediction

requirements.txt — Dependencies

## Acknowledgements

The dataset is taken from [Kaggle](https://www.kaggle.com/).
