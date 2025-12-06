import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


class RowFilter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df = df[df['EPS'] > 0]
        df = df[df['Book Value'] > 0]
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['PE'] = df['Price'] / df['EPS']
        df['Earnings_Yield'] = 1 / df['PE']
        df['PB'] = df['Price'] / df['Book Value']
        df['BP'] = 1 / df['PB']
        df['ROE'] = df['Net Income'] / df['Shareholder Equity']
        df['Operating_Margin'] = df['Operating Income'] / df['Revenue']
        df['Net_Profit_Margin'] = df['Net Income'] / df['Revenue']
        df['ROA'] = df['Net Income'] / df['Total Assets']
        df['Debt_to_Equity'] = df['Total Liabilities'] / df['Shareholder Equity']
        df['Safety_Score'] = 1 / (1 + df['Debt_to_Equity'])
        df['Shares_Outstanding'] = df['Net Income'] / df['EPS']
        df['Market_Cap'] = df['Price'] * df['Shares_Outstanding']
        df['EV'] = df['Market_Cap'] + df['Total Liabilities']
        df['EV_to_sales'] = df['EV'] / df['Revenue']
        df['FCF_Torevenue'] = df['Free Cash Flow'] / df['Revenue']
        df['Cashflow_pershare'] = df['Free Cash Flow'] / df['Shares_Outstanding']
        df['BP_x_EarningsYield'] = df['BP'] * df['Earnings_Yield']
        return df


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features):
        self.selected_features = selected_features
        self.power_transformer = None
        self.logt = []
        self.power = []

    def fit(self, X, y=None):
        df_transformed = X.copy().replace([np.inf, -np.inf], np.nan).dropna()
        self.logt = [col for col in df_transformed.columns if (df_transformed[col] > 0).all()]
        self.power = [col for col in df_transformed.columns if col not in self.logt]
        self.power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        if self.power:
            self.power_transformer.fit(df_transformed[self.power])
        return self

    def transform(self, X):
        df_transformed = X.copy().replace([np.inf, -np.inf], np.nan)
        for col in self.logt:
            df_transformed[col + '_log'] = np.log1p(df_transformed[col])
        if self.power:
            transformed_power = self.power_transformer.transform(df_transformed[self.power])
            for i, c in enumerate(self.power):
                df_transformed[c + '_power'] = transformed_power[:, i]
        return df_transformed[self.selected_features]
