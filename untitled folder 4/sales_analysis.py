import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from datetime import datetime
import calendar

def load_sales_data(file_path):
    try:
        df = pd.read_excel('ELB-Sales-Data.xlsx')
        # Add Month column based on Date
        df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%B')
        # Rename columns to match our needs
        df = df.rename(columns={
            'Primary Sales': 'Primary_Sales',
            'Sales Return': 'Returns',
            'Claim Offer': 'Claims',
            'Rate Difference': 'Rate_Difference',
            'Item Name': 'Product',
            'Sales Team': 'Sales_Team',
            'Return for Reason': 'Return_Type'
        })
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def calculate_primary_sales(df):
    """
    Calculate primary sales by considering returns, claims, and rate differences
    """
    try:
        # Primary sales are already in the dataset as 'Primary_Sales'
        # We'll adjust it by subtracting returns, claims, and rate differences
        df['Adjusted_Primary_Sales'] = (df['Primary_Sales'] 
                                      - df['Returns'] 
                                      - df['Claims'] 
                                      - df['Rate_Difference'])
        return df
    except KeyError as e:
        print(f"Column not found: {e}")
        print("Available columns:", df.columns.tolist())
        return None

def analyze_highest_selling_product(df, month):
    """
    Find the highest-selling product for a specific month
    """
    monthly_data = df[df['Month'] == month]
    product_sales = monthly_data.groupby('Product')['Adjusted_Primary_Sales'].sum()
    return product_sales.idxmax(), product_sales.max()

def analyze_team_sales(df, team, month):
    """
    Analyze sales for a specific team in a given month
    """
    team_data = df[(df['Sales_Team'] == team) & (df['Month'] == month)]
    product_sales = team_data.groupby('Product')['Adjusted_Primary_Sales'].sum()
    return product_sales.idxmax(), product_sales.max()

def analyze_returns(df, hq, month):
    """
    Analyze returns for a specific HQ in a given month
    """
    hq_data = df[(df['HQ'] == hq) & (df['Month'] == month)]
    if hq_data.empty:
        print(f"No data found for {hq} HQ in {month}")
        return None, 0
    customer_returns = hq_data.groupby('Customer')['Returns'].sum()
    if customer_returns.empty:
        return None, 0
    return customer_returns.idxmax(), customer_returns.max()

def calculate_return_percentages(df):
    """
    Calculate return percentages by type (expiry/breakage)
    """
    total_sales = df['Primary_Sales'].sum()
    expiry_returns = df[df['Return_Type'] == 'Expiry']['Returns'].sum()
    breakage_returns = df['Breakage'].sum()
    
    expiry_percentage = (expiry_returns / total_sales) * 100
    breakage_percentage = (breakage_returns / total_sales) * 100
    
    return expiry_percentage, breakage_percentage

def forecast_sales(df):
    """
    Forecast sales using Prophet
    """
    # Prepare data for Prophet
    forecast_df = df.groupby('Date')['Adjusted_Primary_Sales'].sum().reset_index()
    forecast_df.columns = ['ds', 'y']
    
    # Create and fit Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(forecast_df)
    
    # Create future dates dataframe
    future_dates = model.make_future_dataframe(periods=60, freq='D')
    
    # Make predictions
    forecast = model.predict(future_dates)
    return forecast

def create_dashboard(df, forecast):
    """
    Create visualization dashboard
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Monthly Sales Trend
    plt.subplot(2, 2, 1)
    monthly_sales = df.groupby('Month')['Adjusted_Primary_Sales'].sum()
    monthly_sales.plot(kind='line', marker='o')
    plt.title('Monthly Sales Trend')
    plt.xticks(rotation=45)
    
    # Plot 2: Top Products by Sales
    plt.subplot(2, 2, 2)
    product_sales = df.groupby('Product')['Adjusted_Primary_Sales'].sum().sort_values(ascending=False)[:10]
    product_sales.plot(kind='bar')
    plt.title('Top 10 Products by Sales')
    plt.xticks(rotation=45)
    
    # Plot 3: Returns Analysis
    plt.subplot(2, 2, 3)
    returns_by_type = pd.Series({
        'Expiry': df[df['Return_Type'] == 'Expiry']['Returns'].sum(),
        'Breakage': df['Breakage'].sum()
    })
    returns_by_type.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Returns by Type')
    
    # Plot 4: Sales Forecast
    plt.subplot(2, 2, 4)
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3)
    plt.title('Sales Forecast')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('sales_dashboard.png')
    plt.close()

def analyze_team_returns_percentage(df):
    """
    Calculate the percentage of primary sales returned due to expiry for each sales team
    """
    # Group by sales team
    team_analysis = df.groupby('Sales_Team').agg({
        'Primary_Sales': 'sum',
        'Returns': lambda x: x[df['Return_Type'] == 'Expiry'].sum()
    }).reset_index()
    
    # Calculate return percentage
    team_analysis['Return_Percentage'] = (team_analysis['Returns'] / team_analysis['Primary_Sales']) * 100
    
    # Sort by percentage in descending order
    team_analysis = team_analysis.sort_values('Return_Percentage', ascending=False)
    
    return team_analysis

def analyze_breakage_percentage(df):
    """
    Calculate the percentage of primary sales affected by breakage
    """
    try:
        total_primary_sales = df['Primary_Sales'].sum()
        total_breakage = df['Breakage'].sum()
        
        breakage_percentage = (total_breakage / total_primary_sales) * 100
        
        print("\nBreakage Analysis:")
        print(f"Total Primary Sales: {total_primary_sales:,.2f}")
        print(f"Total Breakage: {total_breakage:,.2f}")
        print(f"Percentage of sales affected by breakage: {breakage_percentage:.2f}%")
        
        return breakage_percentage
        
    except Exception as e:
        print(f"Error calculating breakage percentage: {str(e)}")
        return None

def analyze_delhi_september_sales(df):
    """
    Calculate primary sales for Delhi HQ in September
    """
    try:
        # Filter data for Delhi HQ and September
        delhi_sept_data = df[
            (df['HQ'] == 'Delhi') & 
            (df['Month'] == 'September')
        ]
        
        # Calculate total primary sales
        total_sales = delhi_sept_data['Primary_Sales'].sum()
        
        print("\nDelhi HQ September Sales Analysis:")
        print(f"Total Primary Sales: {total_sales:,.2f}")
        
        # Optional: Show breakdown by product
        print("\nTop 5 Products by Sales:")
        product_sales = delhi_sept_data.groupby('Product')['Primary_Sales'].sum().sort_values(ascending=False).head()
        for product, sales in product_sales.items():
            print(f"{product}: {sales:,.2f}")
            
        return total_sales
        
    except Exception as e:
        print(f"Error analyzing Delhi September sales: {str(e)}")
        return None

def forecast_october_sales(df):
    """
    Forecast sales specifically for October using Prophet
    """
    try:
        # Prepare data for Prophet
        forecast_df = df.groupby('Date')['Primary_Sales'].sum().reset_index()
        forecast_df.columns = ['ds', 'y']
        
        # Create and fit Prophet model
        model = Prophet(yearly_seasonality=True, 
                       weekly_seasonality=True,
                       daily_seasonality=False)
        model.fit(forecast_df)
        
        # Create future dates dataframe (just for October)
        future_dates = model.make_future_dataframe(periods=31, freq='D')
        
        # Make predictions
        forecast = model.predict(future_dates)
        
        # Get October forecast
        october_forecast = forecast[forecast['ds'].dt.month == 10]['yhat'].mean()
        
        print("\nOctober Sales Forecast Analysis:")
        print(f"Forecasted Primary Sales for October: {october_forecast:,.2f}")
        
        # Optional: Show confidence intervals
        october_lower = forecast[forecast['ds'].dt.month == 10]['yhat_lower'].mean()
        october_upper = forecast[forecast['ds'].dt.month == 10]['yhat_upper'].mean()
        print(f"Confidence Interval: ({october_lower:,.2f} - {october_upper:,.2f})")
        
        # Create forecast visualization
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['ds'], forecast_df['y'], label='Historical')
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
        plt.fill_between(forecast['ds'], 
                        forecast['yhat_lower'], 
                        forecast['yhat_upper'], 
                        alpha=0.3)
        plt.title('Sales Forecast with October Prediction')
        plt.xlabel('Date')
        plt.ylabel('Primary Sales')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('october_forecast.png')
        plt.close()
        
        return october_forecast
        
    except Exception as e:
        print(f"Error forecasting October sales: {str(e)}")
        return None

def forecast_team_product_sales(df, team_name):
    """
    Forecast sales by product for a specific sales team
    """
    try:
        # Filter data for the specific team
        team_data = df[df['Sales_Team'] == team_name]
        
        # Get unique products for this team
        products = team_data['Product'].unique()
        
        # Store forecasts for each product
        product_forecasts = {}
        
        for product in products:
            # Get data for this product
            product_data = team_data[team_data['Product'] == product]
            
            # Only forecast if we have enough data points
            if len(product_data) >= 30:  # Minimum data points for reliable forecast
                # Prepare data for Prophet
                forecast_df = product_data.groupby('Date')['Primary_Sales'].sum().reset_index()
                forecast_df.columns = ['ds', 'y']
                
                # Create and fit Prophet model
                model = Prophet(yearly_seasonality=True,
                              weekly_seasonality=True,
                              daily_seasonality=False)
                model.fit(forecast_df)
                
                # Create future dates dataframe
                future_dates = model.make_future_dataframe(periods=60, freq='D')
                
                # Make predictions
                forecast = model.predict(future_dates)
                
                # Get November forecast
                november_forecast = forecast[forecast['ds'].dt.month == 11]['yhat'].mean()
                product_forecasts[product] = november_forecast
        
        # Sort products by forecasted sales
        sorted_products = sorted(product_forecasts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nNovember Sales Forecast for {team_name}:")
        print("\nTop 5 Products by Forecasted Sales:")
        for product, forecast_value in sorted_products[:5]:
            print(f"{product}: {forecast_value:,.2f}")
        
        # Return the highest forecasted product
        if sorted_products:
            return sorted_products[0]
        return None
        
    except Exception as e:
        print(f"Error forecasting team product sales: {str(e)}")
        return None

def forecast_specific_product_hq(df, product_name, hq_name, target_month):
    """
    Forecast sales for a specific product in a specific HQ for a target month
    """
    try:
        # Filter data for the specific product and HQ
        filtered_data = df[
            (df['Product'] == product_name) & 
            (df['HQ'] == hq_name)
        ]
        
        if filtered_data.empty:
            print(f"No historical data found for {product_name} in {hq_name} HQ")
            return None
            
        # Prepare data for Prophet
        forecast_df = filtered_data.groupby('Date')['Primary_Sales'].sum().reset_index()
        forecast_df.columns = ['ds', 'y']
        
        # Create and fit Prophet model
        model = Prophet(yearly_seasonality=True,
                       weekly_seasonality=True,
                       daily_seasonality=False)
        model.fit(forecast_df)
        
        # Create future dates dataframe
        future_dates = model.make_future_dataframe(periods=60, freq='D')
        
        # Make predictions
        forecast = model.predict(future_dates)
        
        # Get target month forecast
        target_forecast = forecast[forecast['ds'].dt.month == target_month]['yhat'].mean()
        
        print(f"\nForecast Analysis for {product_name} in {hq_name} HQ:")
        print(f"Forecasted Sales for {calendar.month_name[target_month]}: {target_forecast:,.2f}")
        
        # Show confidence intervals
        target_lower = forecast[forecast['ds'].dt.month == target_month]['yhat_lower'].mean()
        target_upper = forecast[forecast['ds'].dt.month == target_month]['yhat_upper'].mean()
        print(f"Confidence Interval: ({target_lower:,.2f} - {target_upper:,.2f})")
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['ds'], forecast_df['y'], label='Historical')
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
        plt.fill_between(forecast['ds'], 
                        forecast['yhat_lower'], 
                        forecast['yhat_upper'], 
                        alpha=0.3)
        plt.title(f'{product_name} Sales Forecast for {hq_name} HQ')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{product_name}_{hq_name}_forecast.png')
        plt.close()
        
        return target_forecast
        
    except Exception as e:
        print(f"Error forecasting product sales: {str(e)}")
        return None

def main():
    # Load data
    df = load_sales_data('ELB-Sales-Data.xlsx')
    
    if df is None:
        print("Failed to load data. Exiting...")
        return
        
    df = calculate_primary_sales(df)
    
    if df is None:
        print("Failed to calculate primary sales. Exiting...")
        return
    
    # Answer Question 1: Highest selling product in September
    product, sales = analyze_highest_selling_product(df, 'September')
    print(f"Highest selling product in September: {product} (Sales: {sales:,.2f})")
    
    # Answer Question 2: Highest sales for CND Chennai in May
    product, sales = analyze_team_sales(df, 'CND Chennai', 'May')
    print(f"Highest selling product for CND Chennai in May: {product} (Sales: {sales:,.2f})")
    
    # Answer Question 3: Maximum stock returns in Bangalore HQ
    # Let's check available months for Bangalore HQ
    bangalore_months = df[df['HQ'] == 'Bangalore']['Month'].unique()
    print(f"\nAvailable months for Bangalore HQ: {sorted(bangalore_months)}")
    
    # Try each month until we find data
    for month in ['October', 'September', 'August', 'July', 'June', 'May', 'April']:
        customer, returns = analyze_returns(df, 'Bangalore', month)
        if customer is not None:
            print(f"Customer with maximum returns in Bangalore ({month}): {customer} (Returns: {returns:,.2f})")
            break
    
    # Answer Question 4 & 5: Return percentages
    expiry_pct, breakage_pct = calculate_return_percentages(df)
    print(f"Percentage of sales returned due to expiry: {expiry_pct:.2f}%")
    print(f"Percentage of sales affected by breakage: {breakage_pct:.2f}%")
    
    # Generate forecasts
    try:
        forecast = forecast_sales(df)
        # Create dashboard
        create_dashboard(df, forecast)
    except Exception as e:
        print(f"Error in forecasting: {str(e)}")
        forecast = None
    
    # Question 6: Primary sales for Delhi HQ in September
    delhi_sept_sales = df[
        (df['HQ'] == 'Delhi') & 
        (df['Month'] == 'September')
    ]['Adjusted_Primary_Sales'].sum()
    print(f"Delhi HQ September sales: {delhi_sept_sales:,.2f}")
    
    # Question 7: Britorva 20 sales for specific customer
    britorva_sales = df[
        (df['Product'] == 'Britorva 20') & 
        (df['Month'] == 'September') &
        (df['Customer'] == 'PALEPU PHARMA DIST PVT LTD') &
        (df['HQ'] == 'Coimbatore')
    ]['Adjusted_Primary_Sales'].sum()
    print(f"Britorva 20 sales for PALEPU PHARMA: {britorva_sales:,.2f}")
    
    # Forecast for October
    if forecast is not None:
        try:
            october_forecast = forecast[forecast['ds'] == '2023-10-01']['yhat'].values[0]
            print(f"Forecasted October sales: {october_forecast:,.2f}")
        except Exception as e:
            print(f"Error getting October forecast: {str(e)}")
    
    # Sales Team Return Analysis
    print("\nSales Team Return Analysis:")
    team_returns = analyze_team_returns_percentage(df)
    print("\nTop 5 Sales Teams by Return Percentage:")
    print(team_returns[['Sales_Team', 'Return_Percentage']].head().to_string())
    
    # Get the team with maximum returns
    max_return_team = team_returns.iloc[0]
    print(f"\nSales team with maximum returns due to expiry: {max_return_team['Sales_Team']}")
    print(f"Return percentage: {max_return_team['Return_Percentage']:.2f}%")
    
    # Analyze breakage percentage
    print("\nAnalyzing Breakage Impact:")
    breakage_pct = analyze_breakage_percentage(df)
    
    # Analyzing Delhi September Sales
    print("\nAnalyzing Delhi September Sales:")
    delhi_sept_total = analyze_delhi_september_sales(df)
    
    # Calculating October Sales Forecast
    print("\nCalculating October Sales Forecast:")
    october_forecast = forecast_october_sales(df)
    
    # Forecasting CND Chennai Product Sales for November
    print("\nForecasting CND Chennai Product Sales for November:")
    top_product = forecast_team_product_sales(df, "CND Chennai")
    if top_product:
        product, forecast_value = top_product
        print(f"\nProduct with highest forecasted sales for CND Chennai in November:")
        print(f"{product}: {forecast_value:,.2f}")
    
    # Forecasting Britorva 20 Sales for Coimbatore HQ in October
    print("\nForecasting Britorva 20 Sales for Coimbatore HQ in October:")
    britorva_forecast = forecast_specific_product_hq(df, "Britorva 20", "Coimbatore", 10)  # 10 for October
    if britorva_forecast is not None:
        print(f"Forecasted Britorva 20 sales in Coimbatore for October: {britorva_forecast:,.2f}")

if __name__ == "__main__":
    main() 