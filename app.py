import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from prophet import Prophet

def load_and_preprocess_data(file_path=None, uploaded_file=None):
    """Loads, cleans, and preprocesses the TB outcomes data."""
    if uploaded_file:
        df_new = pd.read_csv(uploaded_file)
    elif file_path:
        df_new = pd.read_csv(file_path)
    else:
        raise ValueError("Either file_path or uploaded_file must be provided.")

    # Handle missing iso2 values
    df_new['iso2'] = df_new['iso2'].fillna('Unknown')

    # Fill NaN values in numerical columns with 0
    numerical_cols_with_nan = df_new.select_dtypes(include=['float64']).columns
    for col in numerical_cols_with_nan:
        df_new[col] = df_new[col].fillna(0)

    # Create 'date' column
    df_new['date'] = pd.to_datetime(df_new['year'], format='%Y')

    # Calculate epidemiological indicators
    df_new['Treatment Success Rate (new_sp)'] = ((df_new['new_sp_cur'] + df_new['new_sp_cmplt']) / df_new['new_sp_coh'] * 100).fillna(0)
    df_new.loc[df_new['new_sp_coh'] == 0, 'Treatment Success Rate (new_sp)'] = 0

    df_new['Mortality Rate (new_sp)'] = (df_new['new_sp_died'] / df_new['new_sp_coh'] * 100).fillna(0)
    df_new.loc[df_new['new_sp_coh'] == 0, 'Mortality Rate (new_sp)'] = 0

    df_new['Treatment Failure Rate (new_sp)'] = (df_new['new_sp_fail'] / df_new['new_sp_coh'] * 100).fillna(0)
    df_new.loc[df_new['new_sp_coh'] == 0, 'Treatment Failure Rate (new_sp)'] = 0

    # For early warning system
    df_new['rolling_mean_3yr'] = df_new.groupby('country')['new_sp_coh'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df_new['rolling_std_3yr'] = df_new.groupby('country')['new_sp_coh'].rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True)
    df_new['alert'] = (df_new['new_sp_coh'] > (df_new['rolling_mean_3yr'] + 2 * df_new['rolling_std_3yr']))

    return df_new

# Initialize session state for the DataFrame if not already present
if 'df' not in st.session_state:
    st.session_state['df'] = load_and_preprocess_data(file_path='/content/TB_outcomes_2025-12-13.csv')

df = st.session_state['df']

# Re-calculate derived dataframes using the current df in session state
total_tb_cases_per_year = df.groupby('year')['new_sp_coh'].sum().reset_index()
total_tb_cases_per_year.columns = ['year', 'new_sp_coh']

average_treatment_success_rate_per_year = df.groupby('year')['Treatment Success Rate (new_sp)'].mean().reset_index()
average_treatment_success_rate_per_year.columns = ['year', 'Average Treatment Success Rate (new_sp)']

alert_countries = df[df['alert']][['country', 'year', 'new_sp_coh', 'rolling_mean_3yr', 'rolling_std_3yr', 'alert']]

# For forecasting
prophet_df = pd.DataFrame({
    'ds': pd.to_datetime(total_tb_cases_per_year['year'], format='%Y'),
    'y': total_tb_cases_per_year['new_sp_coh']
})

# For global risk map
tb_burden_by_country = df.groupby(['country', 'iso3'])['new_sp_coh'].sum().reset_index()
tb_burden_by_country.columns = ['country', 'iso3', 'total_new_sp_coh']

# For research insights
mdr_tb_by_country = df.groupby('country')['mdr_coh'].sum().sort_values(ascending=False)
mortality_rate_by_country = df.groupby('country')['Mortality Rate (new_sp)'].mean().sort_values(ascending=False)

st.set_page_config(layout="wide")

st.title('Tuberculosis Surveillance Dashboard')

# Sidebar for navigation
st.sidebar.title('Navigation')
page_selection = st.sidebar.radio(
    'Go to',
    [
        'Global Dashboard',
        'Country Analysis',
        'Early Warning System',
        'Forecasting',
        'Global Risk Map',
        'Research Insights',
        'Data Management',
        'Automated Report Generation'
    ]
)

# Main content area based on selection
if page_selection == 'Global Dashboard':
    st.header('Global Dashboard - Aggregated Trends')
    st.write('### Total New TB Cases Per Year')
    st.line_chart(total_tb_cases_per_year.set_index('year'))

    st.write('### Average Treatment Success Rate Per Year')
    st.line_chart(average_treatment_success_rate_per_year.set_index('year'))

elif page_selection == 'Country Analysis':
    st.header('Country-Level Detailed Analysis')

    unique_countries = df['country'].unique()
    selected_country = 'Afghanistan' # Default selection for demo purposes
    selected_country_analysis = st.selectbox(
        'Select a Country for Detailed Analysis:',
        unique_countries,
        index=list(unique_countries).index(selected_country) if selected_country in unique_countries else 0
    )

    country_analysis_df = df[df['country'] == selected_country_analysis].copy()

    if not country_analysis_df.empty:
        st.write(f'### TB Trends and Epidemiological Indicators in {selected_country_analysis}')
        fig_country, ax_country = plt.subplots(figsize=(12, 6))
        ax_country.plot(country_analysis_df['year'], country_analysis_df['new_sp_coh'], label='New TB Cases (Cohort)', marker='o')
        ax_country.plot(country_analysis_df['year'], country_analysis_df['Treatment Success Rate (new_sp)'], label='Treatment Success Rate (%)', marker='x')
        ax_country.plot(country_analysis_df['year'], country_analysis_df['Mortality Rate (new_sp)'], label='Mortality Rate (%)', marker='s')
        ax_country.plot(country_analysis_df['year'], country_analysis_df['Treatment Failure Rate (new_sp)'], label='Treatment Failure Rate (%)', marker='d')

        ax_country.set_title(f'TB Trends and Epidemiological Indicators in {selected_country_analysis} Over Time')
        ax_country.set_xlabel('Year')
        ax_country.set_ylabel('Value')
        ax_country.legend()
        ax_country.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig_country)
    else:
        st.write('No data available for the selected country.')

elif page_selection == 'Early Warning System':
    st.header('Early Warning System')
    st.write('### Countries with Unusual Increase in TB Cases')
    if not alert_countries.empty:
        st.dataframe(alert_countries)
    else:
        st.write('No unusual increases detected based on current criteria.')

elif page_selection == 'Forecasting':
    st.header('Forecasting Module')
    st.write('### Predicted Future TB Trends')

    m = Prophet(yearly_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=3, freq='YE')
    forecast_prophet = m.predict(future)

    fig1_st = m.plot(forecast_prophet)
    plt.title('TB Cases Forecast')
    plt.xlabel('Year')
    plt.ylabel('New TB Cases Cohort')
    st.pyplot(fig1_st)

    fig2_st = m.plot_components(forecast_prophet)
    st.pyplot(fig2_st)

elif page_selection == 'Global Risk Map':
    st.header('Global Risk Map')
    st.write('### Global TB Burden by Country')
    fig_map = px.choropleth(
        tb_burden_by_country,
        locations='iso3',
        color='total_new_sp_coh',
        hover_name='country',
        color_continuous_scale=px.colors.sequential.Plasma,
        title='Global TB Burden by Country'
    )
    st.plotly_chart(fig_map)

elif page_selection == 'Research Insights':
    st.header('Automated Research Insights')
    st.write('### Top 5 Countries with Highest Total MDR-TB Cohort')
    st.dataframe(mdr_tb_by_country.head())

    st.write('### Top 5 Countries with Highest Average Mortality Rate (new_sp)')
    st.dataframe(mortality_rate_by_country.head())

elif page_selection == 'Data Management':
    st.header('Data Management')
    st.write('Upload a new dataset or reset to the default dataset.')

    uploaded_file = st.file_uploader("Upload your TB surveillance CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            st.session_state['df'] = load_and_preprocess_data(uploaded_file=uploaded_file)
            st.success("File successfully uploaded and processed!")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

    if st.button('Reset to Default Data'):
        st.session_state['df'] = load_and_preprocess_data(file_path='/content/TB_outcomes_2025-12-13.csv')
        st.success("Dataset reset to default!")
        st.experimental_rerun()

elif page_selection == 'Automated Report Generation':
    st.header('Automated Report Generation')
    st.write('Generate a downloadable report for a selected country.')

    unique_countries_report = df['country'].unique()
    selected_country_report = st.selectbox(
        'Select a Country for Report Generation:',
        unique_countries_report
    )

    report_df = df[df['country'] == selected_country_report].copy()

    if not report_df.empty:
        report_data = report_df[['year', 'new_sp_coh', 'Treatment Success Rate (new_sp)',
                                   'Mortality Rate (new_sp)', 'Treatment Failure Rate (new_sp)']]
        st.write(f'### Epidemiological Indicators for {selected_country_report}')
        st.dataframe(report_data)

        csv_data = report_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Report (CSV)",
            data=csv_data,
            file_name=f"{selected_country_report}_TB_Report.csv",
            mime="text/csv",
        )
    else:
        st.write('No data available for the selected country to generate a report.')
