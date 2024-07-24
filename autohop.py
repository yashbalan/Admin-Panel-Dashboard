import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import requests
from PIL import Image
import re
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from folium import plugins
import openpyxl
st.set_page_config(layout="wide", page_title="Hopcharge Dashboard", page_icon=":bar_chart:")

# Function to clean license plates
def clean_license_plate(plate):
    match = re.match(r"([A-Z]+[0-9]+)(_R)$", plate)
    if match:
        return match.group(1)
    return plate




def convert_to_datetime_with_current_year(date_string):
    current_year = datetime.now().year
    date = pd.to_datetime(date_string, errors='coerce')
    return date.replace(year=current_year)

# Function to get data from the API
def fetch_data(url):

    payload = {
        "username": "admin",
        "password": "Hopadmin@2024#"
    }
    headers = {
        'accept': 'application/json',
        'accept-language': 'en-US,en;q=0.9',
        'authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOiI3NDIxZDRmNi0zZWI2LTRhODItOWU0Ny02MWU3MWViOTI5Y2EiLCJlbWFpbCI6ImhlbGxvQGhvcGNoYXJnZS5jb20iLCJwaG9uZU51bWJlciI6Iis5MTkzMTE2NjEyODgsICs5MTkyODkwNDYyOTcsKzkxOTgyMDg1NDAwNiwrOTE4NTg4ODYyNjQ4LCs5MTcwNTM0NTcxMjQiLCJmaXJzdE5hbWUiOiJTdXBlciIsImxhc3ROYW1lIjoiQWRtaW4iLCJjcmVhdGVkIjoiMjAyMS0wNi0wMVQxNzowMTozMC42MjhaIiwidXBkYXRlZCI6IjIwMjQtMDQtMDlUMDU6MTc6NDcuNzQ4WiIsImxhc3RMb2dpbiI6IjIwMjQtMDctMjRUMDU6Mzc6MjMuNTM3WiIsImxhc3RMb2dvdXQiOiIyMDI0LTA0LTA5VDA1OjE3OjQ3Ljc0OVoiLCJ1c2VybmFtZSI6ImFkbWluIiwicm9sZSI6InN1cGVyYWRtaW4iLCJpYXQiOjE3MjE3OTk0NDN9.eg_aP1gUcAJXGX1jNMgFX6CAfeLDSc5JpFFMWVG_ttU'
    }
    response = requests.request("GET", url, headers=headers, data=payload)

    # Print the response text to understand its structure
    #print(response.text)

    # Try to parse the response JSON
    response_json = response.json()
    if 'data' in response_json:
        return pd.json_normalize(response_json['data'])
    else:
        return pd.DataFrame()  # Return an empty DataFrame if 'data' key is not found

# URLs for the APIs
url_bookings = "https://2e855a4f93a0.api.hopcharge.com/admin/api/v1/bookings/past?filter={\"chargedAt_lte\":\"2024-06-01\",\"chargedAt_gte\":\"2024-07-30\"}&range=[0,300000]&sort=[\"created\",\"DESC\"]"
url_drivers = "https://2e855a4f93a0.api.hopcharge.com/admin/api/v1/drivers-shifts/export-list?filter={\"action\":\"exporter\",\"startedAt_lte\":\"2024-06-01\",\"endedAt_gte\":\"2024-07-30\"}"

# Fetch data from the APIs
past_bookings_df = fetch_data(url_bookings)
drivers_shifts_df = fetch_data(url_drivers)

# Save the data to CSV for checking column names
#past_bookings_df.to_csv('data/past_bookings112.csv', index=False)
#drivers_shifts_df.to_csv('data/driver_shifts8.csv', index=False)

# Check if the DataFrame is empty
if past_bookings_df.empty or drivers_shifts_df.empty:
    st.error("No data found in the API response.")
else:
    # Printing the first few rows of the DataFrame for debugging
    print(past_bookings_df.head())
    print(drivers_shifts_df.head())

    #drivers_shifts_df.to_csv('past_bookings902.csv', index=False)

    # Calculate shift hours
    drivers_shifts_df['shiftStartedAt'] = pd.to_datetime(drivers_shifts_df['shiftStartedAt'])
    drivers_shifts_df['shiftEndedAt'] = pd.to_datetime(drivers_shifts_df['shiftEndedAt'])
    drivers_shifts_df['Shift_Hours'] = (drivers_shifts_df['shiftEndedAt'] - drivers_shifts_df[
        'shiftStartedAt']).dt.total_seconds() / 3600

    # For the heatmap, prepare data for V_Mode
    v_mode_drivers_df = drivers_shifts_df[drivers_shifts_df['donorVMode'] == 'TRUE']
    v_mode_drivers_df['Actual OPERATOR NAME'] = drivers_shifts_df['driverFirstName'] + ' ' + drivers_shifts_df[
        'driverLastName']
    #v_mode_drivers_df = v_mode_drivers_df[v_mode_drivers_df['bookingStatus'] == 'completed']
    v_mode_drivers_df['licensePlate'] = v_mode_drivers_df['licensePlate'].apply(clean_license_plate)

    # Process V_Mode Data
    v_mode_final_df = v_mode_drivers_df[
        ['licensePlate', 'shiftUid', 'Actual OPERATOR NAME', 'donorVMode', 'shiftStartedAt', 'shiftEndedAt']]

    v_mode_final_df['Actual Date'] = pd.to_datetime(v_mode_final_df['shiftStartedAt'], errors='coerce')

    # Ensure that there are no NaT values in the Actual Date column
    v_mode_final_df = v_mode_final_df.dropna(subset=['Actual Date'])




    # Removing duplicates based on uid and bookingUid
    #v_mode_final_df = v_mode_final_df.drop_duplicates(subset=['shiftUid', 'Actual Date'])

    #v_mode_final_df.to_csv('past_bookings72882.csv', index=False)

    past_bookings_df['Customer Name'] = past_bookings_df['firstName'] + " " + past_bookings_df['lastName']
    past_bookings_df['optChargeStartTime'] = pd.to_datetime(past_bookings_df['optChargeStartTime'], format='mixed',
                                                            errors='coerce')
    past_bookings_df['optChargeEndTime'] = pd.to_datetime(past_bookings_df['optChargeEndTime'], format='mixed',
                                                          errors='coerce')
    past_bookings_df['Reach Time'] = pd.to_datetime(past_bookings_df['optChargeStartTime'], format='mixed', errors='coerce')
    past_bookings_df.rename(columns={
        'optBatteryBeforeChrg': 'Actual SoC_Start',
        'optBatteryAfterChrg': 'Actual SoC_End'
    }, inplace=True)
    past_bookings_df['Booking Session time'] = pd.to_datetime(past_bookings_df['fromTime'], format='mixed',
                                                    errors='coerce')


    # Combine 'driverFirstName' and 'driverLastName' into 'Actual OPERATOR NAME'
    past_bookings_df['Actual OPERATOR NAME'] = past_bookings_df['driverFirstName'] + ' ' + past_bookings_df['driverLastName']

    # Calculate t-15_kpi
    def calculate_t_minus_15(row):
        booking_time = row['Booking Session time']
        arrival_time = row['Reach Time']

        time_diff = booking_time - arrival_time

        if time_diff >= timedelta(minutes=15):
            return 1
        elif time_diff < timedelta(seconds=0):
            return 2
        else:
            return 0


    # Apply the function to calculate t-15_kpi
    past_bookings_df['t-15_kpi'] = past_bookings_df.apply(calculate_t_minus_15, axis=1)

    if 'cancelledPenalty' not in past_bookings_df.columns:
        past_bookings_df['cancelledPenalty'] = 0
        past_bookings_df.loc[
            (past_bookings_df['canceled'] == True) & ((past_bookings_df['optChargeStartTime'] - past_bookings_df[
                'Reach Time']).dt.total_seconds() / 60 < 15), 'cancelledPenalty'] = 1

    # Filter where donorVMode is False
    filtered_drivers_df = drivers_shifts_df[drivers_shifts_df['donorVMode'] == 'FALSE']
    filtered_drivers_df = filtered_drivers_df.drop_duplicates(subset=['bookingUid'])

    filtered_drivers_df = drivers_shifts_df[drivers_shifts_df['bookingStatus'] == 'completed']

    heatmap_filtered_drivers_df = filtered_drivers_df.drop_duplicates(subset=['bookingUid'])

    heatmap_filtered_drivers_df = drivers_shifts_df[drivers_shifts_df['bookingStatus'] == 'completed']


    # Cleaning license plates
    filtered_drivers_df['licensePlate'] = filtered_drivers_df['licensePlate'].apply(clean_license_plate)

    heatmap_filtered_drivers_df['licensePlate'] = heatmap_filtered_drivers_df['licensePlate'].apply(clean_license_plate)

    #past_bookings_df.to_csv('bookings3.csv', index=False)


    # Extracting Customer Location City by matching bookingUid with uid from past_bookings_df
    merged_df = pd.merge(filtered_drivers_df, past_bookings_df[['uid', 'location.state', 'Customer Name', 'Actual OPERATOR NAME',
                                'optChargeStartTime', 'optChargeEndTime', 'Reach Time', 'Actual SoC_Start',
                                'Actual SoC_End', 'Booking Session time', 'canceled',
                                'cancelledPenalty', 't-15_kpi', 'subscriptionName',
                                'location.lat', 'location.long']],
                         left_on='bookingUid', right_on='uid', how='left')

    heatmap_merged_df = pd.merge(heatmap_filtered_drivers_df,
                         past_bookings_df[['uid', 'location.state', 'Customer Name', 'Actual OPERATOR NAME',
                                           'optChargeStartTime', 'optChargeEndTime', 'Reach Time', 'Actual SoC_Start',
                                           'Actual SoC_End', 'Booking Session time', 'canceled',
                                           'cancelledPenalty', 't-15_kpi', 'subscriptionName',
                                           'location.lat', 'location.long']],
                         left_on='bookingUid', right_on='uid', how='left')


    # Extracting Actual Date from fromTime
    #merged_df['Actual Date'] = pd.to_datetime(merged_df['fromTime'], errors='coerce')

    # Extracting Actual Date from bookingFromTime
    merged_df['Actual Date'] = pd.to_datetime(merged_df['bookingFromTime'], errors='coerce')

    heatmap_merged_df['Actual Date'] = pd.to_datetime(heatmap_merged_df['bookingFromTime'], errors='coerce')

    # Ensure necessary columns are present, and calculate additional columns if needed
    if 'Day' not in merged_df.columns:
        merged_df['Day'] = merged_df['Actual Date'].dt.day_name()

    if 'Day' not in heatmap_merged_df.columns:
        heatmap_merged_df['Day'] = heatmap_merged_df['Actual Date'].dt.day_name()



    # Selecting and renaming the required columns
    final_df = merged_df[['Actual Date', 'licensePlate', 'location.state', 'bookingUid', 'uid', 'bookingFromTime', 'bookingStatus', 'customerUid', 'totalUnitsCharged', 'Customer Name', 'Actual OPERATOR NAME',
                                'optChargeStartTime', 'optChargeEndTime', 'Day', 'Reach Time', 'Actual SoC_Start',
                                'Actual SoC_End', 'Booking Session time', 'canceled',
                                'cancelledPenalty', 't-15_kpi', 'subscriptionName',
                                'location.lat', 'location.long', 'donorVMode']].rename(columns={'location.state': 'Customer Location City', 'totalUnitsCharged':'KWH Pumped Per Session'})

    heatmap_final_df = heatmap_merged_df[
        ['Actual Date', 'licensePlate', 'location.state', 'bookingUid', 'uid', 'shiftUid', 'bookingFromTime', 'bookingStatus', 'shiftStartedAt', 'shiftEndedAt',
         'customerUid', 'totalUnitsCharged', 'Customer Name', 'Actual OPERATOR NAME',
         'optChargeStartTime', 'optChargeEndTime', 'Day', 'Reach Time', 'Actual SoC_Start',
         'Actual SoC_End', 'Booking Session time', 'canceled',
         'cancelledPenalty', 't-15_kpi', 'subscriptionName',
         'location.lat', 'location.long', 'donorVMode']].rename(
        columns={'location.state': 'Customer Location City', 'totalUnitsCharged': 'KWH Pumped Per Session'})

    # Ensure that there are no NaT values in the Actual Date column
    final_df = final_df.dropna(subset=['Actual Date'])
    #final_df['Actual Date'] = pd.to_datetime(final_df['Actual Date']).dt.date

    heatmap_final_df = heatmap_final_df.dropna(subset=['Actual Date'])

    # Removing duplicates based on uid and bookingUid
    final_df = final_df.drop_duplicates(subset=['uid', 'bookingUid', 'Actual Date'])

    heatmap_final_df = heatmap_final_df.drop_duplicates(subset=['uid', 'bookingUid', 'Actual Date'])

    # Drop records where totalUnitsCharged is 0
    final_df = final_df[final_df['KWH Pumped Per Session'] != 0]

    heatmap_final_df = heatmap_final_df[heatmap_final_df['KWH Pumped Per Session'] != 0]


    # Printing the first few rows of the DataFrame for debugging
    #st.write(final_df.head())
    #final_df.to_csv('bookings53.csv', index=False)

    # Reading EPOD data from CSV file
    df1 = pd.read_csv('EPOD NUMBER.csv')

    # Data cleaning and transformation
    final_df['licensePlate'] = final_df['licensePlate'].str.upper()
    final_df['licensePlate'] = final_df['licensePlate'].str.replace('HR55AJ4OO3', 'HR55AJ4003')

    heatmap_final_df['licensePlate'] = heatmap_final_df['licensePlate'].str.upper()
    heatmap_final_df['licensePlate'] = heatmap_final_df['licensePlate'].str.replace('HR55AJ4OO3', 'HR55AJ4003')
    # Replace specific license plates
    replace_dict = {
        'HR551305': 'HR55AJ1305',
        'HR552932': 'HR55AJ2932',
        'HR551216': 'HR55AJ1216',
        'HR555061': 'HR55AN5061',
        'HR554745': 'HR55AR4745',
        'HR55AN1216': 'HR55AJ1216',
        'HR55AN8997': 'HR55AN8997'
    }
    final_df['licensePlate'] = final_df['licensePlate'].replace(replace_dict)
    final_df['Actual Date'] = pd.to_datetime(final_df['Actual Date'], format='mixed', errors='coerce')
    final_df = final_df[final_df['Actual Date'].dt.year > 2021]
    final_df['Actual Date'] = final_df['Actual Date'].dt.date
    final_df['Customer Location City'].replace({'Haryana': 'Gurugram', 'Uttar Pradesh': 'Noida'}, inplace=True)
    cities = ['Gurugram', 'Noida', 'Delhi']
    final_df = final_df[final_df['Customer Location City'].isin(cities)]

    merged_df = pd.merge(final_df, df1, on=["licensePlate"])
    final_df = merged_df

    heatmap_final_df['licensePlate'] = heatmap_final_df['licensePlate'].replace(replace_dict)
    heatmap_final_df['Actual Date'] = pd.to_datetime(heatmap_final_df['Actual Date'], format='mixed', errors='coerce')
    heatmap_final_df = heatmap_final_df[heatmap_final_df['Actual Date'].dt.year > 2021]
    heatmap_final_df['Actual Date'] = heatmap_final_df['Actual Date'].dt.date
    heatmap_final_df['Customer Location City'].replace({'Haryana': 'Gurugram', 'Uttar Pradesh': 'Noida'}, inplace=True)
    cities = ['Gurugram', 'Noida', 'Delhi']
    heatmap_final_df = heatmap_final_df[heatmap_final_df['Customer Location City'].isin(cities)]

    heatmap_merged_df = pd.merge(heatmap_final_df, df1, on=["licensePlate"])
    heatmap_final_df = heatmap_merged_df
    #final_df.to_csv('bookings89000.csv', index=False)

    shift_data_df = heatmap_final_df.copy()
    shift_data_df = shift_data_df.drop_duplicates(subset=['shiftUid', 'Actual Date'])
    shift_data_df['shiftStartedAt'] = pd.to_datetime(shift_data_df['shiftStartedAt'])
    shift_data_df['shiftEndedAt'] = pd.to_datetime(shift_data_df['shiftEndedAt'])
    shift_data_df['Shift_Hours'] = (shift_data_df['shiftEndedAt'] - shift_data_df[
        'shiftStartedAt']).dt.total_seconds() / 3600

    #v_mode_final_df['Actual Date'] = pd.to_datetime(v_mode_final_df['Actual Date'])

    v_mode_shift_hours_df = v_mode_final_df.copy()
    v_mode_shift_hours_df = v_mode_shift_hours_df.drop_duplicates(subset=['shiftUid', 'Actual Date'])
    # Calculate shift hours for V_Mode
    v_mode_shift_hours_df['shiftStartedAt'] = pd.to_datetime(v_mode_shift_hours_df['shiftStartedAt'])
    v_mode_shift_hours_df['shiftEndedAt'] = pd.to_datetime(v_mode_shift_hours_df['shiftEndedAt'])
    v_mode_shift_hours_df['Shift_Hours'] = (v_mode_shift_hours_df['shiftEndedAt'] - v_mode_shift_hours_df[
        'shiftStartedAt']).dt.total_seconds() / 3600
    #v_mode_shift_hours_df['Actual Date'] = pd.to_datetime(v_mode_shift_hours_df['Actual Date'])

    #filtered_drivers_df.to_csv('bookings51113.csv', index=False)


    # Add image to the dashboard
    image = Image.open(r'Hpcharge.png')
    col1, col2, col3, col4, col5 = st.columns(5)
    col3.image(image, use_column_width=False)

    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Executive Dashboard", "Charge Pattern Insights", "EPod Stats", "Subscription Insights",
         "Geographical Insights", "Operators Dashboard", "KM Statistics"])

    with tab1:
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            final_df['Actual Date'] = pd.to_datetime(final_df['Actual Date'], errors='coerce')
            min_date = final_df['Actual Date'].min().date()
            max_date = final_df['Actual Date'].max().date()
            start_date = st.date_input(
                'Start Date', min_value=min_date, max_value=max_date, value=min_date, key="ex-date-start")

        with col1:
            end_date = st.date_input(
                'End Date', min_value=min_date, max_value=max_date, value=max_date, key="ex-date-end")

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_df = final_df[(final_df['Actual Date'] >= start_date)
                               & (final_df['Actual Date'] <= end_date)]

        filtered_df['Actual SoC_Start'] = pd.to_numeric(
            filtered_df['Actual SoC_Start'], errors='coerce')
        filtered_df['Actual SoC_End'] = pd.to_numeric(
            filtered_df['Actual SoC_End'], errors='coerce')


        # Process Actual SoC_Start and Actual SoC_End columns
        def process_soc(value):
            try:
                numeric_value = pd.to_numeric(value, errors='coerce')
                if numeric_value > 100:
                    return int(str(numeric_value)[:2])  # Extract first 2 digits
                return numeric_value
            except:
                return np.nan


        filtered_df['Actual SoC_Start'] = filtered_df['Actual SoC_Start'].apply(process_soc)
        filtered_df['Actual SoC_End'] = filtered_df['Actual SoC_End'].apply(process_soc)

        record_count_df = filtered_df.groupby(
            ['EPOD Name', 't-15_kpi']).size().reset_index(name='Record Count')

        city_count_df = filtered_df.groupby(['Customer Location City', 't-15_kpi']).size().reset_index(
            name='Record Count')
        record_count_df = record_count_df.sort_values(by='Record Count')
        city_count_df = city_count_df.sort_values(by='Record Count')
        start_soc_stats = filtered_df.dropna(subset=['Actual SoC_Start']).groupby('EPOD Name')['Actual SoC_Start'].agg([
            'max', 'min', 'mean', 'median'])

        end_soc_stats = filtered_df.dropna(subset=['Actual SoC_End']).groupby('EPOD Name')['Actual SoC_End'].agg([
            'max', 'min', 'mean', 'median'])
        start_soc_stats = start_soc_stats.sort_values(by='EPOD Name')
        end_soc_stats = end_soc_stats.sort_values(by='EPOD Name')
        kpi_flag_data = filtered_df['t-15_kpi']

        before_time_count = (kpi_flag_data == 1).sum()
        on_time_count = (kpi_flag_data == 0).sum()
        delay_count = (kpi_flag_data == 2).sum()

        total_count = before_time_count + delay_count + on_time_count
        before_time_percentage = (before_time_count / total_count) * 100
        on_time_percentage = (on_time_count / total_count) * 100
        delay_percentage = (delay_count / total_count) * 100
        on_time_sla = (1 - (delay_percentage / 100)) * 100

        start_soc_avg = start_soc_stats['mean'].values.mean()
        start_soc_median = start_soc_stats['median'].values[0]

        end_soc_avg = end_soc_stats['mean'].values.mean()
        end_soc_median = end_soc_stats['median'].values[0]

        col2.metric("T-15 Fulfilled", f"{before_time_percentage.round(2)}%")
        col3.metric("On Time SLA", f"{on_time_sla.round(2)}%")
        col4.metric("Avg Start SoC", f"{start_soc_avg.round(2)}%")
        col5.metric("Avg End SoC", f"{end_soc_avg.round(2)}%")

        total_sessions = filtered_df['t-15_kpi'].count()
        fig = go.Figure(data=[go.Pie(labels=['T-15 Fulfilled', 'Delay', 'T-15 Not Fulfilled'],
                                     values=[before_time_count, delay_count, on_time_count],
                                     hole=0.6,
                                     sort=False,
                                     textinfo='label+percent+value',
                                     textposition='outside',
                                     marker=dict(colors=['green', 'red', 'yellow']))])

        fig.add_annotation(text='Total Sessions',
                           x=0.5, y=0.5, font_size=15, showarrow=False)

        fig.add_annotation(text=str(total_sessions),
                           x=0.5, y=0.45, font_size=15, showarrow=False)
        fig.update_layout(
            title='T-15 KPI (Overall)',
            showlegend=False,
            height=400,
            width=610
        )

        with col2:
            st.plotly_chart(fig, use_container_width=False)

        allowed_cities = final_df['Customer Location City'].dropna().unique()
        city_count_df = city_count_df[city_count_df['Customer Location City'].isin(
            allowed_cities)]

        fig_group = go.Figure()

        color_mapping = {0: 'red', 1: 'green', 2: 'yellow'}
        city_count_df['Percentage'] = city_count_df['Record Count'] / \
                                      city_count_df.groupby('Customer Location City')[
                                          'Record Count'].transform('sum') * 100

        fig_group = go.Figure()

        for flag in city_count_df['t-15_kpi'].unique():
            df_flag = city_count_df[city_count_df['t-15_kpi'] == flag]

            fig_group.add_trace(go.Bar(
                x=df_flag['Customer Location City'],
                y=df_flag['Percentage'],
                name=str(flag),
                text=df_flag['Percentage'].round(1).astype(str) + '%',
                marker=dict(color=color_mapping[flag]),
                textposition='auto'
            ))

        fig_group.update_layout(
            barmode='group',
            title='T-15 KPI (HSZ Wise)',
            xaxis={'categoryorder': 'total descending'},
            yaxis={'tickformat': '.2f', 'title': 'Percentage'},
            height=400,
            width=610,
            margin=dict(t=50, b=50, l=50, r=50),
            showlegend=False
        )

        with col5:
            st.plotly_chart(fig_group)

        filtered_city_count_df = city_count_df[city_count_df['t-15_kpi'] == 1]

        max_record_count_city = filtered_city_count_df.loc[
            filtered_city_count_df['Record Count'].idxmax(), 'Customer Location City']
        min_record_count_city = filtered_city_count_df.loc[
            filtered_city_count_df['Record Count'].idxmin(), 'Customer Location City']

        col6.metric("City with Maximum Sessions", max_record_count_city)
        col7.metric("City with Minimum Sessions", min_record_count_city)

        start_soc_max = start_soc_stats['max'].values.max()

        start_soc_min = start_soc_stats['min'].values.min()

        start_soc_avg = start_soc_stats['mean'].values.mean()
        start_soc_median = np.median(start_soc_stats['median'].values)

        gauge_range = [0, 100]

        start_soc_max_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=start_soc_max,
            title={'text': "Max Start SoC %", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}},
        ))
        start_soc_max_gauge.update_layout(width=150, height=250)

        start_soc_min_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=start_soc_min,
            title={'text': "Min Start SoC %", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))
        start_soc_min_gauge.update_layout(width=150, height=250)

        start_soc_avg_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=start_soc_avg,
            title={'text': "Avg Start SoC %", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))
        start_soc_avg_gauge.update_layout(width=150, height=250)

        start_soc_median_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=start_soc_median,
            title={'text': "Median Start SoC %", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))
        start_soc_median_gauge.update_layout(width=150, height=250)
        start_soc_median_gauge.update_layout(
            # Adjust the margins as needed
            shapes=[dict(
                type='line',
                x0=1,
                y0=-2,
                x1=1,
                y1=2,
                line=dict(
                    color="black",
                    width=1,
                )
            )]
        )
        with col3:
            for i in range(1, 27):
                st.write("\n")
            with col2:
                st.write("#### Start SoC Stats")

        with col6:
            for i in range(1, 27):
                st.write("\n")
            st.write("#### End SoC Stats")

        # Create the layout using grid container
        st.markdown('<div class="grid-container">', unsafe_allow_html=True)

        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        with col1:
            st.plotly_chart(start_soc_min_gauge)
        with col2:
            st.plotly_chart(start_soc_max_gauge)
        with col3:
            st.plotly_chart(start_soc_avg_gauge)
        with col4:
            st.plotly_chart(start_soc_median_gauge)

        end_soc_max = end_soc_stats['max'].values.max()
        end_soc_min = end_soc_stats['min'].values.min()
        end_soc_avg = end_soc_stats['mean'].values.mean()
        end_soc_median = np.median(end_soc_stats['median'].values)

        end_soc_max_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=end_soc_max,
            title={'text': "Max End SoC %", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))

        end_soc_min_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=end_soc_min,
            title={'text': "Min End SoC %", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))
        end_soc_min_gauge.update_layout(
            shapes=[
                dict(
                    type='line',
                    xref='paper',
                    yref='paper',
                    x0=0,
                    y0=-2,
                    x1=0,
                    y1=2,
                    line=dict(
                        color='black',
                        width=1
                    )
                )
            ]
        )
        end_soc_max_gauge.update_layout(width=150, height=250)
        end_soc_min_gauge.update_layout(width=150, height=250)

        end_soc_avg_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=end_soc_avg,
            title={'text': "Avg End SoC %", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))
        end_soc_avg_gauge.update_layout(width=150, height=250)

        end_soc_median_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=end_soc_median,
            title={'text': "Median End SoC %", 'font': {'size': 15}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': gauge_range}}
        ))
        end_soc_median_gauge.update_layout(width=150, height=250)

        with col5:
            st.plotly_chart(end_soc_min_gauge)

        with col6:
            st.plotly_chart(end_soc_max_gauge)
        with col7:
            st.plotly_chart(end_soc_avg_gauge)
        with col8:
            st.plotly_chart(end_soc_median_gauge)

        for city in allowed_cities:
            city_filtered_df = filtered_df[filtered_df['Customer Location City'] == city]

            city_start_soc_stats = city_filtered_df.dropna(subset=['Actual SoC_Start'])['Actual SoC_Start'].agg([
                'max', 'min', 'mean', 'median'])

            city_end_soc_stats = city_filtered_df.dropna(subset=['Actual SoC_End'])['Actual SoC_End'].agg([
                'max', 'min', 'mean', 'median'])

            city_start_soc_max = city_start_soc_stats['max'].max()
            city_start_soc_min = city_start_soc_stats['min'].min()
            city_start_soc_avg = city_start_soc_stats['mean'].mean()
            city_start_soc_median = np.median(city_start_soc_stats['median'])

            city_end_soc_max = city_end_soc_stats['max'].max()
            city_end_soc_min = city_end_soc_stats['min'].min()
            city_end_soc_avg = city_end_soc_stats['mean'].mean()
            city_end_soc_median = np.median(city_end_soc_stats['median'])

            city_start_soc_max_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=city_start_soc_max,
                title={'text': f"Start Max - {city}", 'font': {'size': 15}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': gauge_range}}
            ))
            city_start_soc_max_gauge.update_layout(width=150, height=250)
            city_start_soc_min_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=city_start_soc_min,
                title={'text': f"Start Min - {city}", 'font': {'size': 15}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': gauge_range}}
            ))
            city_start_soc_min_gauge.update_layout(width=150, height=250)
            city_start_soc_avg_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=city_start_soc_avg,
                title={'text': f"Start Avg - {city}", 'font': {'size': 15}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': gauge_range}}
            ))
            city_start_soc_avg_gauge.update_layout(width=150, height=250)
            city_start_soc_median_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=city_start_soc_median,
                title={'text': f"Start Median - {city}", 'font': {'size': 15}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': gauge_range}}
            ))
            city_start_soc_median_gauge.update_layout(
                # Adjust the margins as needed
                shapes=[dict(
                    type='line',
                    x0=1,
                    y0=-2,
                    x1=1,
                    y1=2,
                    line=dict(
                        color="black",
                        width=1,
                    )
                )]
            )
            city_start_soc_median_gauge.update_layout(width=150, height=250)
            city_end_soc_max_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=city_end_soc_max,
                title={'text': f"End Max - {city}", 'font': {'size': 15}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': gauge_range}}
            ))
            city_end_soc_max_gauge.update_layout(width=150, height=250)
            city_end_soc_min_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=city_end_soc_min,
                title={'text': f"End Min - {city}", 'font': {'size': 15}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': gauge_range}}
            ))
            city_end_soc_min_gauge.update_layout(width=150, height=250)
            city_end_soc_min_gauge.update_layout(
                shapes=[
                    dict(
                        type='line',
                        xref='paper',
                        yref='paper',
                        x0=0,
                        y0=-2,
                        x1=0,
                        y1=2,
                        line=dict(
                            color='black',
                            width=1
                        )
                    )
                ]
            )
            city_end_soc_avg_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=city_end_soc_avg,
                title={'text': f"End Avg - {city}", 'font': {'size': 15}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': gauge_range}}
            ))
            city_end_soc_avg_gauge.update_layout(width=150, height=250)
            city_end_soc_median_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=city_end_soc_median,
                title={'text': f"End Median - {city}", 'font': {'size': 15}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': gauge_range}}
            ))
            city_end_soc_median_gauge.update_layout(width=150, height=250)

            st.subheader(city)
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
            with col1:
                st.plotly_chart(city_start_soc_min_gauge)

            with col2:
                st.plotly_chart(city_start_soc_max_gauge)

            with col3:
                st.plotly_chart(city_start_soc_avg_gauge)

            with col4:
                st.plotly_chart(city_start_soc_median_gauge)

            with col5:
                st.plotly_chart(city_end_soc_min_gauge)

            with col6:
                st.plotly_chart(city_end_soc_max_gauge)
            with col7:
                st.plotly_chart(city_end_soc_avg_gauge)
            with col8:
                st.plotly_chart(city_end_soc_median_gauge)

    with tab2:
        CustomerNames = final_df['Customer Name'].unique()
        SubscriptionNames = final_df['subscriptionName'].unique()

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

        with col1:
            final_df['Actual Date'] = pd.to_datetime(final_df['Actual Date'], errors='coerce')
            min_date = final_df['Actual Date'].min().date()
            max_date = final_df['Actual Date'].max().date()
            start_date = st.date_input(
                'Start Date', min_value=min_date, max_value=max_date, value=min_date, key="cpi-date-start")

        with col2:
            end_date = st.date_input(
                'End Date', min_value=min_date, max_value=max_date, value=max_date, key="cpi-date-end")

        with col4:
            Name = st.multiselect(label='Select The Customers',
                                  options=['All'] + CustomerNames.tolist(),
                                  default='All')

        with col3:
            Sub_filter = st.multiselect(label='Select Subscription',
                                        options=['All'] + SubscriptionNames.tolist(),
                                        default='All')

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_data = final_df[(final_df['Actual Date'] >= start_date) & (final_df['Actual Date'] <= end_date)]
        if 'All' in Name:
            Name = CustomerNames
        if 'All' in Sub_filter:
            Sub_filter = SubscriptionNames
        filtered_data = filtered_data[
            (filtered_data['Customer Name'].isin(Name)) & (filtered_data['subscriptionName'].isin(Sub_filter))]


        def generate_multiline_plot(data):
            fig = go.Figure()
            color_map = {0: 'yellow', 1: 'green', 2: 'red'}
            names = {0: "T-15 Not Fulfilled", 1: "T-15 Fulfilled", 2: "Delayed"}

            # Create a new DataFrame to store the counts for each day
            daily_counts = data.pivot_table(index='Day', columns='t-15_kpi', values='count', fill_value=0).reset_index()
            daily_counts['On-Time SLA'] = daily_counts[0] + daily_counts[1]
            daily_counts['Total Count'] = daily_counts[0] + daily_counts[1] + daily_counts[2]

            for kpi_flag in data['t-15_kpi'].unique():
                subset = data[data['t-15_kpi'] == kpi_flag]
                fig.add_trace(go.Scatter(x=subset['Day'], y=subset['count'], mode='lines+text',
                                         name=names[kpi_flag], line_color=color_map[kpi_flag],
                                         text=[
                                             f"{round(count / daily_counts[daily_counts['Day'] == day]['Total Count'].values[0] * 100, 0)}%"
                                             for day, count in zip(subset['Day'], subset['count'])],
                                         textposition='top center',
                                         showlegend=True))

            # Add the "On-Time SLA" line to the plot
            fig.add_trace(go.Scatter(x=daily_counts['Day'], y=daily_counts['On-Time SLA'], mode='lines+text',
                                     name='On-Time SLA', line_color='purple',
                                     text=[
                                         f"{round(count / daily_counts[daily_counts['Day'] == day]['Total Count'].values[0] * 100, 0)}%"
                                         for day, count in zip(daily_counts['Day'], daily_counts['On-Time SLA'])],
                                     textposition='top center',
                                     showlegend=True))

            fig.add_trace(go.Scatter(x=daily_counts['Day'], y=daily_counts['Total Count'], mode='lines+markers+text',
                                     name='Total Count', line_color='blue',
                                     text=daily_counts['Total Count'],
                                     textposition='top center',
                                     showlegend=True))

            fig.update_layout(
                xaxis_title='Day', yaxis_title='Count', legend=dict(x=0, y=1.2, orientation='h'))
            fig.update_yaxes(title='Count', range=[
                0, daily_counts['Total Count'].max() * 1.2])
            fig.update_layout(width=500, height=500)
            return fig


        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        filtered_data['Day'] = pd.Categorical(filtered_data['Day'], categories=day_order, ordered=True)
        daily_count = filtered_data.groupby(['Day', 't-15_kpi']).size().reset_index(name='count')
        maxday = filtered_data.groupby(['Day']).size().reset_index(name='count')
        maxday['count'] = maxday['count'].astype(int)
        max_count_index = maxday['count'].idxmax()
        max_count_day = maxday.loc[max_count_index, 'Day']
        minday = filtered_data.groupby(['Day']).size().reset_index(name='count')
        minday['count'] = minday['count'].astype(int)
        min_count_index = minday['count'].idxmin()
        min_count_day = minday.loc[min_count_index, 'Day']

        with col7:
            for i in range(1, 10):
                st.write("\n")
            st.markdown("Most Sessions on Day")
            st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" + max_count_day + "</span>",
                        unsafe_allow_html=True)

        with col7:
            st.markdown("Min Sessions on Day")
            st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" + min_count_day + "</span>",
                        unsafe_allow_html=True)

        multiline_plot = generate_multiline_plot(daily_count)
        with col4:
            st.plotly_chart(multiline_plot)


        def count_t15_kpi(df):
            try:
                return df.groupby(['t-15_kpi']).size()['1']
            except KeyError:
                return 0


        def count_sessions(df):
            return df.shape[0]


        def count_cancelled(df):
            try:
                return df[df['canceled'] == True].shape[0]
            except KeyError:
                return 0


        def count_cancelled_with_penalty(df):
            try:
                return df[df['cancelledPenalty'] == 1].shape[0]
            except KeyError:
                return 0


        total_sessions = count_sessions(filtered_data)
        cancelled_sessions = count_cancelled(filtered_data)
        cancelled_sessions_with_penalty = count_cancelled_with_penalty(filtered_data)

        # Calculate Cancelled Sessions without Penalty (cancelled but without penalty)
        cancelled_sessions_without_penalty = cancelled_sessions

        labels = ['Actual Sessions', 'Cancelled Sessions', 'Cancelled with Penalty']
        values = [total_sessions, cancelled_sessions_without_penalty, cancelled_sessions_with_penalty]
        colors = ['blue', 'orange', 'red']

        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values, hole=0.7, textinfo='label+percent', marker=dict(colors=colors))])

        fig.update_layout(showlegend=True, width=500)

        fig.add_annotation(
            text=f"Overall Sessions: {total_sessions}", x=0.5, y=0.5, font_size=15, showarrow=False)

        fig.update_layout(width=500, height=400)

        with col1:
            st.plotly_chart(fig)


        def generate_multiline_plot(data):
            fig = go.Figure()
            color_map = {0: 'yellow', 1: 'green', 2: 'red'}
            names = {0: "T-15 Not Fulfilled", 1: "T-15 Fulfilled", 2: "Delayed"}

            time_counts = data.pivot_table(index='Booking Session time', columns='t-15_kpi', values='count',
                                           fill_value=0).reset_index()
            time_counts['On-Time SLA'] = time_counts[0] + time_counts[1]
            time_counts['Total Count'] = time_counts[0] + time_counts[1] + time_counts[2]

            fig.update_layout(xaxis_title='Booking Session Time',
                              yaxis_title='Count', legend=dict(x=0, y=1.2, orientation='h'))

            for kpi_flag in data['t-15_kpi'].unique():
                subset = data[data['t-15_kpi'] == kpi_flag]
                fig.add_trace(go.Scatter(x=subset['Booking Session time'], y=subset['count'], mode='lines+text',
                                         name=names[kpi_flag], line_color=color_map[kpi_flag],
                                         text=[
                                             f"{round(count / time_counts[time_counts['Booking Session time'] == hr]['Total Count'].values[0] * 100, 0)}%"
                                             for hr, count in zip(subset['Booking Session time'], subset['count'])],
                                         textposition='top center',
                                         showlegend=True))

            # Add the "On-Time SLA" line to the plot
            fig.add_trace(
                go.Scatter(x=time_counts['Booking Session time'], y=time_counts['On-Time SLA'], mode='lines+text',
                           name='On-Time SLA', line_color='purple',
                           text=[
                               f"{round(count / time_counts[time_counts['Booking Session time'] == day]['Total Count'].values[0] * 100, 0)}%"
                               for day, count in zip(time_counts['Booking Session time'], time_counts['On-Time SLA'])],
                           textposition='top center',
                           showlegend=True))

            fig.add_trace(go.Scatter(x=time_counts['Booking Session time'], y=time_counts['Total Count'],
                                     mode='lines+markers+text',
                                     name='Total Count', line_color='blue',
                                     text=time_counts['Total Count'],
                                     textposition='top center',
                                     showlegend=True))

            fig.update_yaxes(title='Count', range=[
                0, time_counts['Total Count'].max() * 1.2])
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(
                range(24)), ticktext=list(range(24))))
            fig.update_layout(width=1100, height=530)
            return fig


        filtered_data['Booking Session time'] = pd.to_datetime(
            filtered_data['Booking Session time'], format='mixed').dt.hour
        daily_count = filtered_data.groupby(
            ['Booking Session time', 't-15_kpi']).size().reset_index(name='count')
        maxmindf = filtered_data.groupby(
            ['Booking Session time']).size().reset_index(name='count')
        max_count_index = maxmindf['count'].idxmax()
        max_count_time = maxmindf.loc[max_count_index, 'Booking Session time']
        min_count_index = maxmindf['count'].idxmin()
        min_count_time = maxmindf.loc[min_count_index, 'Booking Session time']
        with col7:
            for i in range(1, 18):
                st.write("\n")
            st.markdown("Max Sessions at Time")
            st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                        str(max_count_time) + "</span>", unsafe_allow_html=True)
        with col7:
            st.markdown("Min Sessions at Time")
            st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                        str(min_count_time) + "</span>", unsafe_allow_html=True)
        multiline_plot = generate_multiline_plot(daily_count)
        with col1:
            st.plotly_chart(multiline_plot)
        st.divider()

        HSZs = final_df['Customer Location City'].dropna().unique()
        for city in HSZs:
            st.subheader(city)
            CustomerNames = final_df['Customer Name'].unique()
            SubscriptionNames = final_df['subscriptionName'].unique()

            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

            with col1:
                final_df['Actual Date'] = pd.to_datetime(final_df['Actual Date'], errors='coerce')
                min_date = final_df['Actual Date'].min().date()
                max_date = final_df['Actual Date'].max().date()
                start_date = st.date_input(
                    'Start Date', min_value=min_date, max_value=max_date, value=min_date, key=f"{city}cpi-date-start")

            with col2:
                end_date = st.date_input(
                    'End Date', min_value=min_date, max_value=max_date, value=max_date, key=f"{city}cpi-date-end")
            with col4:

                Name = st.multiselect(label='Select The Customers',
                                      options=['All'] + CustomerNames.tolist(),
                                      default='All', key=f"{city}names")

            with col3:
                Sub_filter = st.multiselect(label='Select Subscription',
                                            options=['All'] +
                                                    SubscriptionNames.tolist(),
                                            default='All', key=f"{city}sub")
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            filtered_data = final_df[(final_df['Actual Date'] >= start_date)
                                     & (final_df['Actual Date'] <= end_date)]
            if 'All' in Name:
                Name = CustomerNames

            if 'All' in Sub_filter:
                Sub_filter = SubscriptionNames

            filtered_data = filtered_data[
                (filtered_data['Customer Name'].isin(Name)) &
                (filtered_data['subscriptionName'].isin(Sub_filter))
                ]
            filtered_data = filtered_data[
                (filtered_data['Customer Location City'] == city)]


            def generate_multiline_plot(data):
                fig = go.Figure()
                color_map = {0: 'yellow', 1: 'green', 2: 'red'}
                names = {0: "T-15 Not Fulfilled", 1: "T-15 Fulfilled", 2: "Delayed"}

                # Create a new DataFrame to store the counts for each day
                daily_counts = data.pivot_table(index='Day', columns='t-15_kpi', values='count',
                                                fill_value=0).reset_index()
                daily_counts['On-Time SLA'] = daily_counts[0] + daily_counts[1]
                daily_counts['Total Count'] = daily_counts[0] + daily_counts[1] + daily_counts[2]

                for kpi_flag in data['t-15_kpi'].unique():
                    subset = data[data['t-15_kpi'] == kpi_flag]
                    fig.add_trace(go.Scatter(x=subset['Day'], y=subset['count'], mode='lines+text',
                                             name=names[kpi_flag], line_color=color_map[kpi_flag],
                                             text=[
                                                 f"{round(count / daily_counts[daily_counts['Day'] == day]['Total Count'].values[0] * 100, 0)}%"
                                                 for day, count in zip(subset['Day'], subset['count'])],
                                             textposition='top center',
                                             showlegend=True))

                # Add the "On-Time SLA" line to the plot
                fig.add_trace(go.Scatter(x=daily_counts['Day'], y=daily_counts['On-Time SLA'], mode='lines+text',
                                         name='On-Time SLA', line_color='purple',
                                         text=[
                                             f"{round(count / daily_counts[daily_counts['Day'] == day]['Total Count'].values[0] * 100, 0)}%"
                                             for day, count in zip(daily_counts['Day'], daily_counts['On-Time SLA'])],
                                         textposition='top center',
                                         showlegend=True))

                fig.add_trace(
                    go.Scatter(x=daily_counts['Day'], y=daily_counts['Total Count'], mode='lines+markers+text',
                               name='Total Count', line_color='blue',
                               text=daily_counts['Total Count'],
                               textposition='top center',
                               showlegend=True))

                fig.update_layout(
                    xaxis_title='Day', yaxis_title='Count', legend=dict(x=0, y=1.2, orientation='h'))
                fig.update_yaxes(title='Count', range=[
                    0, daily_counts['Total Count'].max() * 1.2])
                fig.update_layout(width=500, height=500)
                return fig


            day_order = ['Monday', 'Tuesday', 'Wednesday',
                         'Thursday', 'Friday', 'Saturday', 'Sunday']
            filtered_data['Day'] = pd.Categorical(
                filtered_data['Day'], categories=day_order, ordered=True)
            daily_count = filtered_data.groupby(
                ['Day', 't-15_kpi']).size().reset_index(name='count')
            maxday = filtered_data.groupby(
                ['Day']).size().reset_index(name='count')
            maxday['count'] = maxday['count'].astype(int)
            max_count_index = maxday['count'].idxmax()
            max_count_day = maxday.loc[max_count_index, 'Day']
            minday = filtered_data.groupby(
                ['Day']).size().reset_index(name='count')
            minday['count'] = minday['count'].astype(int)
            min_count_index = minday['count'].idxmin()
            min_count_day = minday.loc[min_count_index, 'Day']
            with col7:
                for i in range(1, 10):
                    st.write("\n")
                st.markdown("Most Sessions on Day")
                st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                            max_count_day + "</span>", unsafe_allow_html=True)
            with col7:
                st.markdown("Min Sessions on Day")
                st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                            min_count_day + "</span>", unsafe_allow_html=True)
            multiline_plot = generate_multiline_plot(daily_count)

            with col4:
                st.plotly_chart(multiline_plot)


            def count_t15_kpi(df):
                try:
                    return df.groupby(
                        ['t-15_kpi']).size()['1']
                except KeyError:
                    return 0


            def count_sessions(df):
                return df.shape[0]


            def count_cancelled(df):
                try:
                    return df[df['canceled'] == True].shape[0]
                except KeyError:
                    return 0


            def count_cancelled_with_penalty(df):
                try:
                    return df[df['cancelledPenalty'] == 1].shape[0]
                except KeyError:
                    return 0


            total_sessions = count_sessions(filtered_data)
            cancelled_sessions = count_cancelled(filtered_data)
            cancelled_sessions_with_penalty = count_cancelled_with_penalty(filtered_data)

            # Calculate Cancelled Sessions without Penalty (cancelled but without penalty)
            cancelled_sessions_without_penalty = cancelled_sessions

            labels = ['Actual Sessions', 'Cancelled Sessions', 'Cancelled with Penalty']
            values = [total_sessions, cancelled_sessions_without_penalty, cancelled_sessions_with_penalty]
            colors = ['blue', 'orange', 'red']

            fig = go.Figure(
                data=[go.Pie(labels=labels, values=values, hole=0.7, textinfo='label+percent',
                             marker=dict(colors=colors))])

            fig.update_layout(
                showlegend=True, width=500,
            )

            fig.add_annotation(
                text=f"Overall Sessions: {total_sessions}", x=0.5, y=0.5, font_size=15, showarrow=False)

            fig.update_layout(width=500, height=400)

            with col1:
                st.plotly_chart(fig)


            def generate_multiline_plot(data):
                fig = go.Figure()
                color_map = {0: 'yellow', 1: 'green', 2: 'red'}
                names = {0: "T-15 Not Fulfilled", 1: "T-15 Fulfilled", 2: "Delayed"}

                time_counts = data.pivot_table(index='Booking Session time', columns='t-15_kpi', values='count',
                                               fill_value=0).reset_index()
                time_counts['On-Time SLA'] = time_counts[0] + time_counts[1]
                time_counts['Total Count'] = time_counts[0] + time_counts[1] + time_counts[2]

                fig.update_layout(xaxis_title='Booking Session Time',
                                  yaxis_title='Count', legend=dict(x=0, y=1.2, orientation='h'))

                for kpi_flag in data['t-15_kpi'].unique():
                    subset = data[data['t-15_kpi'] == kpi_flag]
                    fig.add_trace(go.Scatter(x=subset['Booking Session time'], y=subset['count'], mode='lines+text',
                                             name=names[kpi_flag], line_color=color_map[kpi_flag],
                                             text=[
                                                 f"{round(count / time_counts[time_counts['Booking Session time'] == hr]['Total Count'].values[0] * 100, 0)}%"
                                                 for hr, count in zip(subset['Booking Session time'], subset['count'])],
                                             textposition='top center',
                                             showlegend=True))

                # Add the "On-Time SLA" line to the plot
                fig.add_trace(
                    go.Scatter(x=time_counts['Booking Session time'], y=time_counts['On-Time SLA'], mode='lines+text',
                               name='On-Time SLA', line_color='purple',
                               text=[
                                   f"{round(count / time_counts[time_counts['Booking Session time'] == day]['Total Count'].values[0] * 100, 0)}%"
                                   for day, count in
                                   zip(time_counts['Booking Session time'], time_counts['On-Time SLA'])],
                               textposition='top center',
                               showlegend=True))

                fig.add_trace(go.Scatter(x=time_counts['Booking Session time'], y=time_counts['Total Count'],
                                         mode='lines+markers+text',
                                         name='Total Count', line_color='blue',
                                         text=time_counts['Total Count'],
                                         textposition='top center',
                                         showlegend=True))

                fig.update_yaxes(title='Count', range=[
                    0, time_counts['Total Count'].max() * 1.2])
                fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(
                    range(24)), ticktext=list(range(24))))
                fig.update_layout(width=1100, height=530)
                return fig


            filtered_data['Booking Session time'] = pd.to_datetime(
                filtered_data['Booking Session time'], format='mixed').dt.hour
            daily_count = filtered_data.groupby(
                ['Booking Session time', 't-15_kpi']).size().reset_index(name='count')
            maxmindf = filtered_data.groupby(
                ['Booking Session time']).size().reset_index(name='count')
            max_count_index = maxmindf['count'].idxmax()
            max_count_time = maxmindf.loc[max_count_index, 'Booking Session time']
            min_count_index = maxmindf['count'].idxmin()
            min_count_time = maxmindf.loc[min_count_index, 'Booking Session time']
            with col7:
                for i in range(1, 18):
                    st.write("\n")
                st.markdown("Max Sessions at Time")
                st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                            str(max_count_time) + "</span>", unsafe_allow_html=True)

            with col7:
                st.markdown("Min Sessions at Time")
                st.markdown("<span style = 'font-size:25px;line-height: 0.8;'>" +
                            str(min_count_time) + "</span>", unsafe_allow_html=True)

            multiline_plot = generate_multiline_plot(daily_count)

            with col1:
                st.plotly_chart(multiline_plot)

            st.divider()

    with tab3:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            final_df['Actual Date'] = pd.to_datetime(final_df['Actual Date'], errors='coerce')
            min_date = final_df['Actual Date'].min().date()
            max_date = final_df['Actual Date'].max().date()
            start_date = st.date_input('Start Date', min_value=min_date, max_value=max_date, value=min_date,
                                       key="epod-date-start")
        with col2:
            end_date = st.date_input('End Date', min_value=min_date, max_value=max_date, value=max_date,
                                     key="epod-date-end")
        final_df['EPOD Name'] = final_df['EPOD Name'].str.replace('-', '')

        epods = final_df['EPOD Name'].unique()
        locations = final_df['Customer Location City'].unique()

        with col3:
            EPod = st.multiselect(label='Select The EPOD', options=['All'] + epods.tolist(), default='All')

        with col4:
            location_city = st.multiselect(label='Select Customer Location City', options=['All'] + locations.tolist(),
                                           default='All')

        with col1:
            st.markdown(":large_green_square: T-15 fulfilled")
        with col2:
            st.markdown(":large_yellow_square: T-15 Not fulfilled")
        with col3:
            st.markdown(":large_red_square: Delay")

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_data = final_df[(final_df['Actual Date'] >= start_date) & (final_df['Actual Date'] <= end_date)]

        if 'All' in EPod:
            EPod = epods

        if 'All' in location_city:
            location_city = locations

        filtered_data = filtered_data[
            (filtered_data['EPOD Name'].isin(EPod)) & (filtered_data['Customer Location City'].isin(location_city))]

        # Filter and process data for 'filtered_data' DataFrame
        filtered_data['KWH Pumped Per Session'] = filtered_data['KWH Pumped Per Session'].replace('', np.nan)
        filtered_data = filtered_data[filtered_data['KWH Pumped Per Session'] != '#VALUE!']
        filtered_data['KWH Pumped Per Session'] = filtered_data['KWH Pumped Per Session'].astype(float)
        filtered_data['KWH Pumped Per Session'] = filtered_data['KWH Pumped Per Session'].abs()

        # Calculate average kWh per EPod per session
        average_kwh = filtered_data.groupby('EPOD Name')['KWH Pumped Per Session'].mean().reset_index().round(1)

        # Calculate average kWh per session across all EPods
        avgkwh = round(average_kwh['KWH Pumped Per Session'].mean(), 2)

        with col5:
            st.markdown("Average kWh/EPod per Session")
            st.markdown("<span style='font-size: 25px;line-height: 0.8;'>" + str(avgkwh) + "</span>",
                        unsafe_allow_html=True)

        # Calculate the total count for each EPod
        total_count_per_epod = filtered_data.groupby('EPOD Name')['t-15_kpi'].count().reset_index(name='Total Count')

        # Merge the total count with record_count_df to get the denominator for percentage calculation
        record_count_df = total_count_per_epod

        # Calculate the percentage for each EPod
        record_count_df['Percentage'] = (record_count_df['Total Count'] / record_count_df['Total Count'].sum()) * 100

        # Calculate the average sessions per EPod
        #average_sessions_per_epod = filtered_data.groupby('EPOD Name')['t-15_kpi'].mean().reset_index(
         #   name='Average Sessions')
        # Calculate the average sessions per EPOD
        total_sessions_per_epod = filtered_data.groupby('EPOD Name')['t-15_kpi'].count().reset_index(
            name='Total Sessions')
        unique_days_per_epod = filtered_data.groupby('EPOD Name')['Actual Date'].nunique().reset_index(
            name='Unique Days')

        # Merge the total sessions with unique days to calculate the average sessions per EPOD
        average_sessions_per_epod = total_sessions_per_epod.merge(unique_days_per_epod, on='EPOD Name')
        average_sessions_per_epod['Average Sessions'] = (
                    average_sessions_per_epod['Total Sessions'] / average_sessions_per_epod['Unique Days']).round(2)

        # Calculate the average sessions per day for each EPod
        avg_sessions_per_day = filtered_data.groupby(['EPOD Name', 'Actual Date']).size().reset_index(
            name='Sessions Count')
        avg_sessions_per_day = avg_sessions_per_day.groupby('EPOD Name')['Sessions Count'].mean().reset_index()

        # Identify EPOD with Minimum and Maximum Sessions per day
        min_sessions_epod = avg_sessions_per_day.loc[avg_sessions_per_day['Sessions Count'].idxmin()]
        max_sessions_epod = avg_sessions_per_day.loc[avg_sessions_per_day['Sessions Count'].idxmax()]

        # Ensure record_count_df is correctly defined with 'Record Count' and 'Color'
        record_count_df = filtered_data.groupby(['EPOD Name', 't-15_kpi']).size().reset_index(name='Record Count')

        # Define colors for the different KPI values
        record_count_df['Color'] = record_count_df['t-15_kpi'].apply(
            lambda x: 'yellow' if x == 0 else ('green' if x == 1 else 'red'))

        # Calculate the total count for each EPod
        total_count_per_epod = filtered_data.groupby('EPOD Name')['t-15_kpi'].count().reset_index(name='Total Count')

        # Merge the total count with record_count_df to get the denominator for percentage calculation
        record_count_df = record_count_df.merge(total_count_per_epod, on='EPOD Name')

        # Calculate the percentage for each EPod
        record_count_df['Percentage'] = (record_count_df['Record Count'] / record_count_df['Total Count']) * 100

        # Calculate the percentage of T-15 Fulfilled and T-15 Not Fulfilled for each EPod
        sla_data = record_count_df.pivot(index='EPOD Name', columns='t-15_kpi', values='Percentage').reset_index()
        sla_data['On-Time SLA'] = sla_data[0] + sla_data[1]

        max_value = max(record_count_df['Record Count'].max(), sla_data['On-Time SLA'].max()) * 1.2

        # Calculate the average sessions per EPOD
        total_sessions_per_epod = filtered_data.groupby('EPOD Name')['t-15_kpi'].count().reset_index(
            name='Total Sessions')
        unique_days_per_epod = filtered_data.groupby('EPOD Name')['Actual Date'].nunique().reset_index(
            name='Unique Days')

        # Merge the total sessions with unique days to calculate the average sessions per EPOD
        average_sessions_per_epod = total_sessions_per_epod.merge(unique_days_per_epod, on='EPOD Name')
        average_sessions_per_epod['Average Sessions'] = (
                    average_sessions_per_epod['Total Sessions'] / average_sessions_per_epod['Unique Days']).round(2)

        # Calculate the overall average sessions per day
        overall_avg_sessions_per_day = round(average_sessions_per_epod['Average Sessions'].mean(), 2)

        # Determine the percentage of EPODs doing more than the average sessions per day
        epods_above_avg = average_sessions_per_epod[
            average_sessions_per_epod['Average Sessions'] > overall_avg_sessions_per_day]
        percent_epods_above_avg = round(len(epods_above_avg) / len(average_sessions_per_epod) * 100, 2)

        # Display the widget
        #st.markdown(f"### Percentage of EPODs doing more than {overall_avg_sessions_per_day} average sessions per day")
        #st.markdown(f"<span style='font-size: 25px;line-height: 0.8;'>{percent_epods_above_avg}%</span>",
        #            unsafe_allow_html=True)

        fig = go.Figure()
        # Add the bar traces for T-15 Fulfilled, T-15 Not Fulfilled, and Delay
        for color, kpi_group in record_count_df.groupby('Color'):
            fig.add_trace(go.Bar(
                x=kpi_group['EPOD Name'],
                y=kpi_group['Percentage'],
                text=kpi_group['Percentage'].round(2).astype(str) + '%',
                textposition='auto',
                name=color,
                marker=dict(color=color),
                width=0.38,
                showlegend=False
            ))

        # Add the line trace for On-Time SLA
        fig.add_trace(go.Scatter(
            x=sla_data['EPOD Name'],
            y=sla_data['On-Time SLA'],
            text=sla_data['On-Time SLA'].round(0).astype(str) + '%',  # Display On-Time SLA text
            textposition='top center',
            mode='lines+markers+text',  # Add 'text' to display the text values
            line=dict(color='purple', width=2),  # Set the line color to purple
            marker=dict(color='purple', size=8),
            name='On-Time SLA',
            yaxis='y2'  # Plot the line on the secondary y-axis
        ))

        fig.update_layout(
            xaxis={'categoryorder': 'array', 'categoryarray': record_count_df['EPOD Name']},
            yaxis={'range': [0, max_value]},
            xaxis_title='EPOD Name',
            yaxis_title='Sessions',
            yaxis2=dict(overlaying='y', side='right', showgrid=False, range=[0, max_value]),
            height=340,
            width=600,
            title="T-15 for each EPod with On-Time SLA",
            legend=dict(title_font=dict(size=14), font=dict(size=12), x=0, y=1.1, orientation='h'),
        )

        with col1:
            st.plotly_chart(fig)

        # Calculate the total sessions per EPod
        total_sessions_per_epod = filtered_data.groupby('EPOD Name')['t-15_kpi'].count().reset_index(
            name='Total Sessions')

        # Create the total sessions per EPod bar graph
        fig_total_sessions = go.Figure()
        fig_total_sessions.add_trace(go.Bar(
            x=total_sessions_per_epod['EPOD Name'],
            y=total_sessions_per_epod['Total Sessions'],
            text=total_sessions_per_epod['Total Sessions'],
            textposition='auto',
            name='Total Sessions'
        ))

        fig_total_sessions.update_layout(
            xaxis_title='EPOD Name',
            yaxis_title='Total Sessions',
            barmode='group',
            width=600,
            height=340,
            title="Total Sessions Per EPod",
        )


        with col6:
            st.markdown("Minimum Sessions per Day (EPOD)")
            st.markdown(
                f"<span style='font-size: 25px;line-height: 0.8;'>{min_sessions_epod['EPOD Name']} - {round(min_sessions_epod['Sessions Count'], 2)}</span>",
                unsafe_allow_html=True
            )

        # Create the average sessions per EPod bar graph in actual numbers
        fig_avg_sessions = go.Figure()
        fig_avg_sessions.add_trace(go.Bar(
            x=average_sessions_per_epod['EPOD Name'],
            y=average_sessions_per_epod['Average Sessions'],
            text=average_sessions_per_epod['Average Sessions'].round(2),
            textposition='auto',
            name='Average Sessions'
        ))

        fig_avg_sessions.update_layout(
            xaxis_title='EPOD Name',
            yaxis_title='Average Sessions',
            barmode='group',
            width=600,
            height=340,
            title="Average Sessions Per EPod",
        )



        with col4:
            st.markdown("Maximum Sessions per Day (EPOD)")
            st.markdown(
                f"<span style='font-size: 25px;line-height: 0.8;'>{max_sessions_epod['EPOD Name']} - {round(max_sessions_epod['Sessions Count'], 2)}</span>",
                unsafe_allow_html=True
            )


        with col5:
            avg_sessions = avg_sessions_per_day['Sessions Count'].mean().round(2)
            st.markdown("Average Sessions per Day (All EPODs)")
            st.markdown(
                f"<span style='font-size: 25px;line-height: 0.8;'>{round(avg_sessions, 2)}</span>",
                unsafe_allow_html=True
            )

        with col6:
            st.markdown(
                "% of EPODs doing more than average sessions per day")
            st.markdown(f"<span style='font-size: 25px;line-height: 0.8;'>{percent_epods_above_avg}%</span>",
                        unsafe_allow_html=True)

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.plotly_chart(fig_total_sessions)

        with col4:
            st.plotly_chart(fig_avg_sessions)


        st.divider()

        # Dropdown for weekly and monthly analysis
        analysis_type = st.selectbox('Select Analysis Type', ['Weekly', 'Monthly'], key='analysis_type')

        # Weekly Analysis
        if analysis_type == 'Weekly':
            # Separate date and EPOD selection for weekly analysis
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                weekly_start_date = st.date_input('Weekly Start Date', min_value=min_date, max_value=max_date,
                                                  value=min_date,
                                                  key="weekly-date-start")
            with col2:
                weekly_end_date = st.date_input('Weekly End Date', min_value=min_date, max_value=max_date,
                                                value=max_date,
                                                key="weekly-date-end")
            with col3:
                weekly_epod = st.multiselect(label='Select The Weekly EPOD', options=['All'] + epods.tolist(),
                                             default='All', key="weekly-epod")
            with col4:
                weekly_location_city = st.multiselect(label='Select Customer Location City',
                                                      options=['All'] + locations.tolist(), default='All',
                                                      key="weekly-location-city")

            weekly_start_date = pd.to_datetime(weekly_start_date)
            weekly_end_date = pd.to_datetime(weekly_end_date)
            weekly_filtered_data = final_df[
                (final_df['Actual Date'] >= weekly_start_date) & (final_df['Actual Date'] <= weekly_end_date)]

            if 'All' in weekly_epod:
                weekly_epod = epods

            if 'All' in weekly_location_city:
                weekly_location_city = locations

            weekly_filtered_data = weekly_filtered_data[(weekly_filtered_data['EPOD Name'].isin(weekly_epod)) & (weekly_filtered_data['Customer Location City'].isin(weekly_location_city))]


            weekly_filtered_data['Week'] = weekly_filtered_data['Actual Date'].dt.isocalendar().week
            weekly_sessions = weekly_filtered_data.groupby(['EPOD Name', 'Week'])['t-15_kpi'].count().reset_index(
                name='Sessions')
            weekly_kwh = weekly_filtered_data.groupby(['EPOD Name', 'Week'])[
                'KWH Pumped Per Session'].mean().reset_index(
                name='Average KWH Pumped')

            weekly_avg_kwh = round(weekly_filtered_data['KWH Pumped Per Session'].mean(), 2)
            weekly_sessions_per_epod = weekly_filtered_data.groupby('EPOD Name')['t-15_kpi'].count().mean().round(2)

            with col5:
                st.markdown("Average KWH Pumped Per EPod (Weekly)")
                st.markdown(f"<span style='font-size: 25px;line-height: 0.8;'>{weekly_avg_kwh}</span>",
                            unsafe_allow_html=True)

            with col6:
                st.markdown("Sessions Per EPod (Weekly)")
                st.markdown(f"<span style='font-size: 25px;line-height: 0.8;'>{weekly_sessions_per_epod}</span>",
                            unsafe_allow_html=True)

            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                # Plot weekly sessions
                fig_weekly_sessions = px.bar(weekly_sessions, x='Week', y='Sessions', color='EPOD Name',
                                             barmode='group',
                                             title='Weekly Sessions Per EPod')
                st.plotly_chart(fig_weekly_sessions)

            with col5:
                # Plot weekly KWH Pumped
                fig_weekly_kwh = px.bar(weekly_kwh, x='Week', y='Average KWH Pumped', color='EPOD Name',
                                        barmode='group',
                                        title='Weekly Average KWH Pumped Per EPod')
                st.plotly_chart(fig_weekly_kwh)







        # Monthly Analysis
        elif analysis_type == 'Monthly':
            # Separate date and EPOD selection for monthly analysis
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                monthly_start_date = st.date_input('Monthly Start Date', min_value=min_date, max_value=max_date,
                                                   value=min_date,
                                                   key="monthly-date-start")
            with col2:
                monthly_end_date = st.date_input('Monthly End Date', min_value=min_date, max_value=max_date,
                                                 value=max_date,
                                                 key="monthly-date-end")
            with col3:
                monthly_epod = st.multiselect(label='Select The Monthly EPOD', options=['All'] + epods.tolist(),
                                              default='All', key="monthly-epod")

            with col4:
                monthly_location_city = st.multiselect(label='Select Customer Location City',
                                                       options=['All'] + locations.tolist(), default='All',
                                                       key="monthly-location-city")

            monthly_start_date = pd.to_datetime(monthly_start_date)
            monthly_end_date = pd.to_datetime(monthly_end_date)
            monthly_filtered_data = final_df[
                (final_df['Actual Date'] >= monthly_start_date) & (final_df['Actual Date'] <= monthly_end_date)]

            if 'All' in monthly_epod:
                monthly_epod = epods

            if 'All' in monthly_location_city:
                monthly_location_city = locations

            monthly_filtered_data = monthly_filtered_data[(monthly_filtered_data['EPOD Name'].isin(monthly_epod)) & (
                monthly_filtered_data['Customer Location City'].isin(monthly_location_city))]


            monthly_filtered_data['Month'] = monthly_filtered_data['Actual Date'].dt.month

            monthly_sessions = monthly_filtered_data.groupby(['EPOD Name', 'Month'])['t-15_kpi'].count().reset_index(
                name='Sessions')
            monthly_kwh = monthly_filtered_data.groupby(['EPOD Name', 'Month'])[
                'KWH Pumped Per Session'].mean().reset_index(
                name='Average KWH Pumped')


            monthly_avg_kwh = round(monthly_filtered_data['KWH Pumped Per Session'].mean(), 2)
            monthly_sessions_per_epod = monthly_filtered_data.groupby('EPOD Name')['t-15_kpi'].count().mean().round(
                2)

            with col5:
                st.markdown("Average KWH Pumped Per EPod (Monthly)")
                st.markdown(f"<span style='font-size: 25px;line-height: 0.8;'>{monthly_avg_kwh}</span>",
                            unsafe_allow_html=True)

            with col6:
                st.markdown("Sessions Per EPod (Monthly)")
                st.markdown(f"<span style='font-size: 25px;line-height: 0.8;'>{monthly_sessions_per_epod}</span>",
                            unsafe_allow_html=True)

            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                # Plot monthly sessions
                fig_monthly_sessions = px.bar(monthly_sessions, x='Month', y='Sessions', color='EPOD Name',
                                              barmode='group',
                                              title='Monthly Sessions Per EPod')
                st.plotly_chart(fig_monthly_sessions)


            with col5:
                # Plot monthly KWH Pumped
                fig_monthly_kwh = px.bar(monthly_kwh, x='Month', y='Average KWH Pumped', color='EPOD Name',
                                         barmode='group',
                                         title='Monthly Average KWH Pumped Per EPod')
                st.plotly_chart(fig_monthly_kwh)


    with tab4:
        def generate_bar_graph(filtered_df):
            type_counts = filtered_df['subscriptionName'].value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']
            total_sessions = type_counts['Count'].sum()
            type_counts['Percentage'] = (type_counts['Count'] / total_sessions) * 100
            type_counts['Percentage'] = type_counts['Percentage'].round(2)
            fig = px.bar(type_counts, x='Type', y='Percentage', text='Percentage',
                         labels={'Type': 'Subscription', 'Percentage': 'Percentage'}, width=525, height=525,
                         title='Total Sessions by Subscription Type')
            fig.update_layout(xaxis=dict(tickangle=-45))
            fig.update_traces(textposition='outside')
            return fig


        def generate_daywise_graph(filtered_df):
            filtered_df['Day'] = filtered_df['Actual Date'].dt.day_name()
            daywise_counts = filtered_df.groupby(['Day', 'subscriptionName']).size().reset_index(name='Count')
            daywise_totals = daywise_counts.groupby('Day')['Count'].sum().reset_index(name='Total')
            daywise_counts = pd.merge(daywise_counts, daywise_totals, on='Day')
            daywise_counts['Percentage'] = (daywise_counts['Count'] / daywise_counts['Total']) * 100
            daywise_counts['Percentage'] = daywise_counts['Percentage'].round(2)
            fig = px.bar(daywise_counts, x='Day', y='Percentage', color='subscriptionName', text='Percentage',
                         labels={'Day': 'Day of the Week', 'Percentage': 'Percentage',
                                 'subscriptionName': 'Subscription'},
                         title='Total Sessions by Subscription Type by Day of the Week')
            fig.update_layout(barmode='group', xaxis=dict(tickangle=-45))
            fig.update_traces(textposition='outside')
            return fig


        final_df['type'] = final_df['subscriptionName'].str.replace('-', '')
        min_date = final_df['Actual Date'].min().date()
        max_date = final_df['Actual Date'].max().date()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Overall")
        with col2:
            start_date = st.date_input('Start Date', min_value=min_date, max_value=max_date, value=min_date,
                                       key="sub_start_date")
        with col3:
            end_date = st.date_input('End Date', min_value=min_date, max_value=max_date, value=max_date,
                                     key="sub_end_date")

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_df = final_df[(final_df['Actual Date'] >= start_date) & (final_df['Actual Date'] <= end_date)]
        bar_graph = generate_bar_graph(filtered_df)
        bar_graph.update_layout(width=400, height=490)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(bar_graph)
            st.write("\n")

        daywise_graph = generate_daywise_graph(filtered_df)
        daywise_graph.update_layout(width=400, height=490)
        with col2:
            st.plotly_chart(daywise_graph)
            st.write("\n")

            # Export the daywise graph data to CSV
            daywise_counts = filtered_df.groupby(
                [filtered_df['Actual Date'].dt.day_name(), 'subscriptionName']).size().unstack(fill_value=0).reset_index()
            csv = daywise_counts.to_csv(index=False)

            st.download_button(
                label="Download Daywise Data as CSV",
                data=csv,
                file_name='daywise_subscription_sessions.csv',
                mime='text/csv',
            )

        filtered_df['KWH Pumped Per Session'] = filtered_df['KWH Pumped Per Session'].replace('', np.nan)
        filtered_df = filtered_df[filtered_df['KWH Pumped Per Session'] != '#VALUE!']
        filtered_df['KWH Pumped Per Session'] = filtered_df['KWH Pumped Per Session'].astype(float)
        filtered_df['KWH Pumped Per Session'] = filtered_df['KWH Pumped Per Session'].abs()
        average_kwh = filtered_df.groupby('subscriptionName')['KWH Pumped Per Session'].mean().reset_index().round(1)
        fig = go.Figure(data=[go.Bar(x=average_kwh['subscriptionName'], y=average_kwh['KWH Pumped Per Session'],
                                     text=average_kwh['KWH Pumped Per Session'], textposition='outside')])
        fig.update_layout(xaxis_title='Subscription', yaxis_title='Average kWh Pumped',
                          title='Average kWh Pumped Per Session by Subscription Type', width=400, height=490,
                          xaxis=dict(tickangle=-45))
        with col3:
            st.plotly_chart(fig)
            st.write("\n")

        for city in filtered_df['Customer Location City'].dropna().unique():
            col1, col2, col3 = st.columns(3)
            with col1:
                for i in range(1, 4):
                    st.write("\n")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader(city)
            with col2:
                start_date = st.date_input('Start Date', min_value=min_date, max_value=max_date, value=min_date,
                                           key=f"{city}sub_start_date")
            with col3:
                end_date = st.date_input('End Date', min_value=min_date, max_value=max_date, value=max_date,
                                         key=f"{city}sub_end_date")

            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            filtered_df = final_df[(final_df['Actual Date'] >= start_date) & (final_df['Actual Date'] <= end_date)]
            filtered_df = filtered_df[filtered_df['Customer Location City'] == city]
            bar_graph = generate_bar_graph(filtered_df)
            bar_graph.update_layout(width=400, height=490)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(bar_graph)
                st.write("\n")

            daywise_graph = generate_daywise_graph(filtered_df)
            daywise_graph.update_layout(width=400, height=490)
            with col2:
                st.plotly_chart(daywise_graph)
                st.write("\n")

            filtered_df['KWH Pumped Per Session'] = filtered_df['KWH Pumped Per Session'].replace('', np.nan)
            filtered_df = filtered_df[filtered_df['KWH Pumped Per Session'] != '#VALUE!']
            filtered_df['KWH Pumped Per Session'] = filtered_df['KWH Pumped Per Session'].astype(float)
            filtered_df['KWH Pumped Per Session'] = filtered_df['KWH Pumped Per Session'].abs()
            average_kwh = filtered_df.groupby('subscriptionName')['KWH Pumped Per Session'].mean().reset_index().round(1)
            fig = go.Figure(data=[go.Bar(x=average_kwh['subscriptionName'], y=average_kwh['KWH Pumped Per Session'],
                                         text=average_kwh['KWH Pumped Per Session'], textposition='outside')])
            fig.update_layout(xaxis_title='Subscription', yaxis_title='Average kWh Pumped',
                              title='Average kWh Pumped Per Session by Subscription Type', width=400, height=490,
                              xaxis=dict(tickangle=-45))
            with col3:
                st.plotly_chart(fig)
                st.write("\n")

    with tab5:
        # Define Subs_Data based on the previously filtered final_df
        Subs_Data = final_df.copy()

        # UI for selecting date range, Customers, and Subscriptions
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

        with col1:
            Subs_Data['Actual Date'] = pd.to_datetime(Subs_Data['Actual Date'], errors='coerce')
            min_date = Subs_Data['Actual Date'].min().date()
            max_date = Subs_Data['Actual Date'].max().date()
            start_date = st.date_input('Start Date', min_value=min_date, max_value=max_date, value=min_date,
                                       key="cpi-date-start-input")
        with col2:
            end_date = st.date_input('End Date', min_value=min_date, max_value=max_date, value=max_date,
                                     key="cpi-date-end-input")
        with col4:
            selected_customers = st.multiselect(label='Select Customers',
                                                options=['All'] + Subs_Data['Customer Name'].unique().tolist(),
                                                default='All')
        with col3:
            selected_subscriptions = st.multiselect(label='Select Subscription',
                                                    options=['All'] + Subs_Data['type'].unique().tolist(),
                                                    default='All')

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        Subs_Data = Subs_Data[(Subs_Data['Actual Date'] >= start_date) & (Subs_Data['Actual Date'] <= end_date)]

        if 'All' not in selected_customers:
            Subs_Data = Subs_Data[Subs_Data['Customer Name'].isin(selected_customers)]
        if 'All' not in selected_subscriptions:
            Subs_Data = Subs_Data[Subs_Data['type'].isin(selected_subscriptions)]

        unique_types = Subs_Data["type"].unique()
        type_colors = {type_: f"#{hash(type_) % 16777215:06x}" for type_ in unique_types}
        st.write("### Subscription Wise Geographical Insights")
        m = folium.Map(location=[Subs_Data['location.lat'].mean(), Subs_Data['location.long'].mean()], zoom_start=10)


        def generate_popup_table(row):
            columns_to_show = ["Customer Name", "EPOD Name", "Actual OPERATOR NAME", "t-15_kpi", "No. of Sessions"]
            customer_location_sessions = Subs_Data[
                (Subs_Data['Customer Name'] == row["Customer Name"]) & (Subs_Data['type'] == row["type"]) & (
                            Subs_Data['location.lat'].round(5) == round(row["location.lat"], 5)) & (
                            Subs_Data['location.long'].round(5) == round(row["location.long"], 5))]
            num_sessions = customer_location_sessions.shape[0]
            table_html = "<table style='border-collapse: collapse;'>"
            for col in columns_to_show:
                if col == "No. of Sessions":
                    table_html += f"<tr><td style='border: 1px solid black; padding: 5px;'><strong>{col}</strong></td><td style='border: 1px solid black; padding: 5px;'>{num_sessions}</td></tr>"
                else:
                    table_html += f"<tr><td style='border: 1px solid black; padding: 5px;'><strong>{col}</strong></td><td style='border: 1px solid black; padding: 5px;'>{row[col]}</td></tr>"
            table_html += "</table>"
            return table_html


        unique_combinations = set()
        for index, row in Subs_Data.iterrows():
            location_name = row["type"]
            longitude = round(row["location.long"], 5)
            latitude = round(row["location.lat"], 5)
            color = type_colors[location_name]
            latitude_rounded = round(latitude, 5)
            longitude_rounded = round(longitude, 5)
            combination_key = (row["Customer Name"], row["type"], latitude_rounded, longitude_rounded)
            if combination_key not in unique_combinations:
                unique_combinations.add(combination_key)
                customer_location_sessions = Subs_Data[
                    (Subs_Data['Customer Name'] == row["Customer Name"]) & (Subs_Data['type'] == row["type"]) & (
                                Subs_Data['location.lat'].round(5) == latitude_rounded) & (
                                Subs_Data['location.long'].round(5) == longitude_rounded)]
                num_sessions = customer_location_sessions.shape[0]
                popup_html = f"""<strong>{location_name}</strong><br>Latitude: {latitude_rounded}<br>Longitude: {longitude_rounded}<br>{generate_popup_table(row)}No. of Sessions: {num_sessions}"""
                folium.CircleMarker(location=[latitude, longitude], radius=5,
                                    popup=folium.Popup(popup_html, max_width=400), color=color, fill=True,
                                    fill_color=color).add_to(m)
                print(
                    f"Customer: {row['Customer Name']}, Subscription: {row['type']}, Location: {row['location.lat']}, {row['location.long']}, No. of Sessions: {num_sessions}")

        with col6:
            most_subscribed_type = Subs_Data['type'].value_counts().idxmax()
            st.markdown("Most Subscribed Type")
            st.markdown("<span style='font-size: 25px; line-height: 0.7;'>{}</span>".format(most_subscribed_type),
                        unsafe_allow_html=True)
        with col7:
            least_subscribed_type = Subs_Data['type'].value_counts().idxmin()
            st.markdown("Least Subscribed Type")
            st.markdown("<span style='font-size: 25px; line-height: 0.7;'>{}</span>".format(least_subscribed_type),
                        unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            folium_static(m)
        with col3:
            legend_items = [(type_, color) for type_, color in type_colors.items()]
            split_point = len(legend_items) // 2
            column1 = legend_items[:split_point]
            column2 = legend_items[split_point:]
            col1, col2 = st.columns(2)
            with col1:
                for type_, color in column1:
                    st.markdown(
                        f'<i style="background:{color}; width: 8px; height: 8px; display:inline-block;"></i> {type_}',
                        unsafe_allow_html=True)
            with col2:
                for type_, color in column2:
                    st.markdown(
                        f'<i style="background:{color}; width: 8px; height: 8px; display:inline-block;"></i> {type_}',
                        unsafe_allow_html=True)
    with tab6:
        min_date = pd.to_datetime(final_df['Actual Date']).min().date()
        max_date = pd.to_datetime(final_df['Actual Date']).max().date()
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

        with col1:
            start_date = st.date_input('Start Date', min_value=min_date, max_value=max_date, value=min_date,
                                       key="ops_start_date")
        with col2:
            end_date = st.date_input('End Date', min_value=min_date, max_value=max_date, value=max_date,
                                     key="ops_end_date")

        start_date = pd.to_datetime(start_date).tz_localize(None)
        end_date = pd.to_datetime(end_date).tz_localize(None)

        final_df['Actual Date'] = pd.to_datetime(final_df['Actual Date']).dt.tz_localize(None)

        filtered_df = final_df[(final_df['Actual Date'] >= start_date) & (final_df['Actual Date'] <= end_date)]

        max_sessions = filtered_df.groupby('Actual OPERATOR NAME')['Actual Date'].count().reset_index()
        max_sessions.columns = ['Actual OPERATOR NAME', 'Max Sessions']

        working_days = filtered_df.groupby('Actual OPERATOR NAME')['Actual Date'].nunique().reset_index()
        working_days.columns = ['Actual OPERATOR NAME', 'Working Days']

        grouped_df = filtered_df.groupby(['Actual OPERATOR NAME', 'Customer Location City']).size().reset_index()
        grouped_df.columns = ['Operator', 'City', 'Count']

        cities_to_include = final_df['Customer Location City'].dropna().unique()
        grouped_df = grouped_df[grouped_df['City'].isin(cities_to_include)]

        pivot_df = grouped_df.pivot(index='Operator', columns='City', values='Count').fillna(0)

        figure_width = 1.9
        figure_height = 6
        font_size_heatmap = 5
        font_size_labels = 4

        plt.figure(figsize=(figure_width, figure_height), facecolor='none')

        sns.heatmap(pivot_df, cmap='YlGnBu', annot=True, fmt='g', linewidths=0.5, cbar=False,
                    annot_kws={'fontsize': font_size_heatmap})

        plt.title('Operator v/s Locations', fontsize=8, color='black')
        plt.xlabel('Customer Location City', fontsize=font_size_labels, color='black')
        plt.ylabel('Operator', fontsize=font_size_labels, color='black')

        plt.xticks(rotation=0, ha='center', fontsize=font_size_labels, color='black')
        plt.yticks(fontsize=font_size_labels, color='black')

        with col1:
            st.pyplot(plt, use_container_width=False)

        grouped_df = filtered_df.groupby(['Actual OPERATOR NAME', 'Customer Location City']).size().reset_index()
        grouped_df.columns = ['Operator', 'City', 'Count']

        cities_to_include = final_df['Customer Location City'].dropna().unique()
        grouped_df = grouped_df[grouped_df['City'].isin(cities_to_include)]

        cities = np.append(grouped_df['City'].unique(), "All")

        with col3:
            selected_city = st.selectbox('Select City', cities)

        if selected_city == "All":
            city_df = grouped_df
        else:
            city_df = grouped_df[grouped_df['City'] == selected_city]

        total_sessions = city_df.groupby('Operator')['Count'].sum().reset_index()

        # Rename the column to match with 'working_days' DataFrame
        total_sessions.columns = ['Actual OPERATOR NAME', 'Count']

        merged_df = pd.merge(total_sessions, working_days, on='Actual OPERATOR NAME')

        avg_sessions = pd.DataFrame()
        avg_sessions['Actual OPERATOR NAME'] = merged_df['Actual OPERATOR NAME']
        avg_sessions['Avg. Sessions'] = merged_df['Count'] / merged_df['Working Days']
        avg_sessions['Avg. Sessions'] = avg_sessions['Avg. Sessions'].round(0)

        fig_sessions = go.Figure()
        fig_sessions.add_trace(go.Bar(
            x=total_sessions['Actual OPERATOR NAME'],
            y=total_sessions['Count'],
            name='Total Sessions',
            text=total_sessions['Count'],
            textposition='auto',
            marker=dict(color='yellow'),
            width=0.5
        ))
        fig_sessions.add_trace(go.Bar(
            x=avg_sessions['Actual OPERATOR NAME'],
            y=avg_sessions['Avg. Sessions'],
            name='Average Sessions',
            text=avg_sessions['Avg. Sessions'],
            textposition='auto',
            marker=dict(color='green'),
            width=0.38
        ))
        fig_sessions.update_layout(
            title='Total Sessions and Average Sessions per Operator',
            xaxis=dict(title='Operator'),
            yaxis=dict(title='Count / Average Sessions'),
            margin=dict(l=50, r=50, t=80, b=80),
            legend=dict(yanchor="top", y=1.1, xanchor="left", x=0.01, orientation="h"),
            width=1050,
            height=500
        )

        with col4:
            for i in range(1, 10):
                st.write("\n")
            st.plotly_chart(fig_sessions)

        if selected_city == "All":
            selected_working_days = working_days
        else:
            selected_working_days = working_days[working_days['Actual OPERATOR NAME'].isin(city_df['Operator'])]

        fig_working_days = go.Figure(data=go.Bar(
            x=selected_working_days['Actual OPERATOR NAME'],
            y=selected_working_days['Working Days'],
            marker=dict(color='lightgreen'),
            text=selected_working_days['Working Days']
        ))
        fig_working_days.update_layout(
            title='Working Days per Operator',
            xaxis=dict(title='Operator'),
            yaxis=dict(title='Working Days'),
            margin=dict(l=50, r=50, t=80, b=80),
            width=800,
            height=500
        )

        with col4:
            st.plotly_chart(fig_working_days)

        # Ensure dates are in datetime format and remove timezone info
        heatmap_final_df['Actual Date'] = pd.to_datetime(heatmap_final_df['Actual Date']).dt.tz_localize(None)
        shift_data_df['Actual Date'] = pd.to_datetime(shift_data_df['Actual Date']).dt.tz_localize(None)
        v_mode_final_df['Actual Date'] = pd.to_datetime(v_mode_final_df['Actual Date']).dt.tz_localize(None)
        v_mode_shift_hours_df['Actual Date'] = pd.to_datetime(v_mode_shift_hours_df['Actual Date']).dt.tz_localize(None)

        # Filter dataframes based on the given date range
        heatmap_final_df_filtered = heatmap_final_df[
            (heatmap_final_df['Actual Date'] >= start_date) & (heatmap_final_df['Actual Date'] <= end_date)]
        shift_data_df_filtered = shift_data_df[
            (shift_data_df['Actual Date'] >= start_date) & (shift_data_df['Actual Date'] <= end_date)]
        v_mode_final_df_filtered = v_mode_final_df[
            (v_mode_final_df['Actual Date'] >= start_date) & (v_mode_final_df['Actual Date'] <= end_date)]
        v_mode_shift_hours_df_filtered = v_mode_shift_hours_df[
            (v_mode_shift_hours_df['Actual Date'] >= start_date) & (v_mode_shift_hours_df['Actual Date'] <= end_date)]

        # Calculate D Mode stats
        d_mode_stats = heatmap_final_df_filtered.groupby('Actual OPERATOR NAME').agg(
            Total_Sessions=('Actual Date', 'count'),
            Avg_Sessions=('Actual Date', lambda x: len(x) / x.nunique()),
            Delay_Count=('t-15_kpi', lambda x: (x == 2).sum()),
            D_Mode=('donorVMode', lambda x: (x == 'FALSE').sum())
        ).reset_index()

        # Calculate total shift hours and distinct shift count for D Mode
        d_mode_shift_hours = shift_data_df_filtered.groupby('Actual OPERATOR NAME').agg(
            D_Mode_Shift_Hours=('Shift_Hours', 'sum'),
            D_Mode_Total_Shifts=('shiftUid', 'count')
        ).reset_index()

        # Merge D Mode stats with shift hours
        d_mode_stats = pd.merge(d_mode_stats, d_mode_shift_hours, on='Actual OPERATOR NAME', how='left')

        # Calculate V Mode stats
        v_mode_stats = v_mode_final_df_filtered.groupby('Actual OPERATOR NAME').agg(
            V_Mode=('donorVMode', lambda x: (x == 'TRUE').sum())
        ).reset_index()

        # Calculate total shift hours and distinct shift count for V Mode
        v_mode_shift_hours = v_mode_shift_hours_df_filtered.groupby('Actual OPERATOR NAME').agg(
            V_Mode_Shift_Hours=('Shift_Hours', 'sum'),
            V_Mode_Total_Shifts=('shiftUid', 'count')
        ).reset_index()

        # Merge V Mode stats with shift hours
        v_mode_stats = pd.merge(v_mode_stats, v_mode_shift_hours, on='Actual OPERATOR NAME', how='left')

        # Merge D Mode and V Mode metrics
        operator_stats = pd.merge(d_mode_stats, v_mode_stats, on='Actual OPERATOR NAME', how='outer').fillna(0)

        # Calculate average shift hours per shift considering both D Mode and V Mode
        operator_stats['Total_Shift_Hours'] = operator_stats['D_Mode_Shift_Hours'] + operator_stats[
            'V_Mode_Shift_Hours']
        operator_stats['Total_Shifts'] = operator_stats['D_Mode_Total_Shifts'] + operator_stats['V_Mode_Total_Shifts']
        operator_stats['Avg_Shift_Hours'] = operator_stats['Total_Shift_Hours'] / operator_stats['Total_Shifts']

        # Rename columns for better display
        operator_stats.columns = ['Operator', 'Total Sessions', 'Avg Sessions', 'Delay Count', 'D Mode',
                                  'D_Mode_Shift_Hours', 'D_Mode_Total_Shifts', 'V Mode', 'V_Mode_Shift_Hours',
                                  'V_Mode_Total_Shifts', 'Total Shift Hours', 'Total Shifts', 'Avg Shift Hours']

        # Display the table without D Mode and V Mode Shift Hours columns, but with Total Shifts
        st.markdown("### Operator Statistics Table")
        st.table(operator_stats[
                     ['Operator', 'Total Sessions', 'Avg Sessions', 'D Mode', 'Delay Count', 'V Mode', 'Total Shifts',
                      'Avg Shift Hours']])

        # Export the table to CSV
        csv = operator_stats[
            ['Operator', 'Total Sessions', 'Avg Sessions', 'D Mode', 'Delay Count', 'V Mode', 'Total Shifts',
             'Avg Shift Hours']].to_csv(index=False)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='operator_statistics.csv',
            mime='text/csv',
        )

    with tab7:
        final_df['EPOD Name'] = final_df['EPOD Name'].str.replace('-', '')

        epods = final_df['EPOD Name'].unique()

        # Placeholder for storing uploaded files
        uploaded_files = []


        # Function to process and merge all uploaded files
        def process_files(files):
            dataframes = []
            for file in files:
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)

                # Ensure 'Total' column is ignored
                if 'Total' in df.columns:
                    df = df.drop(columns=['Total'])

                # Remove rows where 'Name' is "Total" or "TOTAL"
                df = df[~df['Name'].str.lower().isin(['total'])]

                dataframes.append(df)

            # Merge all dataframes
            merged_df = pd.concat(dataframes, ignore_index=True)

            # Melt the dataframe to convert dates to rows
            id_vars = ['Name', 'Number', 'Year', 'Make', 'Model', 'Fuel Type', 'Driver Name', 'Driver Number']
            value_vars = merged_df.columns.difference(id_vars)
            vehicle_df_melted = pd.melt(merged_df, id_vars=id_vars, value_vars=value_vars,
                                        var_name='Actual Date', value_name='KM Travelled for Session')

            # Strip any leading/trailing spaces from 'Actual Date'
            vehicle_df_melted['Actual Date'] = vehicle_df_melted['Actual Date'].str.strip()

            # Convert the 'Actual Date' column to datetime
            vehicle_df_melted['Actual Date'] = pd.to_datetime(vehicle_df_melted['Actual Date'], format='%d/%m',
                                                              errors='coerce', dayfirst=True)
            vehicle_df_melted = vehicle_df_melted.dropna(
                subset=['Actual Date'])  # Drop rows where date conversion failed
            vehicle_df_melted['Actual Date'] = vehicle_df_melted['Actual Date'].apply(
                lambda x: x.replace(year=datetime.now().year))

            vehicle_df_melted['KM Travelled for Session'] = vehicle_df_melted['KM Travelled for Session'].replace('-',
                                                                                                                  0).astype(
                float)
            vehicle_df_melted = vehicle_df_melted.dropna(subset=['KM Travelled for Session'])

            # Group by 'Name' and calculate the average KM for each EPod
            average_kms = vehicle_df_melted.groupby('Name')['KM Travelled for Session'].mean().reset_index()
            average_kms['KM Travelled for Session'] = average_kms['KM Travelled for Session'].round(2)

            # Calculate the overall average KM travelled per session
            overall_avg_km = average_kms['KM Travelled for Session'].mean().round(2)

            # Weekly analysis
            vehicle_df_melted['Week'] = vehicle_df_melted['Actual Date'].dt.isocalendar().week
            weekly_kms = vehicle_df_melted.groupby(['Name', 'Week'])['KM Travelled for Session'].mean().reset_index()

            # Monthly analysis
            vehicle_df_melted['Month'] = vehicle_df_melted['Actual Date'].dt.month
            monthly_kms = vehicle_df_melted.groupby(['Name', 'Month'])['KM Travelled for Session'].mean().reset_index()

            return average_kms, overall_avg_km, weekly_kms, monthly_kms


        # File uploader for KM data
        st.markdown("### Upload KM Data (CSV or Excel)")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], accept_multiple_files=True)

        if uploaded_file:
            for file in uploaded_file:
                uploaded_files.append(file)

            average_kms_df, overall_avg_km, weekly_kms_df, monthly_kms_df = process_files(uploaded_files)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # Display the DataFrame with the calculated average KM per EPod
                st.write(average_kms_df)


            with col2:
                # Display the overall average KM travelled per session
                st.metric("Overall Average KM Travelled per Session", overall_avg_km)

            with col3:
                # Plotting the average kilometers per EPod per session
                fig = px.bar(average_kms_df, x='Name', y='KM Travelled for Session',
                             title='Average Kms per EPod per Session')
                st.plotly_chart(fig)


            # Dropdown for weekly and monthly analysis
            analysis_t = st.selectbox('Select Analysis Type', ['Weekly', 'Monthly'], key='analysis_t')

            # Weekly Analysis
            if analysis_t == 'Weekly':
                # Plot weekly KM Travelled
                fig_weekly_kms = px.bar(weekly_kms_df, x='Week', y='KM Travelled for Session', color='Name',
                                        barmode='group', title='Weekly Average KM Travelled Per EPod')
                st.plotly_chart(fig_weekly_kms)

            # Monthly Analysis
            elif analysis_t == 'Monthly':
                # Plot monthly KM Travelled
                fig_monthly_kms = px.bar(monthly_kms_df, x='Month', y='KM Travelled for Session', color='Name',
                                         barmode='group', title='Monthly Average KM Travelled Per EPod')
                st.plotly_chart(fig_monthly_kms)
        else:
            st.markdown("Please upload a valid file to see the data.")







