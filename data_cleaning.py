"""
Data Cleaning Script for Enron Email Dataset

This script processes the Enron email dataset by extracting relevant fields 
from email messages, cleaning the data, and saving the cleaned dataset to a CSV file.
"""

import numpy as np
import pandas as pd
import email
from dateutil import parser

# Load the dataset
df = pd.read_csv("./emails.csv")

# Function to extract specific fields from email messages
def get_field(field, messages):
    """
    Extracts a specific field from a list of email messages.

    Args:
        field (str): The email header field to extract (e.g., 'Date', 'Subject').
        messages (pd.Series): A pandas Series containing raw email messages.

    Returns:
        list: A list of extracted field values.
    """
    column = []
    for message in messages:
        e = email.message_from_string(message)
        column.append(e.get(field))
    return column

# Extract relevant fields from the email messages
df['date'] = get_field("Date", df['message'])
df['subject'] = get_field("Subject", df['message'])
df['X-From'] = get_field("X-From", df['message'])
df['X-To'] = get_field("X-To", df['message'])

# Function to extract the body of email messages
def extract_body(messages):
    """
    Extracts the body content from a list of email messages.

    Args:
        messages (pd.Series): A pandas Series containing raw email messages.

    Returns:
        list: A list of email body contents.
    """
    column = []
    for message in messages:
        e = email.message_from_string(message)
        column.append(e.get_payload())
    return column

# Extract the body of the email messages
df['body'] = extract_body(df['message'])

# Function to convert date strings to a uniform datetime format
def change_date_format(dates):
    """
    Converts date strings to a uniform format (DD-MM-YYYY HH:MM:SS).

    Args:
        dates (pd.Series): A pandas Series containing date strings.

    Returns:
        list: A list of formatted date strings.
    """
    column = []
    for date in dates:
        column.append(parser.parse(date).strftime("%d-%m-%Y %H:%M:%S"))
    return column

# Standardize the date format
df['date'] = change_date_format(df['date'])

# Function to replace empty strings with NaN
def replace_empty_with_nan(values):
    """
    Replaces empty strings in a list with NaN.

    Args:
        values (pd.Series): A pandas Series containing string values.

    Returns:
        list: A list with empty strings replaced by NaN.
    """
    column = []
    for val in values:
        column.append(np.nan if val == "" else val)
    return column

# Replace empty strings with NaN in relevant columns
df['subject'] = replace_empty_with_nan(df['subject'])
df['X-To'] = replace_empty_with_nan(df['X-To'])
df['X-From'] = replace_empty_with_nan(df['X-From'])

# Drop rows with missing values
df.dropna(axis=0, inplace=True)

# Drop unnecessary columns
cols_to_drop = ['file', 'message']
df.drop(cols_to_drop, axis=1, inplace=True)

# Save the cleaned dataset to a CSV file
df.to_csv("cleaned_data.csv", index=False)
