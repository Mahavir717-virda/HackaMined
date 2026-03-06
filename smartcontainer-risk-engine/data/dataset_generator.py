import random
import csv
from datetime import datetime, timedelta

origin_countries = ["BE"]
destination_countries = ["UG"]

ports = ["PORT_20","PORT_30","PORT_40"]
hs_codes = ["440890","620822","690722","940350","851712","071080"]
shipping_lines = ["LINE_MODE_10","LINE_MODE_40"]

def random_time():
    return (datetime.min + timedelta(seconds=random.randint(0,86399))).time()

def generate_normal(container_id):

    declared_weight = round(random.uniform(100,10000),2)
    measured_weight = declared_weight + random.uniform(-5,5)

    return [
        container_id,
        "2020-01-01",
        str(random_time()),
        "Import",
        "BE",
        random.choice(ports),
        "UG",
        random.choice(hs_codes),
        f"IMP{random.randint(100,999)}",
        f"EXP{random.randint(100,999)}",
        round(random.uniform(10000,500000),2),
        declared_weight,
        measured_weight,
        random.choice(shipping_lines),
        round(random.uniform(10,80),2),
        0
    ]

def generate_anomaly(container_id):

    case = random.choice([
        "weight_mismatch",
        "huge_value",
        "low_value",
        "negative_value",
        "huge_dwell"
    ])

    declared_weight = round(random.uniform(100,10000),2)
    measured_weight = declared_weight

    declared_value = round(random.uniform(10000,500000),2)
    dwell = round(random.uniform(10,80),2)

    if case == "weight_mismatch":
        measured_weight = declared_weight * random.uniform(2,5)

    if case == "huge_value":
        declared_value = random.uniform(1e8,1e10)

    if case == "low_value":
        declared_value = random.uniform(0,5)

    if case == "negative_value":
        declared_value = -random.uniform(1000,10000)

    if case == "huge_dwell":
        dwell = random.uniform(300,1000)

    return [
        container_id,
        "2020-01-01",
        str(random_time()),
        "Import",
        "BE",
        random.choice(ports),
        "UG",
        random.choice(hs_codes),
        f"IMP{random.randint(100,999)}",
        f"EXP{random.randint(100,999)}",
        declared_value,
        declared_weight,
        measured_weight,
        random.choice(shipping_lines),
        dwell,
        1
    ]


rows = []
total_rows = 10000

for i in range(total_rows):

    container_id = 10000000 + i

    if random.random() < 0.15:
        rows.append(generate_anomaly(container_id))
    else:
        rows.append(generate_normal(container_id))


with open("shipping_dataset.csv","w",newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Container_ID",
        "Declaration_Date",
        "Declaration_Time",
        "Trade_Regime",
        "Origin_Country",
        "Destination_Port",
        "Destination_Country",
        "HS_Code",
        "Importer_ID",
        "Exporter_ID",
        "Declared_Value",
        "Declared_Weight",
        "Measured_Weight",
        "Shipping_Line",
        "Dwell_Time_Hours",
        "Risk_Flag"
    ])

    writer.writerows(rows)

print("Dataset generated: shipping_dataset.csv")