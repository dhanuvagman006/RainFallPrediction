import csv
import json
import sys
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

# NASA POWER API config for Dakshina Kannada district (approx. centroid).
LATITUDE = 12.87
LONGITUDE = 74.88
START_DATE = "20000101"
END_DATE = "20241231"
# Daily parameters for rainfall predictors (see NASA POWER docs for units).
PARAMETERS = [
    "PRECTOTCORR",
    "PS",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "WS2M",
    "WD2M",
    "ALLSKY_SFC_SW_DWN",
]
OUTPUT_CSV = "TrainModels/dakshina_kannada_rainfall_daily_2000_2024.csv"


def build_url(parameters):
    base = "https://power.larc.nasa.gov/api/temporal/daily/point"
    query = urlencode(
        {
            "parameters": ",".join(parameters),
            "community": "AG",
            "longitude": LONGITUDE,
            "latitude": LATITUDE,
            "start": START_DATE,
            "end": END_DATE,
            "format": "JSON",
        }
    )
    return f"{base}?{query}"


def fetch_payload(url):
    with urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_parameter(param):
    url = build_url([param])
    try:
        payload = fetch_payload(url)
    except HTTPError as exc:
        if exc.code == 422:
            print(f"Skipping parameter '{param}': not supported for this endpoint.")
            return None
        print(f"HTTP error for '{param}': {exc}")
        sys.exit(1)
    except URLError as exc:
        print(f"URL error for '{param}': {exc}")
        sys.exit(1)

    try:
        return payload["properties"]["parameter"][param]
    except KeyError:
        # Print a short payload preview to diagnose API response structure.
        preview = json.dumps(payload, indent=2)
        print(f"Unexpected response format for '{param}'. Payload preview (first 1200 chars):")
        print(preview[:1200])
        sys.exit(1)


def to_rows(data_by_param):
    if not data_by_param:
        print("No parameters were returned by the API.")
        sys.exit(1)

    primary_param = next(iter(data_by_param))
    date_keys = sorted(data_by_param[primary_param].keys())
    rows = []
    for date_str in date_keys:
        # Date format from API is YYYYMMDD
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        row_values = [date_obj.year, date_obj.month, date_obj.day]
        for param in PARAMETERS:
            data = data_by_param.get(param)
            row_values.append("" if data is None else data.get(date_str, ""))
        rows.append(row_values)

    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["year", "month", "day"] + [p.lower() for p in PARAMETERS]
        writer.writerow(header)
        writer.writerows(rows)


def main():
    data_by_param = {}
    for param in PARAMETERS:
        print(f"Fetching parameter: {param}")
        data = fetch_parameter(param)
        if data is not None:
            data_by_param[param] = data

    rows = to_rows(data_by_param)
    write_csv(rows, OUTPUT_CSV)
    print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
