import pandas as pd
import re

class TimeSeriesParser:
    def __init__(self, filename):
        self.filename = filename

    def parse(self, *country_codes):
        metadata = []
        i_country_code = -1
        with open(self.filename, 'r') as file:
            for line in file:
                elems = [elem.strip('"') for elem in line.strip().split(',')]            
                if len(elems) and elems[0] == "Country Name":
                    metadata = elems
                    i_country_code = [i for i, elem in enumerate(metadata) if elem == "Country Code"][0]
                elif i_country_code != -1 and elems[i_country_code] in country_codes:
                    country = dict(zip(metadata, elems))
                    series_years = {int(k): float(value) for k, value in country.items() if re.match(r"^\d{4}$", k) and value}
                    df = pd.DataFrame({"year": list(series_years.keys()), "popgrowth": list(series_years.values())})
                    yield df
                    
                    