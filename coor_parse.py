def dms_to_decimal(degrees, minutes, seconds, direction):
    """
    Convert DMS (Degrees, Minutes, Seconds) to Decimal Degrees.
    
    Parameters:
    degrees (int): Degrees component.
    minutes (int): Minutes component.
    seconds (float): Seconds component.
    direction (str): Direction component ('N', 'S', 'E', 'W').
    
    Returns:
    float: Decimal degrees.
    """
    decimal = degrees + minutes / 60 + seconds / 3600
    
    if direction in ['S', 'W']:
        decimal *= -1
        
    return decimal

def parse_dms(dms_str):
    """
    Parse a DMS string to degrees, minutes, seconds, and direction.
    
    Parameters:
    dms_str (str): DMS string (e.g., "5°45'40\"N").
    
    Returns:
    tuple: (degrees, minutes, seconds, direction)
    """
    import re
    pattern = re.compile(r"(\d+)[°](\d+)['](\d+(?:\.\d+)?)[\"]?([NSEW])")
    match = pattern.match(dms_str)
    if not match:
        raise ValueError("Invalid DMS format")
    
    degrees = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    direction = match.group(4)
    
    return degrees, minutes, seconds, direction

# Example usage
# lat_dms = "5°45'0\"N"
# lon_dms = "0°14'25\"W"

# lat_deg, lat_min, lat_sec, lat_dir = parse_dms(lat_dms)
# lon_deg, lon_min, lon_sec, lon_dir = parse_dms(lon_dms)

# latitude = dms_to_decimal(lat_deg, lat_min, lat_sec, lat_dir)
# longitude = dms_to_decimal(lon_deg, lon_min, lon_sec, lon_dir)

# print(f"Latitude: {latitude}, Longitude: {longitude}")
