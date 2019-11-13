"""
Useful links:
https://pyowm.readthedocs.io/en/latest/usage-examples-v2/weather-api-usage-examples.html
https://pyowm.readthedocs.io/en/latest/#pyowm-v2-usage-documentation
https://github.com/csparpa/pyowm
"""
import pyowm
owm = pyowm.OWM('fd1014d3d0bd73d600e3d4445f6912d0')
location = owm.weather_at_place('Auckland, NZ')
weather = location.get_weather()

print(weather.get_temperature('celsius')['temp'])
print(weather.get_wind())                  # {'speed': 4.6, 'deg': 330}
print(weather.get_humidity())            # 87
print(weather.get_sunrise_time(timeformat='iso')) # Prints time in GMT timezone
print(weather.get_sunset_time(timeformat='iso')) # Prints time in GMT timezone
print(weather.get_detailed_status())
print(weather.get_weather_icon_url())
print(weather.get_pressure())
print(weather.get_rain())
print(weather.get_reference_time(timeformat='iso'))