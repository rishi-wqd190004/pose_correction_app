import subprocess

# Run the 'adb shell getevent -S' command to get the device event list
getevent_cmd = ['adb', 'shell', 'getevent', '-S']
result = subprocess.run(getevent_cmd, capture_output=True, text=True)

# Extract the gyroscope event ID from the output
gyro_event_id = None
output = result.stdout
for line in output.splitlines():
    if 'Gyroscope' in line:
        parts = line.split()
        gyro_event_id = parts[0]
        break

# Run the 'adb shell getevent -t' command to get the gyroscope readings
gyro_cmd = ['adb', 'shell', 'getevent', '-t', gyro_event_id]
result = subprocess.Popen(gyro_cmd, stdout=subprocess.PIPE)

# Extract the gyroscope readings from the output
gyro_reading = None
for line in iter(result.stdout.readline, ''):
    if 'ABS_RX' in line:
        parts = line.split()
        gyro_reading = [int(p, 16) for p in parts[2:5]]
        break

# Print the gyroscope reading
print(gyro_reading)
