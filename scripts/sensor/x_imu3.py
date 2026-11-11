import serial, time, datetime

port = serial.Serial("/dev/ttyACM0", 115200, timeout=0.2)

def send(cmd):
    port.reset_input_buffer()
    port.write((cmd + "\n").encode())
    buf, end = b"", time.time() + 1.5
    while time.time() < end:
        buf += port.read(4096)
    # Replies are JSON lines starting with "{"; everything else is streaming sensor data
    for line in buf.split(b"\n"):
        if line.startswith(b"{"):
            print(line.decode(errors="replace"))

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
send('{"ping":null}')                                        # confirms connection
send(f'{{"time":"{now}"}}')                                  # set clock so file names are correct
send('{"data_logger_automatic_start_stop_enabled":true}')
send('{"data_logger_file_name_prefix":"rec"}')
# Optional: log at full 400 Hz instead of the default 50 Hz
# send('{"inertial_message_rate_divisor":1}')
# send('{"ahrs_message_rate_divisor":1}')
send('{"apply":null}')
send('{"save":null}')
send('{"data_logger_automatic_start_stop_enabled":null}')   # read back to verify