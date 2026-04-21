import serial

ser = serial.Serial("COM10", timeout=1)
ser.write(b"v")
print(ser.readline())