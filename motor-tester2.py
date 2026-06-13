from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16, address=0x40)

BASE_CHANNEL = 3   # change this if your base is on a different channel

kit.servo[BASE_CHANNEL].set_pulse_width_range(500, 2500)
kit.servo[BASE_CHANNEL].actuation_range = 180

print("Sending base servo to 90 degrees / center")
kit.servo[BASE_CHANNEL].angle = 90

time.sleep(2)

print("Now attach the arm/horn so this physical position is your mean position.")
