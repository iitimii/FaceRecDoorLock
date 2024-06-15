from RPLCD.i2c import CharLCD
import time

lcd = CharLCD(i2c_expander='PCF8574', address=0x3F, port=1, cols=16, rows=2, dotsize=8, backlight_enabled=True)

lcd.clear()
lcd.write_string("Security Door Lock")
counter = 0

while True:
    print(f"Counter: {counter}")
    lcd.clear()
    lcd.write_string(f"Counter: {counter}")
    counter += 1
    time.sleep(1)
