EXTERNAL_PERIPH_BASE=$(SAPI_BASE)/external_peripherals

# Display ---------------------------------------------------------------------

SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/display/fonts/greek_chars_5x7/src/*.c)
SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/display/fonts/icon_chars_5x7/src/*.c)
SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/display/lcd/src/*.c)
SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/display/led_segments/7segment/src/*.c)

# IMU -------------------------------------------------------------------------

SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/imu/mpu60X0/src/*.c)
SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/imu/mpu9250/src/*.c)

# Keypad ----------------------------------------------------------------------

SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/keypad/src/*.c)

# LED RGB ---------------------------------------------------------------------

SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/led_rgb/src/*.c)

# MAGNETOMETER ----------------------------------------------------------------

SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/magnetometer/hmc5883l/src/*.c)
SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/magnetometer/qmc5883l/src/*.c)

# MEMORY ----------------------------------------------------------------------

SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/memory/eeprom/src/*.c)

# MOTOR -----------------------------------------------------------------------

SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/motor/servo/src/*.c)

# Temperature and Humidity ----------------------------------------------------

SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/temperature_humudity/dht11/src/*.c)

# Ultrasonlic -----------------------------------------------------------------

SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/ultrasonic/hcsr04/src/*.c)

# Wi-Fi -----------------------------------------------------------------------

SRC += $(wildcard $(EXTERNAL_PERIPH_BASE)/wifi/esp8266_at/src/*.c)
