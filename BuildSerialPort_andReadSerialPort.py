# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:25:39 2020

@author: Mingming
"""

import serial

# Build the serial port connection
ser = serial.Serial(port='COM5',    # comport number on your computer 
                    baudrate=230400,timeout=2,  # bauderate your device is running at.
                    bytesize=serial.EIGHTBITS,
                    parity  =serial.PARITY_NONE,  # parity is a method of detecting error in transmission, here no parity bit will be sent
                    stopbits=serial.STOPBITS_ONE)

number_bytes = 1
data = ser.read(number_bytes)  # read the given number of bytes of data from the serial port