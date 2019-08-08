import argparse
import math

binarized = False
display = False
temperature_init = 0.15
temperature_final = 60
maxvalue = 0
kernel_size = 3
temperature = temperature_init
temperature_update_mult = (temperature_final / temperature_init) ** (1/(500 * 200))
temperature_update = (temperature_final -temperature_init) / (500 * 200)

def update_temperature():
    global temperature
    temperature += temperature_update

def update_temperature_mult():
    global temperature
    temperature *= temperature_update_mult

def set_binarized(new_value):
    global binarized
    binarized = new_value
