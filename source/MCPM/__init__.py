from os import path

MODULE_PATH = path.abspath(__file__)
for i in range(3):
    MODULE_PATH = path.dirname(MODULE_PATH)

