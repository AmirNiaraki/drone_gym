# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 09:43:09 2023

@author: amire
"""

from math import degrees, cos, tan, acos, sqrt

action=(10,0)
wind=(-10,0)

theta=degrees(acos((action[0]*wind[0]+action[1]*wind[1])/(sqrt(action[0]**2+action[1]**2)*sqrt(wind[0]**2+wind[1]**2))))
print(theta)



