import predict_angle_algorithm as pa
import numpy as np
#test
print('[0,1]: ', pa.vec2mil([0, 1]))
print('[1,1]: ', pa.vec2mil([1, 1]))
print('[1,0]: ', pa.vec2mil([1, 0]))
print('[1,-1]: ', pa.vec2mil([1, -1]))
print('[0,-1]: ', pa.vec2mil([0, -1]))
print('[-1,-1]: ', pa.vec2mil([-1, -1]))
print('[-1,0]: ', pa.vec2mil([-1, 0]))
print('[-1,1]: ', pa.vec2mil([-1, 1]))
