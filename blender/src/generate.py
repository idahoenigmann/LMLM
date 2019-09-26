import bpy
import math
import random

bpy.ops.object.select_all(action='DESELECT')

PATH = r"C:/Users/Peter/Desktop/Diplomarbeit/renders/{}-{}.jpg"
TARGET_NAME = "Suzanne"
ITERATIONS = 50

YMIN = -8
YMAX = 8
ZMIN = 2
ZMAX = 5

bpy.context.scene.render.image_settings.file_format = 'JPEG'

def center_camera_on_object(name):
    bpy.ops.object.constraint_add(type='TRACK_TO')
    bpy.context.object.constraints["Track To"].name = "Track To"
    bpy.context.object.constraints["Track To"].target = bpy.data.objects[name]
    bpy.context.object.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
    bpy.context.object.constraints["Track To"].up_axis = 'UP_Y'


def render(iteration, part):
    bpy.context.scene.render.filepath = PATH.format(iteration, part)
    bpy.ops.render.render(write_still=True)
    

def add_camera(x, y, z):
    bpy.ops.object.camera_add(align='VIEW', location=(x, y, z), rotation=(0,0,0))
    bpy.context.scene.camera = bpy.context.object
    center_camera_on_object(TARGET_NAME)
    
    
def del_cameras():
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()


for i in range(0, ITERATIONS):
    # Calculate values
    Y1 = random.randint(YMIN, YMAX)
    Y2 = Y1 + 3 if Y1 > 0 else Y1 - 3
    Z = random.randint(ZMIN, ZMAX)
    
    # Add first camera
    add_camera(0, Y1, Z)
    
    render(i, 1)
    del_cameras()
    
    # Add second camera
    add_camera(0, Y2, Z)
    
    render(i, 2)
    del_cameras()
