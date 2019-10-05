import bpy
import math
import random
import os

bpy.ops.object.select_all(action='DESELECT')

RENDERPATH = r"./renders/{}-{}.jpg"
INFOPATH = r"./info/{}-{}.txt"
TARGET_NAME = "Suzanne"
ITERATIONS = 50

TARGET_LOC = bpy.data.objects[TARGET_NAME].location

tx = TARGET_LOC.x
ty = TARGET_LOC.y
tz = TARGET_LOC.z

XMIN = -3
XMAX = 3
YMIN = int(ty - 8)
YMAX = int(ty + 8)
ZMIN = 2
ZMAX = 5

bpy.context.scene.render.image_settings.file_format = 'JPEG'

try:
    os.mkdir("renders/")
    os.mkdir("info/")
except OSError:
    print("Directories available or failed")
    

def center_camera_on_object(name):
    bpy.ops.object.constraint_add(type='TRACK_TO')
    bpy.context.object.constraints["Track To"].name = "Track To"
    bpy.context.object.constraints["Track To"].target = bpy.data.objects[name]
    bpy.context.object.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
    bpy.context.object.constraints["Track To"].up_axis = 'UP_Y'


def render(iteration, part):
    bpy.context.scene.render.filepath = RENDERPATH.format(iteration, part)
    bpy.ops.render.render(write_still=True)
    

def add_camera(x, y, z):
    bpy.ops.object.camera_add(align='VIEW', location=(x, y, z), rotation=(0,0,0))
    bpy.context.scene.camera = bpy.context.object
    center_camera_on_object(TARGET_NAME)
    
    
def del_cameras():
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()
    

def write_dist_info(iteration, part):
    file = open(INFOPATH.format(iteration, part), "w+")
    camera_loc = bpy.context.object.location;
    cx = camera_loc.x
    cy = camera_loc.y
    cz = camera_loc.z
    distance = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2 + (cz - tz) ** 2)
    file.write(str(distance))
    file.close()


for i in range(0, ITERATIONS):
    # Calculate values
    X = random.randint(XMIN, XMAX)
    Y1 = random.randint(YMIN, YMAX)
    Y2 = Y1 + 3 if Y1 > 0 else Y1 - 3
    Z = random.randint(ZMIN, ZMAX)
    
    # Add first camera
    add_camera(X, Y1, Z)
    write_dist_info(i, 1)
    render(i, 1)
    del_cameras()
    
    # Add second camera
    add_camera(X, Y2, Z)
    write_dist_info(i, 2)
    render(i, 2)
    del_cameras()