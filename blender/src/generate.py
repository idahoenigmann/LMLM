import bpy
import math
import random
import os


def make_dirs(renderpath, infopath):
    try:
        os.makedirs(renderpath)
    except OSError:
        print("[RENDERPATH] found or failed to create")
        
    try:
        os.makedirs(infopath)
    except OSError:
        print("[INFOPATH] found or failed to create")
    

def center_camera_on_object(name):
    bpy.ops.object.constraint_add(type='TRACK_TO')
    bpy.context.object.constraints["Track To"].name = "Track To"
    bpy.context.object.constraints["Track To"].target = bpy.data.objects[name]
    bpy.context.object.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
    bpy.context.object.constraints["Track To"].up_axis = 'UP_Y'


def render(path, iteration, part):
    bpy.context.scene.render.filepath = path.format(iteration, part)
    bpy.ops.render.render(write_still=True)
    

def add_camera(x, y, z, tname):
    bpy.ops.object.camera_add(align='VIEW', location=(x, y, z), rotation=(0,0,0))
    bpy.context.scene.camera = bpy.context.object
    center_camera_on_object(tname)
    
    
def del_cameras():
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()
    
    
def get_shortest_dist(tname):
    target = bpy.data.objects[tname]
    target_world = target.matrix_world
    target_msh = target.data
    camera_loc = bpy.context.object.matrix_world.translation;
    dists = [(target_world @ v.co - camera_loc).length for v in target_msh.vertices]
    return min(dists)
    

def write_dist_info(path, tname, iteration, part):
    file = open(path.format(iteration, part), "w+")
    file.write(str(get_shortest_dist(tname)))
    file.close()


# Generate for 1 object
def generate_1(renderpath, infopath, target_name, iterations):
    target_loc = bpy.data.objects[target_name].location

    XMIN = -3
    XMAX = 3
    YMIN = int(target_loc.y - 8)
    YMAX = int(target_loc.y + 8)
    ZMIN = 2
    ZMAX = 5
    
    RENDERPATH = renderpath + target_name
    INFOPATH = infopath + target_name

    RENDERFILEPATH = RENDERPATH + "/{}-{}.jpg"
    INFOFILEPATH = INFOPATH + "/{}-{}.txt"
    
    make_dirs(RENDERPATH, INFOPATH)
    
    for i in range(0, iterations):
        # Calculate values
        X = random.randint(XMIN, XMAX)
        Y1 = random.randint(YMIN, YMAX)
        Y2 = Y1 + 3 if Y1 > 0 else Y1 - 3
        Z = random.randint(ZMIN, ZMAX)
        
        # Add first camera
        add_camera(X, Y1, Z, target_name)
        write_dist_info(INFOFILEPATH, target_name, i, 1)
        render(RENDERFILEPATH, i, 1)
        del_cameras()
        
        # Add second camera
        add_camera(X, Y2, Z, target_name)
        write_dist_info(INFOFILEPATH, target_name, i, 2)
        render(RENDERFILEPATH, i, 2)
        del_cameras()
        

# Generate for list of objects
def generate_n(renderpath, infopath, target_names, iterations):
    for tn in target_names:
        generate_1(renderpath, infopath, tn, iterations)
