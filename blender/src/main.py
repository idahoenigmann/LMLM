import bpy
gen = bpy.data.texts["generate.py"].as_module()


bpy.ops.object.select_all(action='DESELECT')
bpy.context.scene.render.image_settings.file_format = 'JPEG'

TARGET_NAMES = ["Suzanne.XS", "Suzanne.S", "Suzanne.M", "Suzanne.L", "Suzanne.XL"]
RENDERPATH = r"H:/DAData/renders/"
INFOPATH = r"H:/DAData/info/"

gen.generate_n(RENDERPATH, INFOPATH, TARGET_NAMES, 50)
