import bpy
gen = bpy.data.texts["generate.py"].as_module()


bpy.ops.object.select_all(action='DESELECT')
bpy.context.scene.render.image_settings.file_format = 'JPEG'

TARGET_NAMES = ["Vase.XS", "Vase.S", "Vase.M", "Vase.L", "Vase.XL"]
RENDERPATH = r"H:/DAData/renders/"
INFOPATH = r"H:/DAData/info/"

gen.generate_n(RENDERPATH, INFOPATH, TARGET_NAMES, 50)
