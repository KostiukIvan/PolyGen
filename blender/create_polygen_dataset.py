import os
import bpy
import numpy as np
from pathlib import Path
from random import randrange
import math
import random

def discard_exported_models(dir, obj_list, export_path):
    path = os.path.join(export_path, dir)
    if os.path.exists(path) == False:
        return obj_list
    
    exported = [d for r, d, f in os.walk(path)][0]
    # print(exported)
    # exported = [f for f in exported if f.endswith('.txt')]
    # exported = [f[0:-len('.txt')] for f in exported]
    # exported = set([f[0:-len('_mesh_data')] for f in exported if f.endswith('_mesh_data')])

    # return [(p, o) for (p, o) in obj_list if o not in exported]
    # print(exported)
    return [(p, o) for (p, o) in obj_list if o not in exported]

def get_models_directories(path):
    obj_list = []
    dir_list = sorted(os.listdir(path))
    dir_list = [os.path.join(path, dir) for dir in dir_list if os.path.isdir(os.path.join(path, dir))]
    for dir in dir_list:
        for r1, d1, f1 in os.walk(os.path.join(dir, 'models')):
            for file in f1:
                if file.endswith('.obj'):
                    full_path = os.path.join(dir, 'models', file)
                    path = Path(full_path)
                    obj_name = str(Path(os.path.relpath(full_path, path.parent.parent.parent)).parent.parent)
                    obj_list.append((full_path, obj_name))
    
    return obj_list

def free_memory():
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

def scale_diagonal_to_1(obj):
    # scale so that diagonal axis has length 1
    max_len = max(obj.dimensions)
    scaling_factor = max_len * math.sqrt(3)

    # obj.dimensions = np.array([1, 1, 1]) * (math.sqrt(3) / 3)
    obj.dimensions = obj.dimensions / scaling_factor
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

if __name__ == "__main__":
    path_to_obj_dir = 'D:\\ShapeNetCore.v2'
    export_path = 'D:\\polygen_exports\\'

    bpy.ops.object.delete({"selected_objects": bpy.context.scene.objects})

    # dir_list = ['02691156']
    dir_list = [dir for dir in os.listdir(path_to_obj_dir) if os.path.isdir(os.path.join(path_to_obj_dir, dir))]
    dir_list = [(os.path.join(path_to_obj_dir, dir), dir) for dir in dir_list]

    for j in range(len(dir_list)):
        dir = dir_list[j]
        obj_list = get_models_directories(dir[0])
        obj_list = discard_exported_models(dir[1], obj_list, export_path) #todo: adjust
        Path(os.path.join(export_path, dir[1])).mkdir(parents=True, exist_ok=True)
        for i in range(len(obj_list)):
            file, obj_name = obj_list[i]
            bpy.ops.import_scene.obj(filepath = file, use_split_objects=False, use_image_search=False)
            obj = bpy.data.objects[0]
            obj.name = obj_name
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)

            verts_len = len(obj.data.vertices)
            if verts_len > 800 and verts_len <= 8000:
                bpy.ops.object.modifier_add(type='DECIMATE')
                obj.modifiers['Decimate'].decimate_type = 'COLLAPSE'
                obj.modifiers['Decimate'].ratio = 600 / verts_len
                bpy.ops.object.modifier_apply(modifier='Decimate')
                verts_len = len(obj.data.vertices)
                if verts_len > 800:
                    print('could not decimate', verts_len)
            
            
            if verts_len <= 800:
                Path(os.path.join(export_path, dir[1], obj_name)).mkdir(parents=True, exist_ok=True)
                for aug_index in range(50):
                    # bpy.ops.mesh.select_all(action='DESELECT')
                    obj.select_set(True)
                    bpy.ops.object.duplicate_move()
                    obj.select_set(False)
                    duplicated_obj = bpy.context.selected_objects[0]
                    duplicated_obj.select_set(True)
                    bpy.context.view_layer.objects.active = duplicated_obj

                    duplicated_obj.scale = [random.uniform(0.75, 1.25) for _ in range(3)]
                    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                    scale_diagonal_to_1(duplicated_obj)

                    bpy.ops.object.modifier_add(type='DISPLACE')
                    duplicated_obj.modifiers['Displace'].strength = 0.01
                    duplicated_obj.modifiers['Displace'].mid_level = random.uniform(0.5, 1.0)
                    bpy.ops.object.modifier_apply(modifier='Displace')

                    bpy.ops.object.modifier_add(type='SIMPLE_DEFORM')
                    duplicated_obj.modifiers['SimpleDeform'].deform_method = 'BEND'
                    duplicated_obj.modifiers['SimpleDeform'].deform_axis = 'Z'
                    duplicated_obj.modifiers['SimpleDeform'].angle = math.radians(random.uniform(-15, 15))
                    bpy.ops.object.modifier_apply(modifier='SimpleDeform')

                    bpy.ops.object.modifier_add(type='SIMPLE_DEFORM')
                    duplicated_obj.modifiers['SimpleDeform'].deform_method = 'TAPER'
                    duplicated_obj.modifiers['SimpleDeform'].factor = random.uniform(-0.1, 0.1)
                    bpy.ops.object.modifier_apply(modifier='SimpleDeform')

                    bpy.ops.object.modifier_add(type='SIMPLE_DEFORM')
                    duplicated_obj.modifiers['SimpleDeform'].deform_method = 'STRETCH'
                    duplicated_obj.modifiers['SimpleDeform'].deform_axis = random.choice(['X', 'Y', 'Z'])
                    duplicated_obj.modifiers['SimpleDeform'].factor = random.uniform(-0.1, 0.1)
                    bpy.ops.object.modifier_apply(modifier='SimpleDeform')

                    bpy.ops.object.modifier_add(type='DECIMATE')
                    duplicated_obj.modifiers['Decimate'].decimate_type = 'DISSOLVE'
                    duplicated_obj.modifiers['Decimate'].angle_limit = math.radians(random.uniform(1, 20))
                    bpy.ops.object.modifier_apply(modifier='Decimate')
                    scale_diagonal_to_1(duplicated_obj)
                    
                    out_dir = os.path.join(export_path, dir[1], obj_name, obj_name + '_' + str(aug_index) + '.obj')
                    bpy.ops.export_scene.obj(filepath=out_dir, use_selection=True, use_normals=False, use_uvs=False, use_materials=False)

                    # bpy.ops.object.delete({"selected_objects": bpy.context.scene.objects})
                    bpy.ops.object.delete()
            bpy.ops.object.delete({"selected_objects": bpy.context.scene.objects})
            free_memory()
            print('export progress:', (i + 1), 'of', len(obj_list), 'dir', j + 1, 'of', len(dir_list))