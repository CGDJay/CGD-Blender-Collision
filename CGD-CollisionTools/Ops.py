import bpy
from bpy.types import SpaceView3D, Operator
from itertools import chain
from math import radians
from mathutils import Matrix, Vector
import bmesh
import re
import numpy as np
import gpu
from gpu_extras.batch import batch_for_shader

#deprecated originaly used to draw colored wireframe
# def drawColWire():

#     for obj in bpy.data.objects:

#         if 'Collision' in obj and obj.visible_get(view_layer=bpy.context.view_layer) == True:
#             loc = obj.matrix_world

#             mesh = obj.data
#             depsgraph = bpy.context.evaluated_depsgraph_get()
#             eval = obj.evaluated_get(depsgraph)

#             coords = [loc @ v.co  for v in eval.data.vertices]
#             indices = [(e.vertices[0], e.vertices[1]) for e in eval.data.edges]

#             shader = gpu.shader.from_builtin('UNIFORM_COLOR')
#             batch = batch_for_shader(shader, 'LINES', {"pos": coords}, indices=indices)

#             shader.bind()
#             shader.uniform_float("color", (0, 1, 0, 1))

#             batch.draw(shader)

#         else:
#             pass
#     return


def get_BoundBoxCorners(object,AAB):
    obj = object
    AAB=AAB
    if obj.mode == 'EDIT':
        print('in edit mode')
        bm = bmesh.from_edit_mesh(obj.data)
        verts = [v.co for v in bm.verts if v.select]
    else:
        depsgraph = bpy.context.evaluated_depsgraph_get()

        bm = bmesh.new()

        bm.from_object(obj,depsgraph)
        
        bm.verts.ensure_lookup_table()

        verts = [v.co for v in bm.verts]

    points = np.asarray(verts)
    cov = np.cov(points, y=None, rowvar=0, bias=1)
    v, vect = np.linalg.eig(cov)
    tvect = np.transpose(vect)


    if AAB == True:
        co_min = np.min(points, axis=0)
        co_max = np.max(points, axis=0)
    else:
        points_r = np.dot(points, np.linalg.inv(tvect))
        co_min = np.min(points_r, axis=0)
        co_max = np.max(points_r, axis=0)

      
    xmin, xmax = co_min[0], co_max[0]
    ymin, ymax = co_min[1], co_max[1]
    zmin, zmax = co_min[2], co_max[2]

    xdif = (xmax - xmin) * 0.5
    ydif = (ymax - ymin) * 0.5
    zdif = (zmax - zmin) * 0.5

    cx = xmin + xdif
    cy = ymin + ydif
    cz = zmin + zdif

    if AAB==True:
        corners = np.array([
            [cx - xdif, cy - ydif, cz - zdif],
            [cx - xdif, cy + ydif, cz - zdif],
            [cx - xdif, cy + ydif, cz + zdif],
            [cx - xdif, cy - ydif, cz + zdif],
            [cx + xdif, cy + ydif, cz + zdif],
            [cx + xdif, cy + ydif, cz - zdif],
            [cx + xdif, cy - ydif, cz + zdif],
            [cx + xdif, cy - ydif, cz - zdif],
        ])
        object_BB = corners
    else:
        
        corners = np.array([
            [cx - xdif, cy - ydif, cz - zdif],
            [cx - xdif, cy + ydif, cz - zdif],
            [cx - xdif, cy + ydif, cz + zdif],
            [cx - xdif, cy - ydif, cz + zdif],
            [cx + xdif, cy + ydif, cz + zdif],
            [cx + xdif, cy + ydif, cz - zdif],
            [cx + xdif, cy - ydif, cz + zdif],
            [cx + xdif, cy - ydif, cz - zdif],
        ])

        object_BB = np.dot(corners, tvect)

    if AAB==True:
        center = [cx, cy, cz]
    else:
        center = np.dot([cx, cy, cz], tvect)
    corners = [Vector((el[0], el[1], el[2])) for el in corners]

    A = np.array(verts)
    M = np.mean(A.T, axis=1)  # Find mean
    C = A - M  # Center around mean
    V = np.cov(C.T)  # Calculate covariance matrix of centered matrix
    U, s, Vh = np.linalg.svd(V)  # Singular value decomposition

    
    if AAB==True:
        axis, _ = Vector((0, 0, 0)), Vector(M)

    else:
        axis, _ = Vector(U[:, 0]), Vector(M)
    return (corners[1], corners[3], center, axis)


def get_BoundBox(object,AAB):
    '''Generates a object aligned bounding box 
    to be used to generate collision meshes'''
    obj = object
    AAB=AAB


    if obj.mode == 'EDIT':
        print('in edit mode')
        bm = bmesh.from_edit_mesh(obj.data)
        verts = [v.co for v in bm.verts if v.select]

    else:

        depsgraph = bpy.context.evaluated_depsgraph_get()

        bm = bmesh.new()

        bm.from_object(obj,depsgraph)
        
        bm.verts.ensure_lookup_table()

        verts = [v.co for v in bm.verts]


    points = np.asarray(verts)
    cov = np.cov(points, y=None, rowvar=0, bias=1)
    v, vect = np.linalg.eig(cov)
    tvect = np.transpose(vect)


    if AAB == True:
        co_min = np.min(points, axis=0)
        co_max = np.max(points, axis=0)
    else:
        points_r = np.dot(points, np.linalg.inv(tvect))
        co_min = np.min(points_r, axis=0)
        co_max = np.max(points_r, axis=0)

      
    xmin, xmax = co_min[0], co_max[0]
    ymin, ymax = co_min[1], co_max[1]
    zmin, zmax = co_min[2], co_max[2]

    xdif = (xmax - xmin) * 0.5
    ydif = (ymax - ymin) * 0.5
    zdif = (zmax - zmin) * 0.5

    cx = xmin + xdif
    cy = ymin + ydif
    cz = zmin + zdif

    if AAB==True:
        corners = np.array([
            [cx - xdif, cy - ydif, cz - zdif],
            [cx - xdif, cy + ydif, cz - zdif],
            [cx - xdif, cy + ydif, cz + zdif],
            [cx - xdif, cy - ydif, cz + zdif],
            [cx + xdif, cy + ydif, cz + zdif],
            [cx + xdif, cy + ydif, cz - zdif],
            [cx + xdif, cy - ydif, cz + zdif],
            [cx + xdif, cy - ydif, cz - zdif],
        ])
        object_BB = corners
    else:
        
        corners = np.array([
            [cx - xdif, cy - ydif, cz - zdif],
            [cx - xdif, cy + ydif, cz - zdif],
            [cx - xdif, cy + ydif, cz + zdif],
            [cx - xdif, cy - ydif, cz + zdif],
            [cx + xdif, cy + ydif, cz + zdif],
            [cx + xdif, cy + ydif, cz - zdif],
            [cx + xdif, cy - ydif, cz + zdif],
            [cx + xdif, cy - ydif, cz - zdif],
        ])

        object_BB = np.dot(corners, tvect)

    return (object_BB)


def get_parent_name(object):
    '''retrieve parent name of the object'''
    if object.parent:
        name = object.name
        try:
            parent = object.parent
            name = parent.name
            print("Parent Name:")
            print(name)
        except BaseException:
            print("no parent object")
    else:
        name = object.name

    return (name)


def get_collection(context, name, allow_duplicate=False, clean=True):
    '''Ensures that a collection with the given name exists in the scene'''
    collection = None
    collections = [context.scene.collection]

    while collections:
        cl = collections.pop()
        if cl.name == name or allow_duplicate and re.match(rf"^{name}(?:\.\d\d\d)?$", cl.name):
            collection = cl
            break
        collections.extend(cl.children)
        cl = None

    if not collection:
        collection = bpy.data.collections.new(name)

    elif clean:
        for obj in collection.objects[:]:
            collection.objects.unlink(obj)

    if name not in context.scene.collection.children:
        context.scene.collection.children.link(collection)

    return collection


def create_col_object_from_bm(self, context, obj, bm, prefix=None):
    '''creates collision mesh from a bmesh by 
    cleaning up data and assigning correct settings'''

    name = find_free_col_name(prefix, obj.name)
    data = bpy.data.meshes.new(name)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(data)

    col_obj = bpy.data.objects.new(name, data)
    col_obj.matrix_world = obj.matrix_world
    col_obj.show_wire = True
    col_obj.display_type = 'WIRE'
    col_obj.color = (0, 1, 0, 1)
    col_obj.show_in_front = True
    col_obj.display.show_shadows = False
    col_obj.location = col_obj.location - obj.location
    col_obj.parent = obj
    col_obj['Collision'] = True

    # remove vertex group and shapekeys
    col_obj.vertex_groups.clear()
    col_obj.shape_key_clear()

    collection = get_collection(
        context, "Collision", allow_duplicate=True, clean=False)
    collection.color_tag = 'COLOR_04'

    for coll in col_obj.users_collection:
        # Unlink the object
        coll.objects.unlink(col_obj)

    collection.objects.link(col_obj)

    


    return col_obj


def find_free_col_name(prefix, name):
    '''Finds free collision name for object'''

    n = 1
    while True:
        if n >= 10:
            col_name = f"{prefix}_{name}_{n}"
            n += 1
        else:

            col_name = f"{prefix}_{name}_0{n}"
            n += 1

        if col_name not in bpy.context.scene.objects:
            break

    return col_name


class TempModifier:
    '''Convenient modifier wrapper to use in a `with` block
     to be automatically applied at the end'''
    # sourced from : https://github.com/greisane/gret/blob/85290f8ccb7122cf4cbd4dc94236f30aad2c38f9/mesh/helpers.py

    def __init__(self, obj, type):
        self.obj = obj
        self.type = type

    def __enter__(self,):
        self.saved_mode = bpy.context.mode
        if bpy.context.mode == 'EDIT_MESH':
            bpy.ops.object.editmode_toggle()

        self.modifier = self.obj.modifiers.new(type=self.type, name="")
        # Move first to avoid the warning on applying
        context_override= bpy.context.copy()
        context_override["object"] = self.obj
        with bpy.context.temp_override(**context_override):
            bpy.ops.object.modifier_move_to_index(
                modifier=self.modifier.name, index=0)

        return self.modifier

    def __exit__(self, exc_type, exc_value, exc_traceback):
        context_override=  bpy.context.copy()
        context_override["object"] = self.obj

        with bpy.context.temp_override(**context_override):
            bpy.ops.object.modifier_apply( modifier=self.modifier.name)

        if self.saved_mode == 'EDIT_MESH':
            bpy.ops.object.editmode_toggle()

class _OT_collision_assign(Operator):
    '''Assign selected collision meshes to the active object'''

    bl_idname = 'col.collision_assign'
    bl_label = "Assign Collision"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):

        return len(context.selected_objects) > 1 and context.object and context.mode == 'OBJECT'

    def execute(self, context):
        bpy.context.space_data.shading.wireframe_color_type = 'OBJECT'
        collision_prefixes = ("UCX", "UBX", "UCP", "USP")
        for obj in context.selected_objects[:]:
            if obj == context.active_object:
                continue

            prefix = obj.name[:3]
            if prefix in collision_prefixes:
                obj.name = find_free_col_name(
                    prefix, context.active_object.name)

                if obj.data.users == 1:
                    obj.data.name = obj.name

        return {'FINISHED'}


class _OT_collision_makeUBX(Operator):
    # tooltip
    '''Generate Box collision mesh'''

    bl_idname = 'col.collision_make_ubx'
    bl_label = "Make UBX Collision"
    bl_options = {'REGISTER', 'UNDO'}

    AAB: bpy.props.BoolProperty(
        name="AAB",
        description="Absolute Bounding Box",
        default=True
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object

        return obj and obj.type == 'MESH' and obj.mode in {'OBJECT', 'EDIT'}

    def create_cube(self, object_BB):

        vertices = object_BB

        # Top      Left     right     bottom    front      back
        faces = [(1, 0, 3, 2), (4, 6, 7, 5), (0, 7, 6, 3),
                 (7, 0, 1, 5), (2, 4, 5, 1), (4, 2, 3, 6)]
        edges = []
        # Create an empty mesh and the object.
        mesh = bpy.data.meshes.new('Basic_Cube')
        mesh.from_pydata(vertices, edges, faces)

        new_object = bpy.data.objects.new("new_object", mesh)
        bm = bmesh.new()
        me = new_object.data
        bm.from_mesh(me)  # load bmesh
        for f in bm.faces:
            f.normal_flip()
        bm.normal_update()  # not sure if req'd
        bm.to_mesh(me)
        me.update()
        bm.clear()  # .. clear before load next

        view_layer = bpy.context.view_layer
        view_layer.active_layer_collection.collection.objects.link(new_object)
        return (new_object)

    def execute(self, context):
        bpy.context.space_data.shading.wireframe_color_type = 'OBJECT'

        obj = context.active_object

        if obj.mode != 'EDIT':
            # When working from object mode, it follows that there should be only one collision shape
            pattern = re.compile(rf"^U[A-Z][A-Z]_{obj.name}_\d+")
            for mesh in [mesh for mesh in bpy.data.meshes if pattern.match(mesh.name)]:
                bpy.data.meshes.remove(mesh)
        for obj in bpy.context.selected_objects:
            object_BB = get_BoundBox(obj, self.AAB)

            new_object = self.create_cube(object_BB)
            new_object.display_type = 'WIRE'
            new_object.color = (0, 1, 0, 1)
            new_object.show_in_front = True

            new_object['Collision'] = True

            new_object.location = obj.location
            new_object.parent = bpy.data.objects[obj.name]

            col_name = find_free_col_name('UBX', obj.name)
            new_object.name = col_name

            if new_object.parent != None:
                print("something")
                new_object.matrix_parent_inverse = new_object.parent.matrix_world.inverted()

            collection = get_collection(
                context, "Collision", allow_duplicate=True, clean=False)
            collection.color_tag = 'COLOR_04'

            collection.objects.link(new_object)

        return {'FINISHED'}


class _OT_collision_makeUCX(Operator):

    '''Creates convex collision mesh'''

    bl_idname = 'col.collision_make_ucx'
    bl_label = "Make UCX Collision"
    bl_options = {'REGISTER', 'UNDO'}

    # Convex settings
    planar_angle: bpy.props.FloatProperty(
        name="Max Face Angle",
        description="Use to remove decimation bias towards large, bumpy faces",
        subtype='ANGLE',
        default=radians(10.0),
        min=0.0,
        max=radians(180.0),
        soft_max=radians(90.0),
    )
    decimate_ratio: bpy.props.FloatProperty(
        name="Decimate Ratio",
        description="Percentage of edges to collapse",
        subtype='FACTOR',
        default=1.0,
        min=0.0,
        max=1.0,
    )
    use_symmetry: bpy.props.BoolProperty(
        name="Symmetry",
        description="Maintain symmetry on an axis",
        default=False,
    )
    symmetry_axis: bpy.props.EnumProperty(
        name="Symmetry Axis",
        description="Axis of symmetry",
        items=[
            ('X', "X", "X"),
            ('Y', "Y", "Y"),
            ('Z', "Z", "Z"),
        ],
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object

        return obj and obj.type == 'MESH' and obj.mode in {'OBJECT', 'EDIT'}

    def make_convex_collision(self, context, obj):

        if context.mode == 'EDIT_MESH':
            bm = bmesh.from_edit_mesh(obj.data).copy()
            bm.verts.ensure_lookup_table()
            bmesh.ops.delete(
                bm, geom=[v for v in bm.verts if not v.select], context='VERTS')

        else:
            bm = bmesh.new()
            dg = context.evaluated_depsgraph_get()
            bm.from_object(obj, dg)

        # Clean incoming mesh
        bm.edges.ensure_lookup_table()
        for edge in bm.edges:
            edge.seam = False
            edge.smooth = True

        bm.faces.ensure_lookup_table()
        for face in bm.faces:
            face.smooth = False

        geom = list(chain(bm.verts, bm.edges, bm.faces))
        result = bmesh.ops.convex_hull(bm, input=geom, use_existing_faces=True)

        bmesh.ops.delete(
            bm, geom=result['geom_interior'], context='TAGGED_ONLY')
        bm.normal_update()
        bmesh.ops.dissolve_limit(bm, angle_limit=self.planar_angle,
                                 verts=bm.verts, edges=bm.edges, use_dissolve_boundaries=False, delimit=set())
        bmesh.ops.triangulate(bm, faces=bm.faces)
        col_obj = create_col_object_from_bm(self, context, obj, bm, "UCX")
        bm.free()

        # Decimate (no bmesh op for this currently?)
        with TempModifier(col_obj, type='DECIMATE') as dec_mod:
            dec_mod.ratio = self.decimate_ratio
            dec_mod.use_symmetry = self.use_symmetry
            dec_mod.symmetry_axis = self.symmetry_axis

        return (col_obj)

    def execute(self, context):
        bpy.context.space_data.shading.wireframe_color_type = 'OBJECT'
        obj = context.active_object

        if obj.mode != 'EDIT':
            # When working from object mode, it follows that there should be only one collision shape
            pattern = re.compile(rf"^U[A-Z][A-Z]_{obj.name}_\d+")
            for mesh in [mesh for mesh in bpy.data.meshes if pattern.match(mesh.name)]:
                bpy.data.meshes.remove(mesh)
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                col_obj = self.make_convex_collision(context, obj)
                col_obj['Collision'] = True
                col_obj.location = obj.location
                col_obj.parent = bpy.data.objects[obj.name]
                col_obj['Collision'] = True
                if col_obj.parent != None:
                    print("something")
                    col_obj.matrix_parent_inverse = col_obj.parent.matrix_world.inverted()

        return {'FINISHED'}


class _OT_collision_makeCYL(Operator):

    '''Generate cylinder collision mesh'''

    bl_idname = 'col.collision_make_cyl'
    bl_label = "Make Cylinder Collision"
    bl_options = {'REGISTER', 'UNDO'}

    cyl_sides: bpy.props.IntProperty(
        name="Sides",
        description="Number of sides",
        default=8,
        min=3,
    )

    cyl_radius1: bpy.props.FloatProperty(
        name="Radius 1",
        description="First cylinder radius",
        subtype='DISTANCE',

    )
    cyl_radius2: bpy.props.FloatProperty(
        name="Radius 2",
        description="Second cylinder radius",
        subtype='DISTANCE',

    )

    AAB: bpy.props.BoolProperty(
        name="AAB",
        description="Absolute Bounding Box",
        default=True
    )
    @classmethod
    def poll(cls, context):
        obj = context.active_object

        return obj and obj.type == 'MESH' and obj.mode in {'OBJECT', 'EDIT'}

    def make_cylinder_collision(self, context, obj, v1: Vector, v2: Vector, center, axis, r1=0.1, r2=0.1):
        mat_loc = Matrix.LocRotScale(center, Vector(
            (0, 0, 1)).rotation_difference(axis), (1, 1, 1))
        bm = bmesh.new()
        dv: Vector = v2 - v1

        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=self.cyl_sides,
                              radius1=r1/1.5, radius2=r2/1.5, depth=dv.length, matrix=mat_loc)

        col_obj = create_col_object_from_bm(self, context, obj, bm, "UCX")
        bm.free

        return (col_obj)

    def execute(self, context):
        bpy.context.space_data.shading.wireframe_color_type = 'OBJECT'
        obj = context.active_object

        if obj.mode != 'EDIT':
            # When working from object mode, it follows that there should be only one collision shape
            pattern = re.compile(rf"^U[A-Z][A-Z]_{obj.name}_\d+")
            for mesh in [mesh for mesh in bpy.data.meshes if pattern.match(mesh.name)]:
                bpy.data.meshes.remove(mesh)

        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                BB = get_BoundBoxCorners(obj,self.AAB)

                col_obj = self.make_cylinder_collision(context, obj, Vector(BB[1]), Vector(BB[0]), Vector(
                    BB[2]), (BB[3]), BB[0][1]-BB[1][1]+self.cyl_radius1, BB[0][1]-BB[1][1]+self.cyl_radius2)

                col_obj['Collision'] = True
                col_obj.location = obj.location
                col_obj.parent = bpy.data.objects[obj.name]

                if col_obj.parent != None:
                    print("something")
                    col_obj.matrix_parent_inverse = col_obj.parent.matrix_world.inverted()

        return {'FINISHED'}


class _OT_collision_makeSPH(Operator):

    '''Generate sphere collision mesh'''

    bl_idname = 'col.collision_make_sph'
    bl_label = "Make Spherical Collision"
    bl_options = {'REGISTER', 'UNDO'}

    # Sphere settings
    sph_radius: bpy.props.FloatProperty(
        name="Radius",
        description="Sphere radius",
        subtype='DISTANCE',
        default=1,
        min=.5,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object

        return obj and obj.type == 'MESH' and obj.mode in {'OBJECT', 'EDIT'}

    def get_BBCenter(self, object):
        '''Generates a object aligned bounding box 
        to be used to generate collision meshes'''
        obj = object

        if obj.mode == 'EDIT':
            print('in edit mode')
            bm = bmesh.from_edit_mesh(obj.data)
            verts = [v.co for v in bm.verts if v.select]
        else:

            verts = [v.co for v in obj.data.vertices]

        points = np.asarray(verts)

        cov = np.cov(points, y=None, rowvar=0, bias=1)

        v, vect = np.linalg.eig(cov)

        tvect = np.transpose(vect)
        points_r = np.dot(points, np.linalg.inv(tvect))

        co_min = np.min(points_r, axis=0)
        co_max = np.max(points_r, axis=0)

        xmin, xmax = co_min[0], co_max[0]
        ymin, ymax = co_min[1], co_max[1]
        zmin, zmax = co_min[2], co_max[2]

        xdif = (xmax - xmin) * 0.5
        ydif = (ymax - ymin) * 0.5
        zdif = (zmax - zmin) * 0.5

        cx = xmin + xdif
        cy = ymin + ydif
        cz = zmin + zdif

        center = np.dot([cx, cy, cz], tvect)
        return (center)

    def make_sphere_collision(self, context, obj):
        center = self.get_BBCenter(obj)

        mat = Matrix.Translation(center)
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=self.sph_radius,
                                   calc_uvs=False, matrix=mat)
        col_obj = create_col_object_from_bm(self, context, obj, bm, "USP")
        bm.free()
        return (col_obj)

    def execute(self, context):
        bpy.context.space_data.shading.wireframe_color_type = 'OBJECT'
        obj = context.active_object

        if obj.mode != 'EDIT':
            # When working from object mode, it follows that there should be only one collision shape
            pattern = re.compile(rf"^U[A-Z][A-Z]_{obj.name}_\d+")
            for mesh in [mesh for mesh in bpy.data.meshes if pattern.match(mesh.name)]:
                bpy.data.meshes.remove(mesh)

        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':

                col_obj = self.make_sphere_collision(context, obj)
                col_obj['Collision'] = True
                col_obj.location = obj.location
                col_obj.parent = bpy.data.objects[obj.name]

                if col_obj.parent != None:
                    print("something")
                    col_obj.matrix_parent_inverse = col_obj.parent.matrix_world.inverted()

        return {'FINISHED'}


classes = (

    _OT_collision_makeUBX,
    _OT_collision_makeUCX,
    _OT_collision_assign,
    _OT_collision_makeSPH,
    _OT_collision_makeCYL,

)


# try:
#     SpaceView3D.draw_handler_remove(SpaceView3D.my_handler, 'WINDOW')
# except (AttributeError, ValueError):
#     pass

# SpaceView3D.my_handler = SpaceView3D.draw_handler_add(
#     drawColWire, (), 'WINDOW', 'POST_VIEW')


def register():
    


    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
