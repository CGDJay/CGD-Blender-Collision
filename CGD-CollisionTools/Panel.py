import bpy


class VIEW3D_MT_PIE_ColOps(bpy.types.Menu):
    # label is displayed at the center of the pie menu.
    bl_label = "_MT_col.operations"
    bl_idname = "_MT_col.operations"
    
    def draw(self, context):
        layout = self.layout

        pie = layout.menu_pie()
        box = pie.box()

        

#############################################################
        #type of operators
        
        row=box.row()
        row.scale_y=1.5
        row.operator("col.collision_make_ubx")
        row.operator("col.collision_make_ucx")

        row=box.row()
        row.scale_y=1.5
        row.operator("col.collision_make_cyl")
        row.operator("col.collision_make_sph")
        
        box = pie.box()
        row=box.row()  
        row.scale_y=2
        row.operator("col.collision_assign")   



classes = (
VIEW3D_MT_PIE_ColOps,

)


def register():

    for cls in classes :
        bpy.utils.register_class(cls)

def unregister():

    for cls in classes :
     bpy.utils.unregister_class(cls)