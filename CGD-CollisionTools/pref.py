import bpy
import rna_keymap_ui 
from bpy.types import  AddonPreferences
from bpy.props import (
	StringProperty,
	BoolProperty,
	IntProperty,
	FloatProperty,
	FloatVectorProperty,
	EnumProperty,
	PointerProperty,
)


class Panel_Preferences(AddonPreferences):

    bl_idname = __package__
    #properties menu

    
    
    def draw (self, context):
        layout = self.layout
     
        box = layout.box()
        col = box.column()
        col.label(text="Keymap List:",icon="KEYINGSET")


        wm = bpy.context.window_manager
        kc = wm.keyconfigs.user
        old_km_name = ""
        get_kmi_l = []
        for km_add, kmi_add in addon_keymaps:
            for km_con in kc.keymaps:
                if km_add.name == km_con.name:
                    km = km_con
                    break

            for kmi_con in km.keymap_items:
                if kmi_add.idname == kmi_con.idname:
                    if kmi_add.name == kmi_con.name:
                        get_kmi_l.append((km,kmi_con))

        get_kmi_l = sorted(set(get_kmi_l), key=get_kmi_l.index)

        for km, kmi in get_kmi_l:
            if not km.name == old_km_name:
                col.label(text=str(km.name),icon="DOT")
            col.context_pointer_set("keymap", km)
            rna_keymap_ui.draw_kmi([], kc, km, kmi, col, 0)
            col.separator()
            old_km_name = km.name


        box= layout.box()
        box.scale_y = 2
        box.scale_x = 2
        box.label(icon="DOT")
        box.prop(self, "bool_Enable_Panel")
        box.prop(self, "String_Catergory")


classes = ( 

Panel_Preferences,

)

#-------------------------------------------------------
#KeyMaps
disabled_kmis = []

# Find a keymap item by traits.
# Returns None if the keymap item doesn't exist.


def get_active_kmi(space: str, **kwargs) -> bpy.types.KeyMapItem:
    kc = bpy.context.window_manager.keyconfigs.user
    km = kc.keymaps.get(space)
    if km:
        for kmi in km.keymap_items:
            for key, val in kwargs.items():
                if getattr(kmi, key) != val and val is not None:
                    break
            else:
                return kmi

def disable_Used_kmi():
    # Be explicit with modifiers shift/ctrl/alt so we don't
    # accidentally disable a different keymap with same traits.
    kmi = get_active_kmi("Mesh",
                         idname="mesh.split",
                         type='C',
                         shift=False,
                         ctrl=False,
                         alt=False)


    if kmi is not None:
        kmi.active = False

        disabled_kmis.append(kmi)



addon_keymaps = []


def register():
    for cls in classes :
        bpy.utils.register_class(cls)

    #-------------------------------------------------------
#KeyMaps

    disable_Used_kmi()
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = wm.keyconfigs.addon.keymaps.new(name='3D View', space_type='VIEW_3D')
        
        kmi = km.keymap_items.new('wm.call_menu_pie', type = 'C', value = 'PRESS', ctrl = False, shift = False)
        kmi.properties.name = "_MT_col.operations"

        addon_keymaps.append((km, kmi))


def unregister():
    for cls in classes :
        bpy.utils.unregister_class(cls)

    #-------------------------------------------------------
#KeyMaps

    # unregister shortcuts
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    km = kc.keymaps['3D View']
    
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
            wm.keyconfigs.addon.keymaps.remove(km)
        except ReferenceError as e:
            e
            
    addon_keymaps.clear()
