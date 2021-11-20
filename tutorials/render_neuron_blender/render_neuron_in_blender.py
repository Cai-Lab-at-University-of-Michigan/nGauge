import math, mathutils
import bpy, bmesh
from bpy.props import *
from bpy_extras.io_utils import ExportHelper
import random
import ngauge

def draw_line_3D( start, stop, mball, min_radius=1.0, meta_ball_scale_factor=1.0 ):
#    print( "\tStart...", start )
#    print( "\tStop... ", stop )
    
    start_coord = [start.x, start.y, start.z, start.r]
    end_coord = [stop.x, stop.y, stop.z, stop.r]
    
    start_coord[3] = max( start_coord[3], min_radius )
    end_coord[3] = max( end_coord[3], min_radius )
    current_coord = list( start_coord )
    
    segment_length = sum( (i-j)**2.0 for i,j in zip(start_coord[:3], end_coord[:3]) )
    segment_length += 0.01
    deltas = [ j-i for i,j in zip(start_coord, end_coord) ]
    
    length_so_far = 0.0
    while length_so_far <= segment_length:
        # Make a sphere at this point
        ele = mball.elements.new()
        r = current_coord[3]
        ele.radius = r * meta_ball_scale_factor
        ele.co = tuple(current_coord[:3])

        # Move x, y, z, and r to the next point
        length_so_far += r/2
        current_coord = [
            scoord + (length_so_far * delta / segment_length)
            for scoord,delta in zip(start_coord, deltas)
        ]
        
def draw_node_recursive( n, mball ):
    for child in n.children:
        draw_line_3D( n, child, mball )
        draw_node_recursive( child, mball )


def render_neuron(
        n,
        scale_file_data=1.0,
        min_radius=1.0,
        resolution=0.25,
        name='Neuron',
        color=(1,1,1,1),
        z_scale=-1.0
    ):
        
    ### 1. OPEN REQUIRED OBJECTS
    # Print debug message to console
    print( "Plotting", n )
    
    # Select scene and create a new "metaball" object to put the rendered
    # neuron into
    scene = bpy.context.scene
    mball = bpy.data.metaballs.new( name.lower() )
    mball.resolution = resolution
    obj = bpy.data.objects.new(name,mball)
    scene.collection.objects.link(obj)
    
    ### 2. DRAW THE MODEL
    #draw_line_3D( n.branches[0], n.branches[0].children[0], mball )
    
    # 2.1. Plot the Soma Points
    for layer in n.soma_layers.values():
        # Draw a line between each point in the soma
        for i,j in zip( layer, layer[1:] ):
            draw_line_3D( i, j, mball )
        # Draw a line between the first and last point
        if len( layer ) > 1:
            draw_line_3D( layer[-1], layer[0], mball )
        
    # 2.2. Plot each branch
    for branch in n.branches:
        draw_node_recursive( branch, mball )
    
    ### 3. POSTPROCESS AND SAVE
    # Convert the metaball into an object
    depsgraph = bpy.context.evaluated_depsgraph_get()
    object_eval = obj.evaluated_get(depsgraph)
    tmpMesh = bpy.data.meshes.new_from_object(object_eval)        
    tmpMesh.transform(obj.matrix_world)
    tmp_ground = bpy.data.objects.new(name=name, object_data=tmpMesh)
    
    # Rescale the model in XYZ
    for i in range(3):
        tmp_ground.scale[i] = .1
    
    # Remove tmp object
    scene.collection.objects.unlink(obj)
    
    # Apply appearance variables to the object
    mat = bpy.data.materials.new("PKHG")
    mat.diffuse_color = color
    tmp_ground.active_material = mat
    
    # These codes can be used to apply a "Decimate" filter which
    # downsamples
    #decimate=tmp_ground.modifiers.new('DecimateMod','DECIMATE')
    #decimate.ratio=1
    #decimate.use_collapse_triangulate=True
    
    # Link object to the scene
    scene.collection.objects.link(tmp_ground)
    bpy.context.view_layer.objects.active = tmp_ground
    #bpy.ops.object.modifier_apply(modifier=decimate.name)

# RUN THE CODE
from glob import glob
import random

# Find and low files using glob
#files = glob("[LOCATION OF DATA FOLDER]")
files = glob("/Users/loganaw/Documents/GitHub/nGauge/tutorials/render_neuron_blender/ntracer_tracing_swc/*.swc")
neurons = [ngauge.Neuron.from_swc(f) for f in files]

# Loop through the neurons
for neuron in neurons:
    # Generate a random RGB color
    color = ( random.random() * 0.75, random.random() * 0.75, random.random() * 0.75, 1)
    
    # Render neuron
    render_neuron( neuron, color=color )