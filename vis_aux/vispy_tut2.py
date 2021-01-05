import numpy as np
from vispy import app  # to display a canvas
from vispy import gloo  # object-oriented interface to OpenGL

# In order to display a window, we need to create a Canvas
c = app.Canvas(keys='interactive')

# When using vispy.gloo, we need to write shaders. These programs, written in a C-like language called GLSL,
# run on the GPU and give us full flexibility for our visualizations. Here, we create a trivial vertex shader
# that directly displays 2D data points (stored in the a_position variable) in the canvas.
# The function main() executes once per data point (also called vertex). The variable a_position contains the (x, y)
# coordinates of the current vertex. All this function does is to pass these coordinates to the next stage of processing
# in the rendering pipeline.
vertex = """
attribute vec2 a_position
void main (void)
{
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

# The other shader we need to create is the fragment shader. It lets us control the pixels' color.
# Here, we display all data points in black. This function runs once per generated pixel.
fragment = """
void main()
{
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
}
"""

# Create OpenGL Program. This object contains the shaders and allows us to link the shader variables to Py/NumPy data
program = gloo.Program(vertex, fragment)

# Link the variable a_position to a (1000, 2) NumPy array containing the coordinates of 1000 data points.
# In the default coordinate system, the coordinates of the four canvas corners are (+/-1, +/-1).
# Here, we generate a random time-dependent signal in [-1,1]
program['a_position'] = np.c_[
    np.linspace(-1.0, +1.0, 1000),
    np.random.uniform(-0.5, +0.5, 1000)].astype(np.float32)


# Create a callback function called when the window is being resized.
# Updating the OpenGL viewport lets us ensure that Vispy uses the entire canvas
@c.connect
def on_resize(event):
    gloo.set_viewport(0, 0, *event.size)


# We create a callback function called when the canvas needs to be refreshed. This on_draw function renders the entire
# scene. First, we clear the window in white (it is necessary to do that at every frame). Then, we draw a succession of
# line segments using our OpenGL program. The vertices used for this visual are those returned by the vertex shader.
@c.connect
def on_draw(event):
    gloo.clear((1, 1, 1, 1))
    program.draw('line_strip')


#  Show canvas and run application
c.show()
app.run()
