from paraview.simple import *
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

def get_column(reader, name):
    reader.UpdatePipeline()
    table = servermanager.Fetch(reader)
    return vtk_to_numpy(table.GetColumnByName(name))

reader = CSVReader(
    FileName = ["/Users/mitchellhung/Desktop/Mitchell folder/High School Internships/paraview_data/analysis_results.csv"]
)
interested_parameter = "height"
height = get_column(reader, interested_parameter)
alpha = np.diff(np.unique(height)).min()

table = TableToPoints(
    Input = reader
)

table.XColumn = "x"
table.YColumn = "y"
table.ZColumn = interested_parameter

view = GetActiveViewOrCreate("RenderView")

#Show(table, view)

surface = Delaunay2D(Input = table)
surface.Alpha = 0
Show(surface, view)

#ColorBy(display, ("POINTS", "cluster"))

Render()
