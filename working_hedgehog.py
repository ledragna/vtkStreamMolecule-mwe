import sys
import os
import vtk
import numpy as np
from math import ceil
from PySide2.QtWidgets import QWidget, QMainWindow, QApplication, QVBoxLayout, QFrame
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

B2AN = 0.52917720859

def cube_parser(cubfile):
    """
    Read and extract data from a cube file
    Return:
        dict -- keys: natms, loc2wrd, nval, npts, atm, crd, cube and if mocude
        lbs
    """
    if not os.path.exists(cubfile):
        print('ERROR: Cube file "{}" not found'.format(cubfile))
        raise OSError

    data_tmp = {}
    with open(cubfile, 'r') as fi_le:
        # 1st 2 lines are titles/comments. Ignored for now
        line = fi_le.readline()  # Title
        line = fi_le.readline()  # Comment
        line = fi_le.readline()  # NAtoms, X0, Y0, Z0, NVal
        token = line.split()
        mo_cube = False
        if int(token[0]) < 0:
            mo_cube = True
        data_tmp['natms'] = int(token[0])
        data_tmp['loc2wrd'] = np.identity(4)
        data_tmp['loc2wrd'][:3, 3] = [float(e)*B2AN for e in token[1:4]]
        if len(token) > 4:
            nval = int(token[4])
        else:
            nval = 1
        data_tmp['nval'] = nval
        # REMINDER: Gaussian does not require the grid to be "rectangular"
        # N1, X1, Y1, Z1 (displacement along 1st coord)
        data_tmp['npts'] = np.zeros(3, dtype=int)
        for i_th in range(3):
            line = fi_le.readline()
            token = line.split()
            data_tmp['npts'][i_th] = int(token[0])
            data_tmp['loc2wrd'][:3, i_th] = [float(e)*B2AN for e in token[1:4]]
        #data_tmp['wrd2loc'] = np.linalg.inv(data_tmp['loc2wrd'])
        data_tmp['atm'] = []
        data_tmp['crd'] = []
        for i_th in range(data_tmp['natms']):
            line = fi_le.readline()
            # AtNum(i)x2, X(i), Y(i), Z(i)
            token = line.split()
            data_tmp['atm'].append(int(token[0]))
            data_tmp['crd'].append([float(x)*B2AN for x in token[2:]])
        if mo_cube:
            nlines = ceil((data_tmp['nval']+1)/10)
            labels = []
            for i_th in range(nlines):
                line = fi_le.readline()
                labels.extend([str(x) for x in line.split()])
            data_tmp['lbs'] = labels[1:]

        content = fi_le.readlines()
    vec_num = []
    for line in content:
        vec_num.extend([float(x) for x in line.split()])
    vec_num = np.array(vec_num)
    if nval > 1:
        vec_num.resize(int(vec_num.shape[0] / nval), nval)
        vec_num = vec_num.transpose()
    else:
        vec_num = np.expand_dims(vec_num, axis=0)
    data_tmp['cube'] = vec_num

    return data_tmp

def fillcubeimage(data):
    """
    Fills a vtkImageData object 
    
    Arguments:
        data {dict} -- dictionary with a cube dataset

    Returns:
        vtkImageData
    """
    cubeimage = vtk.vtkImageData()
    cubeimage.SetDimensions(*data['npts'])
    cubeimage.SetOrigin(*data['loc2wrd'][:3, 3])
    cubeimage.AllocateScalars(vtk.VTK_DOUBLE, data['nval']+1)
    cubeimage.SetSpacing(*np.diag(data['loc2wrd'][:3,:3]))
    vect = vtk.vtkDoubleArray()
    vect.SetNumberOfComponents(3)
    vect.SetNumberOfTuples(cubeimage.GetNumberOfPoints())
    norm = vtk.vtkDoubleArray()
    norm.SetNumberOfComponents(1)
    norm.SetNumberOfTuples(cubeimage.GetNumberOfPoints())

    for i in range(data['npts'][0]):
        for j in range(data['npts'][1]):
            for k in range(data['npts'][2]):
                indices = (k * data['npts'][1] + j) * data['npts'][0] + i
                indicesf = (i * data['npts'][1] + j) * data['npts'][2] + k
                norval = np.sqrt(np.dot(data['cube'][:, indicesf], data['cube'][:, indicesf]))
                norm.SetValue(indices, norval)
                vect.SetTuple3(indices, *data['cube'][:, indicesf])
#                for h in range(data['nval']):
#                    cubeimage.SetScalarComponentFromDouble(i, j, k, h,
#                                                        data['cube'][h, indicesf])

    vect.SetName('vector')
    norm.SetName('norm')
    # print(norm.GetRange())
    cubeimage.GetPointData().AddArray(vect)
    cubeimage.GetPointData().AddArray(norm)

    return cubeimage



class MainWindow(QMainWindow):

    def __init__(self, parent = None):
        QMainWindow.__init__(self, parent)

        self.frame = QFrame()
        self.vl = QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(1.0, 1.0, 1.0)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()


        mol = vtk.vtkMolecule()
        
        cubedata = cube_parser("oxirane_TCD_S001.cube")
        # cubedata = cube_parser("g+t_p1_pw6_frq_nosym_VTCD47_U150.cube")
        atoms = []
        for i in range(len(cubedata['atm'])):
            atoms.append(mol.AppendAtom(cubedata['atm'][i], *cubedata['crd'][i]))
        bond = vtk.vtkSimpleBondPerceiver()
        bond.SetInputData(mol)
        bond.Update()
        mol = bond.GetOutput()

        # Vectors stuff
        # https://stackoverflow.com/questions/57309203/plotting-vector-fields-efficiently-using-vtk-avoiding-excessive-looping
        grid = fillcubeimage(cubedata)



        grid.GetPointData().SetActiveVectors("vector")
        grid.GetPointData().SetActiveScalars('norm')
        bounds = grid.GetScalarRange()
        bounds2 = (0, bounds[1]/100)
        print(bounds)

        arrow = vtk.vtkArrowSource()
        glyphs = vtk.vtkGlyph3D()
        glyphs.SetInputData(grid)
        glyphs.SetSourceConnection(arrow.GetOutputPort())
        # glyphs.Update()

        glyph_mapper =  vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputConnection(glyphs.GetOutputPort())
        glyph_actor = vtk.vtkActor()
        glyph_actor.SetMapper(glyph_mapper)
        glyph_actor.VisibilityOn()

        glyphs.SetVectorModeToUseVector()
        glyphs.SetScaleModeToScaleByScalar()
        # #glyphs.SetScaleModeToDataScalingOff()
        
        # glyph_mapper.UseLookupTableScalarRangeOn()


        # glyphs.SetScaleModeToScaleByVector()
        glyphs.SetScaleFactor(10)
        glyphs.SetColorModeToColorByScalar()

        # s0,sf = glyphs.GetOutput().GetScalarRange()
        # print(s0,sf)
        lut = vtk.vtkColorTransferFunction()
        lut.AddRGBPoint(0.001, 1,0,0)
        lut.AddRGBPoint(0.15, 0,1,0)
        glyph_mapper.SetLookupTable(lut)

        threshold = vtk.vtkThresholdPoints()
        threshold.SetInputData(grid)
        threshold.ThresholdBetween(0.001,0.15)
        glyphs.SetInputConnection(threshold.GetOutputPort())

        # Create a mapper
        # mapper = vtk.vtkPolyDataMapper()
        mapper = vtk.vtkMoleculeMapper()
        mapper.UseLiquoriceStickSettings()
        # mapper.SetInputConnection(source.GetOutputPort())
        mapper.SetInputData(mol)

        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        #actor.SetRenderStyleToVolume()

        self.ren.AddActor(actor)
        self.ren.AddActor(glyph_actor)

        self.ren.ResetCamera()

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.show()
        self.iren.Initialize()
        self.iren.Start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
