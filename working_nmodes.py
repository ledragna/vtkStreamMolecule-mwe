import sys
import os
import vtk
import numpy as np
from math import ceil
from PySide6.QtWidgets import QWidget, QMainWindow, QApplication, QVBoxLayout, QFrame
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from estampes.parser import DataFile, build_qlabel
# from estampes.tools.vib import orient_modes
from estampes.tools.atom import convert_labsymb
from estampes.data.physics import PHYSFACT, PHYSCNST


B2AN = 0.52917720859

def get_vibmol(fname):
    """_summary_

    Args:
        fname (str): gaussian fchk filename

    Returns:
        dict: read quantities
    """
    dkeys = {'aat': build_qlabel(102,None,1),
             'apt': build_qlabel(101,None,1)
            }
    dkeys2 = {'eng': build_qlabel(1),
              'atcrd': build_qlabel('atcrd', 'last'),
              'atnum': build_qlabel('atnum'),
              'atmas': build_qlabel('atmas'),
              'hessvec': build_qlabel('hessvec'),
              'hessval': build_qlabel('hessval')
            }
    # print(fname)
    dfile = DataFile(fname)
    res = {}
    res['fname'] = fname
    data = dfile.get_data(*dkeys2.values())
    atmnum = len(data[dkeys2['atnum']]['data'])
    res['atnum'] = data[dkeys2['atnum']]['data']
    res['eng'] = data[dkeys2['eng']]['data']
    res['atlab'] = convert_labsymb(True, *data[dkeys2['atnum']]['data'])
    res['atcrd'] = np.array(data[dkeys2['atcrd']]['data'])*PHYSFACT.bohr2ang
    res['eng'] = data[dkeys2['eng']]['data']
    res['atmas'] = np.array(data[dkeys2['atmas']]['data'])
    # BUG no linear
    nmnum = atmnum*3-6
    res['evec'] = np.reshape(np.array(data[dkeys2['hessvec']]['data']), (-1, atmnum*3))
    tmp = np.array(data[dkeys2['hessval']]['data'])
    res['freq'] = tmp[:nmnum]
    res['rmas'] = tmp[nmnum:2*nmnum]

    res['lx'] = res['evec']/np.sqrt(res['rmas'])[:, np.newaxis]

    try:
        data = dfile.get_data(dkeys['apt'])
        res['apt'] = np.array(data[dkeys['apt']]['data']).reshape(-1,3)
        res['edi'] = np.einsum("ij,jk->ik",res['lx'],res['apt'])
    except:
        res['apt'] = None
    try:
        data = dfile.get_data(dkeys['aat'])
        res['aat'] = np.array(data[dkeys['aat']]['data']).reshape(-1,3)
        res['mdi'] = np.einsum("ij,jk->ik",res['lx'],res['aat'])
    except:
        res['aat'] = None

    return res


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

        moldata = get_vibmol("oxirane_freq.fchk")
        # cubedata = cube_parser("g+t_p1_pw6_frq_nosym_VTCD47_U150.cube")
        atoms = []
        for i in range(len(moldata['atnum'])):
            atoms.append(mol.AppendAtom(moldata['atnum'][i],
                                        *moldata['atcrd'][i]))
        bond = vtk.vtkSimpleBondPerceiver()
        bond.SetInputData(mol)
        bond.Update()
        mol = bond.GetOutput()

        evec = moldata['evec'][4].reshape(-1,3)
        ian = moldata['atmas']

        norms = np.sqrt(np.einsum('ij,ij->i', evec, evec))
        # weighted by the charge
        # normalized
        norm_evec = evec/ np.max(norms)
        # change the sign to be opposed to the TCD
        natm = int(moldata['atcrd'].shape[0])
        # PolyData
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(natm)
        vect = vtk.vtkDoubleArray()
        vect.SetNumberOfComponents(3)
        vect.SetNumberOfTuples(natm)
        norm = vtk.vtkDoubleArray()
        norm.SetNumberOfComponents(1)
        norm.SetNumberOfTuples(natm)
        for i in range(natm):
            points.SetPoint(i, moldata['atcrd'][i, :])
            # norm.SetValue(i, norm_evec[i])
            norm.SetValue(i, 1.)
            vect.SetTuple3(i, *norm_evec[i, :])
        vect.SetName('vector')
        norm.SetName('norm')
 
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().AddArray(vect)
        polydata.GetPointData().AddArray(norm)
        polydata.GetPointData().SetActiveVectors("vector")
        polydata.GetPointData().SetActiveScalars('norm')

        arrow = vtk.vtkArrowSource()
        glyphs = vtk.vtkGlyph3D()
        glyphs.SetInputData(polydata)
        glyphs.SetSourceConnection(arrow.GetOutputPort())
        # the mapper
        glyph_mapper =  vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputConnection(glyphs.GetOutputPort())
        glyph_actor = vtk.vtkActor()
        glyph_actor.SetMapper(glyph_mapper)
        glyph_actor.VisibilityOn()

        glyphs.SetVectorModeToUseVector()
        glyphs.SetScaleModeToScaleByVector()
        # glyphs.SetScaleModeToScaleByScalar()
        # Scale factor
        glyphs.SetScaleFactor(1.)
        clrs = vtk.vtkNamedColors()
        glyphs.SetColorModeToColorByScalar()
        glyph_actor.GetProperty().SetColor(clrs.GetColor3d('Blue'))

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
    sys.exit(app.exec())
