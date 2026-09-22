import numpy as np
from scipy.constants import epsilon_0 as e0

class Scene(object):
    def __init__(self):
        self.materials = [] 

    def print_materials(self):
        for m in self.materials:
            print(f"{m.ID} er={m.er:.6g} sigma={m.se:.5g}")

class Material(object):
    waterer = 80.1
    watereri = 4.9
    waterdeltaer = waterer - watereri
    watertau = 9.231e-12
    maxpoles = 0

    def __init__(self, ID):

        self.ID = ID
        self.type = ''

        self.er = 1.0
        self.se = 0.0
        self.mr = 1.0
        self.sm = 0.0

        self.poles = 0
        self.deltaer = []
        self.tau = []
        self.alpha = []

    def __repr__(self):
        return f"<{self.ID} er={self.er:.6g}>"

    def calculate_er(self, freq):
        er = self.er

        if self.poles > 0:
            w = 2 * np.pi * freq
            er += self.se / (1j * w * e0)
            if 'debye' in self.type:
                for pole in range(self.poles):
                    er += self.deltaer[pole] / (1 + 1j * w * self.tau[pole])
            elif 'lorentz' in self.type:
                for pole in range(self.poles):
                    er += (self.deltaer[pole] * self.tau[pole]**2) / (self.tau[pole]**2 + 2j * w * self.alpha[pole] - w**2)
            elif 'drude' in self.type:
                ersum = 0
                for pole in range(self.poles):
                    ersum += self.tau[pole]**2 / (w**2 - 1j * w * self.alpha[pole])
                    er -= ersum
        return er
    
class Peplinskisoil(object):

    def __init__(self, ID, sandfraction, clayfraction, bulkdensity, sandpartdensity, *watervolfraction):
        self.ID = ID
        self.S = sandfraction
        self.C = clayfraction
        self.rb = bulkdensity
        self.rs = sandpartdensity
        self.fw = watervolfraction
        self.startmaterialnum = 0
    
    def calculate_er(self, nmat, scene ,fractalboxname):
        f=1.3e9
        a = 0.65
        w = 2 * np.pi * f
        erealw = Material.watereri + ((Material.waterdeltaer) / (1 + (w * Material.watertau)**2))
        es = (1.01 + 0.44 * self.rs)**2 - 0.062 
        b1 = 1.2748 - 0.519 * self.S - 0.152 * self.C
        b2 = 1.33797 - 0.603 * self.S - 0.166 * self.C

        sigf = 0.0467 + 0.2204 * self.rb - 0.411 * self.S + 0.6614 * self.C

        fwbins = np.linspace(self.fw[0], self.fw[1], nmat)
        fw = fwbins + (fwbins[1] - fwbins[0]) / 2

        fwiter = np.nditer(fw, flags=['c_index'])
        while not fwiter.finished:
            er = (1 + (self.rb / self.rs) * ((es**a) - 1) + (fwiter[0]**b1 * erealw**a) - fwiter[0]) ** (1 / a)
            er = 1.15 * er - 0.68
            eri = er - (fwiter[0]**(b2 / a) * Material.waterdeltaer)
            sig = fwiter[0]**(b2 / a) * ((sigf * (self.rs - self.rb)) / (self.rs * fwiter[0]))

            digitscount =  len(str(int(nmat)))
            materialID = '|{}_{}|'.format(fractalboxname, str(fwiter.index + 1).zfill(digitscount))
            m = Material(materialID)
            m.type = 'debye'
            m.poles = 1
            if m.poles > Material.maxpoles:
                Material.maxpoles = m.poles
            m.er = eri
            m.se = sig
            m.deltaer.append(er - eri)
            m.tau.append(Material.watertau)

            scene.materials.append(m)

            fwiter.iternext()

if __name__ == "__main__":
    scene = Scene()
    soil = Peplinskisoil(0,0.5,0.5,2,2.66,0.001,0.25)
    soil.calculate_er(20,scene,"soil")
    scene.print_materials()
