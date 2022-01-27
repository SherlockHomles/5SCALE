from math import sin, cos, tan, asin, pi, sqrt, acos, atan, radians
from abc import ABC, abstractmethod
from Leaf import Leaf, Needle


class TreeSpecies(ABC):
    '''
    the abstract class contains tree properties that may vary across species
    Attributes:
        shape: the shape of the tree, 'CONE_CYLINDER' or 'SPHEROID'
        V: volume of a tree crown
        Hc: cone height
        alpha: apex angle of the cone
    '''

    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    @abstractmethod
    def V(self):
        pass

    @property
    @abstractmethod
    def Hc(self):
        pass

    @property
    @abstractmethod
    def alpha(self):
        pass


class Tree(object):
    '''
    this class stores common properties of a tree
    Attributes:
        leaf: leaf in the tree
        R: radius of the cylinder
        alpha: apex angle of the cone on top of the cylinder, it must be in degree, it will be changed to radians in getter
        Ha: stick height
        Hb: height of the cylinder
        LAI: leaf area index
        Omega_E: clumping index
        gamma_E: needle to shoot ratio, default: 1 (broadleaf)
        ge_choice: whether the tree has branches or not
        alpha_l: leaf angle if the tree has branches,default:0, it will be converted into radians in getter
        alpha_b: branch angle if the tree has branches, default:0, it will be converted into radians in getter
        Ll: branch area index if the tree has branches
        Rb: branch thickness if the tree has branches, default:10
        DeltaLAI: this parameter is used to increase LAI in multiple scattering in
    '''

    def __init__(self, leaf: Leaf, R: float, Ha: float, Hb: float, LAI: float, Omega_E: float, gamma_E: float = 1,
                 ge_choice: str = 'NO_BRANCH', alpha_l: float = 20, alpha_b: float = 10, Ll: float = 0.8,
                 Rb: float = 0.1, DeltaLAI: float = 0.2) -> None:
        self.leaf = leaf
        self.R = R
        self.Ha = Ha
        self.Hb = Hb
        self.LAI = LAI
        self.Omega_E = Omega_E
        self.gamma_E = gamma_E
        self.ge_choice = ge_choice
        self.alpha_l = alpha_l
        self.alpha_b = alpha_b
        self.Ll = Ll
        self.Rb = Rb
        self.DeltaLAI = DeltaLAI

    @property
    def DeltaLAI(self):
        return self._DeltaLAI

    @DeltaLAI.setter
    def DeltaLAI(self, value: float):
        if value < 0:
            raise ValueError('DeltaLAI must be larger than 0')
        self._DeltaLAI = value

    @property
    def ge_choice(self):
        return self._ge_choice

    @ge_choice.setter
    def ge_choice(self, value: str):
        if value.upper() not in ['BRANCH', 'NO_BRANCH']:
            raise ValueError(
                'ge_choice must be BRANCH or NO_BRANCH (case insensitive)')
        self._ge_choice = value.upper()

    @property
    def Ha(self):
        return self._Ha

    @Ha.setter
    def Ha(self, value: float):
        if value < 0:
            raise ValueError('Ha (stick height) must be larger than 0')
        self._Ha = value

    @property
    def leaf(self):
        return self._leaf

    @leaf.setter
    def leaf(self, value: Leaf):
        self._leaf = value

    @property
    def LAI(self):
        return self._LAI

    @LAI.setter
    def LAI(self, value: float):
        if value < 0:
            raise ValueError('LAI must be larger than 0')
        self._LAI = value

    @property
    def Omega_E(self):
        return self._Omega_E

    @Omega_E.setter
    def Omega_E(self, value: float):
        if 0 <= value <= 1:
            self._Omega_E = value
        else:
            raise ValueError('Clumping index (CI) should lie in 0-1')

    @property
    def gamma_E(self):
        return self._gamma_E

    @gamma_E.setter
    def gamma_E(self, value: float):
        if value < 0:
            raise ValueError('Needle to shoot ratio should larger than 0')
        self._gamma_E = value

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value: float):
        if value < 0:
            raise ValueError('Crown radius must be larger than 0')
        self._R = value

    @property
    def Hb(self):
        return self._Hb

    @Hb.setter
    def Hb(self, value: float):
        if value < 0:
            raise ValueError('Cylinder height be larger than 0')
        self._Hb = value

    @property
    def alpha_l(self):
        return radians(self._alpha_l)

    @alpha_l.setter
    def alpha_l(self, value: float):
        #if value < 0:
        #    raise ValueError('leaf angle should be larger than 0')
        self._alpha_l = value

    @property
    def alpha_b(self):
        return radians(self._alpha_b)

    @alpha_b.setter
    def alpha_b(self, value: float):
        if value < 0:
            raise ValueError('branch angle should be larger than 0')
        self._alpha_b = value

    @property
    def Ll(self):
        return self._Ll

    @Ll.setter
    def Ll(self, value: float):
        if value < 0:
            raise ValueError('branch area index should be larger than 0')
        self._Ll = value

    @property
    def Rb(self):
        return self._Rb

    @Rb.setter
    def Rb(self, value: float):
        if value < 0:
            raise ValueError('branch thickness should be larger than 0')
        self._Rb = value


class ConeTree(Tree, TreeSpecies):
    '''
    this denotes a tree with cone_cylinderial shape, inherited from tree
    Attributes:
        leaf: leaf in the tree
        R: radius of the cylinder
        alpha: apex angle of the cone on top of the cylinder, it must be in degree, it will be changed to radians in getter
        Ha: stick height
        Hb: height of the cylinder
        LAI: leaf area index
        Omega_E: clumping index
        gamma_E: needle to shoot ratio, default: 1 (broadleaf)
        ge_choice: whether the tree has branches or not
        shape: shape of the tree, it cannot be changed
        alpha_l: leaf angle if the tree has branches,default:0
        alpha_b: branch angle if the tree has branches, default:0
        Ll: branch area index if the tree has branches
        Rb: branch thickness if the tree has branches, default:10
    '''

    def __init__(self, leaf: Leaf, R: float, alpha: float, Ha: float, Hb: float, LAI: float, Omega_E: float,
                 gamma_E: float = 1,
                 ge_choice: str = 'NO_BRANCH', alpha_l: float = 20, alpha_b: float = 10, Ll: float = 0.8,
                 Rb: float = 0.1, DeltaLAI: float = 0.2):
        Tree.__init__(self, leaf, R, Ha, Hb, LAI, Omega_E, gamma_E, ge_choice, alpha_l, alpha_b, Ll, Rb, DeltaLAI)
        self.alpha = alpha
        self.__shape = 'CONE_CYLINDER'

    @property
    def shape(self):
        return self.__shape

    @property
    def Hc(self):
        return self.R / tan(self.alpha)

    @property
    def V(self):
        r = self.R
        hb = self.Hb
        hc = self.Hc
        volume = pi * r ** 2 * (hb + hc / 3)
        return volume

    @property
    def alpha(self):
        return radians(self._alpha)

    @alpha.setter
    def alpha(self, value: float):
        if value < 0:
            raise ValueError('Apex angle must be larger than 0')
        self._alpha = value


class SpheroidTree(Tree, TreeSpecies):
    '''
    this denotes a tree with spherical shape, inherited from tree
    Attributes:
        leaf: leaf in the tree
        R: radius of the cylinder
        alpha: apex angle of the cone on top of the cylinder, it must be in degree, it will be changed to radians in getter
        Ha: stick height
        Hb: height of the cylinder
        LAI: leaf area index
        Omega_E: clumping index
        gamma_E: needle to shoot ratio, default: 1 (broadleaf)
        ge_choice: whether the tree has branches or not
        shape: shape of the tree, it cannot be changed
        Ll: branch area index if the tree has branches
        Rb: branch thickness if the tree has branches, default:10
    '''

    def __init__(self, leaf: Leaf, R: float, Ha: float, Hb: float, LAI: float, Omega_E: float, gamma_E: float = 1,
                 ge_choice: str = 'NO_BRANCH', alpha_l: float = 20, alpha_b: float = 10, Ll: float = 0.8,
                 Rb: float = 0.1, DeltaLAI: float = 0.2):
        Tree.__init__(self, leaf, R, Ha, Hb, LAI, Omega_E, gamma_E, ge_choice, alpha_l, alpha_b, Ll, Rb, DeltaLAI)
        self.__shape = 'SPHEROID'

    @property
    def shape(self):
        return self.__shape

    @property
    def V(self):
        r = self.R
        hb = self.Hb
        volume = 2 / 3. * pi * r * r * hb
        return volume

    @property
    def Hc(self):
        return 0

    @property
    def alpha(self):
        return 0


if __name__ == '__main__':
    leaf = Needle(40, 1.6, 0.045, 0.0005, 2., 200, 40, 1, 100)
    ct = ConeTree(leaf, 1, 13, 5, 3.5, 0.98, 1.41, 'NO_BRANCH')
    print('original LAI: {:.1f}'.format(ct.LAI))
    ct.LAI = 5
    print('current LAI: {:.1f}'.format(ct.LAI))
