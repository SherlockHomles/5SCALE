from Tree import Tree, ConeTree, SpheroidTree
from math import sin, cos, acos, tan, atan, asin, pi, sqrt, radians
from abc import abstractmethod, ABC
from Leaf import Needle


def _triangle(xa, ya, xb, yb, xc, yc):
    '''
    subroutine to calculate the area of any triangle
    @param xa: x of point A
    @param ya: y of point A
    @param xb: x of point B
    @param yb: y of point B
    @param xc: x of point C
    @param yc: y of point C
    @return:
    '''
    a = sqrt((yc - yb) * (yc - yb) + (xc - xb) * (xc - xb))
    b = sqrt((yc - ya) * (yc - ya) + (xc - xa) * (xc - xa))
    c = sqrt((yb - ya) * (yb - ya) + (xb - xa) * (xb - xa))

    cc = (a ** 2 + b ** 2 - c ** 2) / (2. * a * b)
    C = acos(cc)
    answer = 0.5 * a * b * sin(C)
    return answer


def _equation1(x, xa):
    '''solution for an integral'''
    if xa * xa > x * x:
        answer = x * sqrt(xa * xa - x * x) + xa * xa * asin(x / xa)
    else:
        answer = 0
    return answer


class GeoVI(object):
    '''
    This class represents view and illumination geometries
    Attributes:
        SZA: sun zenith angle, in degree
        VZA: view zenith angle, in degree
        phi: relative azimuth angle between viewer and sun, in degree
        xi: the angle between sun and viewer
    '''

    def __init__(self, SZA: float, VZA: float, phi: float):
        self.SZA = SZA
        self.VZA = VZA
        self.phi = phi

    @property
    def SZA(self):
        return radians(self.__SZA)

    @SZA.setter
    def SZA(self, value: float):
        self.__SZA = value

    @property
    def VZA(self):
        return radians(self.__VZA)

    @VZA.setter
    def VZA(self, value: float):
        self.__VZA = value

    @property
    def phi(self):
        return radians(self.__phi)

    @phi.setter
    def phi(self, value: float):
        self.__phi = value

    @property
    def xi(self):
        sza = self.SZA
        vza = self.VZA
        PHI = self.phi
        return GeoVI.XI(sza, vza, PHI)

    @staticmethod
    def XI(sza, vza, PHI):
        '''calculate the scattering angle i.e. phase angle'''
        uy = sin(sza)
        uz = cos(sza)
        vy = sin(vza) * cos(PHI)
        vz = cos(vza)
        return acos(uy * vy + uz * vz)


class AOPSpecies(ABC):
    '''
    the abstract class ensembles apparent optical properties(AOP) of a tree that varies across tree species, AOP
    is optical properties depend on view and illumination geometries, thus cannot be changed
    Attributes:
        tab (tac): tree cylinder(cone) projection to viewer (Eq.12 & 13)
        tib (tic): sunlit tree cylinder(cone) projection to viewer (Eq.14 & Appendix)
        Vg_0 (Vgc): projection area of cylinder (cone) on the ground (Eq.11)
        Sg_0 (Sgc): sunlit projection area of cylinder (cone) on the ground (Eq.10)
        Vg_0_mean (Vgc_mean): projection area of cylinder (cone) on the ground on zenith angle
        Sv (Ss): mean path length of the viewer (solar) beam through a tree (Eq.25)
        H: height in Eq.50

    Methods:
        _ta: calculate tac and tab
        _ti: calculate tic and tib
        _gs: calculate Vg_0, Vgc, Sg_0, Sgc, Vg_0_mean, Vgc_mean

    Reference:
        Chen, J. M., & Leblanc, S. G. (1997). A four-scale bidirectional reflectance model based on canopy
         architecture. IEEE Transactions on Geoscience and Remote Sensing, 35(5), 1316-1337.
    '''

    @property
    @abstractmethod
    def tab(self):
        pass

    @property
    @abstractmethod
    def tac(self):
        pass

    @property
    @abstractmethod
    def tib(self):
        pass

    @property
    @abstractmethod
    def tic(self):
        pass

    @property
    @abstractmethod
    def Vg_0(self):
        pass

    @property
    @abstractmethod
    def Vgc(self):
        pass

    @property
    @abstractmethod
    def Sg_0(self):
        pass

    @property
    @abstractmethod
    def Sgc(self):
        pass

    @property
    @abstractmethod
    def Vg_0_mean(self):
        pass

    @property
    @abstractmethod
    def Sv(self):  # mean path length of the solar beam through a tree
        pass

    @property
    @abstractmethod
    def Ss(self):  # mean path length of the view beam through a tree
        pass

    @property
    @abstractmethod
    def H(self):  # height in Eq.50
        pass

    @abstractmethod
    def _ta(self):
        '''calculate crown projection to viewer and the crown volume'''
        pass

    @abstractmethod
    def _ti(self):
        '''calculate tic and tib, an explanation of this function can be found in appendix A'''
        pass

    @abstractmethod
    def _gs(self, option: str):
        '''calculate Vg_0, Vgc, Sg_0, Sgc, Vg_0_mean, Vgc_mean'''
        pass


class AOPTree(object):
    '''
    this class stores apprarent optical properties of a Tree that don't depend oon tree species, these properties are
    sun and view dependent
    Attributes:
        tree: a tree object contains tree information such as cone height, cylinder height .etc
        geovi: an object contains information about view and illumination geometries
        A, C: two coefficients required to calculate tree projection coefficient G (Eq.22)
    '''

    def __init__(self, tree: Tree, geovi: GeoVI) -> None:
        self.tree = tree
        self.geovi = geovi

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, value: Tree):
        self._tree = value

    @property
    def geovi(self):
        return self._geovi

    @geovi.setter
    def geovi(self, value: GeoVI):
        self._geovi = value


class AOPConeTree(AOPTree, AOPSpecies):
    '''
    this class stores apprarent optical properties of a Cone Tree, these properties are sun and view dependent
    Attributes:
        tree: a tree object contains tree information such as cone height, cylinder height .etc
        geovi: an object contains information about view and illumination geometries
        tab (tac): tree cylinder(cone) projection to viewer
        tib (tic): sunlit tree cylinder(cone) projection to viewer
        Vg_0 (Vgc): projection area of cylinder (cone) on the ground (Eq.11)
        Sg_0 (Sgc): sunlit projection area of cylinder (cone) on the ground (Eq.10)
        Vg_0_mean (Vgc_mean): projection area of cylinder (cone) on the ground on zenith angle
        Sv (Ss): mean path length of the viewer (solar) beam through a tree (Eq.25)
    '''

    def __init__(self, tree: ConeTree, geovi: GeoVI):
        AOPTree.__init__(self, tree, geovi)

    @AOPTree.tree.setter
    def tree(self, tree: ConeTree):
        AOPTree.tree.fset(self, tree)

    @property
    def tab(self):
        Tab, Tac = self._ta()
        return Tab

    @property
    def tac(self):
        Tab, Tac = self._ta()
        return Tac

    @property
    def tib(self):
        xi = self.geovi.xi
        if 0.000001 >= xi >= 0:
            Tib = self.tab
        else:
            Tib, Tic = self._ti
        return Tib

    @property
    def tic(self):
        xi = self.geovi.xi
        if 0.000001 >= xi >= 0:
            Tic = self.tac
        else:
            Tib, Tic = self._ti
        return Tic

    @property
    def Vg_0(self):
        g_s, g_sc = self._gs('VZA')
        return g_s

    @property
    def Vgc(self):
        #g_s, g_sc = self._gs('VZA')
        g_s, g_sc = self._gs('LAI')
        return g_sc

    @property
    def Sg_0(self):
        g_s, g_sc = self._gs('SZA')
        return g_s

    @property
    def Sgc(self):
        g_s, g_sc = self._gs('SZA')
        return g_sc

    @property
    def Vg_0_mean(self):
        g_s, g_sc = self._gs('LAI')
        return g_s

    #@property
    #def Vgc_mean(self):
    #    g_s, g_sc = self._gs('LAI')
    #    return g_sc

    @property
    def H(self):
        SZA = self.geovi.SZA
        if SZA == pi or SZA == 0:
            h = 0
        else:
            Ha = self.tree.Ha
            Hb = self.tree.Hb
            Hc = self.tree.Hc
            h = (Ha + Hb + Hc / 3) / cos(SZA)
        return h

    @property
    def Sv(self):
        V = self.tree.V
        Vg_0 = self.Vg_0
        VZA = self.geovi.VZA
        s = V / (Vg_0 * cos(VZA))
        return s

    @property
    def Ss(self):
        V = self.tree.V
        Sg_0 = self.Sg_0
        SZA = self.geovi.SZA
        s = V / (Sg_0 * cos(SZA))
        return s

    def _ta(self):
        '''calculate crown projection to viewer and the crown volume'''
        gamma, xa, xb, yd = 0., 0., 0., 0.
        alpha = self.tree.alpha
        R = self.tree.R
        Hb = self.tree.Hb
        Hc = self.tree.Hc
        VZA = self.geovi.VZA
        if alpha < VZA: gamma - asin(tan(alpha) / tan(VZA))
        if VZA == 0:
            tac = pi * R ** 2
            tab = 0
        elif 0 < VZA < alpha:
            tac = pi * R * R * cos(VZA)
            tab = 2 * R * Hb * sin(VZA)
        else:
            xa = R * cos(VZA)
            xb = R * sin(VZA) / tan(alpha)
            yd = R * (1 - 2 * xa * xa / (xb * xb + xa * xa))
            tac = pi * R * xa
            tac = tac + 2 * xb / R * (R * yd - yd * yd / 2)
            tac = tac - xa / (R * _equation1(yd, R))
            tab = 2 * sin(VZA) * R * Hb
        return tab, tac

    def _gs(self, option: str):
        ''' Subroutine that calculates crown projection on the ground'''
        if 'VZA' == option.upper():
            ZA = self.geovi.VZA
        elif 'SZA' == option.upper():
            ZA = self.geovi.SZA
        elif 'LAI' == option.upper():
            LAI = self.tree.LAI
            ZA = acos(0.537 + 0.025 * LAI)
        else:
            raise ValueError('Error: wrong option, it must be one of the three: VZA, SZA or LAI')
        gamma, g_sc, g_s = 0., 0., 0.
        alpha = self._tree.alpha
        R = self.tree.R
        Hb = self.tree.Hb
        Hc = self.tree.Hc
        if ZA == pi / 2:
            ZA = pi / 2. - 0.00000000001

        if ZA == 0:
            g_s = pi * R * R
        elif 0 < ZA < alpha:
            g_s = 2 * tan(ZA) * R * Hb + pi * R * R
        elif alpha <= ZA < pi / 2:
            gamma = asin(tan(alpha) / tan(ZA))
            g_s = tan(ZA) * Hb * R * 2 + (1. / tan(gamma) + pi / 2 + gamma) * R * R
            g_sc = (1. / tan(gamma) - pi / 2 + gamma) * R * R
        return g_s, g_sc

    @property
    def _ti(self):
        '''calculate tic and tib, an explanation of this function can be found in appendix A'''
        alpha = self.tree.alpha
        R = self.tree.R
        Hb = self.tree.Hb
        Hc = self.tree.Hc
        SZA = self.geovi.SZA
        VZA = self.geovi.VZA
        phi = self.geovi.phi
        m, a, b, c, xa, ya, xb, xd = 0., 0., 0., 0., 0., 0., 0., 0.
        yd, xe, xf, yf, xf, xg, xf2, yf2 = 0., 0., 0., 0., 0., 0., 0., 0.
        B, C, xe2, ye2, m1, m2, b1, b2 = 0., 0., 0., 0., 0., 0., 0., 0.
        A1, A2, A3, A4, arg1 = 0., 0., 0., 0., 0.
        gamma = pi / 2
        if SZA == 0 and phi == 0:
            phi = pi
        if alpha <= SZA:
            arg1 = tan(alpha) / tan(SZA)
            gamma = asin(arg1)
        xa = R * cos(VZA)
        ya = 0
        yf = R * cos(gamma - phi)
        xf = xa * sin(gamma - phi)
        xb = R * sin(VZA) / tan(alpha)
        xg = xb
        tib = 2 * R * sin(VZA) * Hb * (1 - phi / pi)
        tic = 0
        '''case 1 2 3'''
        if VZA == 0:
            tic = (pi / 2 + gamma) * R * xa

        '''case 4 5'''
        if 0 <= SZA < alpha < VZA and VZA > 0:
            tic = pi * R * xa

        '''case 6'''
        if pi / 2 > SZA >= alpha >= VZA > 0:
            yf2 = -R * cos(gamma + phi)
            xf2 = xa * sin(gamma + phi)
            yf = R * cos(gamma - phi)
            xf = xa * sin(gamma - phi)
            xg = xb
            if yf2 < 0:
                A1 = -xa / (2. * R) * _equation1(yf2, R)
                A1 = A1 + yf2 * yf2 / (2. * yf2 / xf2)
            if xf > 0:
                A2 = -xa / (2. * R) * _equation1(yf, R)
                A2 = A2 - yf * yf / (2. * yf / xf)
            if xf > 0 and yf2 > 0.:
                A2 = -xa / (2. * R) * _equation1(yf2, R)
                A2 = A2 + yf2 * yf2 / (2. * yf2 / xf2)
            if xf <= 0 and yf2 > 0.:
                A2 = pi * R * xa / 4.
                A2 = A2 - xa / (2. * R) * _equation1(yf2, R)
                A2 = A2 + yf2 * yf2 / (2. * yf2 / xf2)
            if xf <= 0. and yf2 <= 0: A2 = pi * R * xa / 4
            if xf <= 0. and xf2 <= 0: A2 = 0

            if xf <= 0 and yf > 0:
                A3 = -R / (2. * xa) * _equation1(xf, xa)
                A3 = A3 + yf / xf * xf * xf / 2.

            if xf <= 0. and yf > 0 and xf2 < 0:
                A3 = A3 - R / (2. * xa) * _equation1(xf2, xa)
                A3 += yf / xf * xf * xf / 2.
            if xf <= 0 and yf < 0 and xf2 < 0:
                A3 = pi * R * xa / 4
                A3 = A3 + R / (2. * xa) * _equation1(xf2, xa)
                A3 = A3 + yf / xf * xf * xf / 2.

            if xf <= 0 and yf <= 0 and xf2 > 0:
                A3 = pi * R * xa / 4
            if xf <= 0 and yf < 0:
                A4 = -xa / (2 * R) * _equation1(yf, R)
                A4 = A4 - yf * yf / (2 * yf / xf)

            if yf > 0: B = _triangle(xf, yf, xg, 0., 0., 0.)
            if yf2 < 0: C = _triangle(xf2, yf2, xg, 0., 0., 0.)
            if yf2 > 0: C = - _triangle(xf2, yf2, xg, 0., 0., 0.)
            if yf < 0: B = -_triangle(xf, yf, xg, 0., 0., 0.)

            if yf > -0.00000000001 or yf < 0.00000000001:
                B = 0
            if yf2 > -0.00000000001 or yf2 < 0.00000000001:
                C = 0
            tic = pi * R * xa
            tic = tic - A1
            tic = tic - A2
            tic = tic - A3
            tic = tic - A4
            tic = tic + B
            tic = tic + C

        '''case 7 8'''
        if SZA <= alpha < VZA <= pi / 2.:
            yd = R * (1 - 2 * xa * xa / (xb * xb + xa * xa))
            tic = pi * R * xa
            tic = tic + 2 * xb / R * (R * yd - yd * yd / 2)
            tic = tic - xa / R * _equation1(yd, R)

        '''case 9'''
        if alpha <= SZA < pi / 2 and alpha < VZA <= pi / 2:
            yd = R * (1 - 2 * xa * xa / (xb * xb + xa * xa))
            xd = 2 * xa * xa * xb / (xb * xb + xa * xa)
            yf2 = -R * cos(gamma + phi)
            xf2 = xa * sin(gamma + phi)
            xg = xb
            m1 = yf / (xf - xg)
            b1 = -m1 * xg
            a = R * R + xa * xa * m1 * m1
            b = 2. * xa * xa * m1 * b1
            c = xa * xa * (b1 * b1 - R * R)
            if b * b > 4. * a * c:
                xe = (-b + sqrt(b * b - 4. * a * c)) / (2. * a)
            else:
                xe = -b / (2. * a)
            ye = m1 * xe + b1

            m2 = yf2 / (xf2 - xg)
            b2 = yf2 - m2 * xf2
            a = R * R + xa * xa * m2 * m2
            b = 2. * xa * xa * m2 * b2
            if b * b > 4. * a * c:
                xe2 = (-b + sqrt(b * b - 4. * a * c)) / (2. * a)
            else:
                xe2 = -b / (2. * a)
            ye2 = m2 * xe + b2

            if yf2 < 0:
                A1 = xa / (2. * R) * (_equation1(ye2, R) - _equation1(yf2, R)) - (
                        ye2 * ye2 - yf2 * yf2) / (
                             2. * m2) - (
                             -b2 * ye2 / m2 + b2 * yf2 / m2)
            else:
                A1 = 0

            if xf > 0 and yf > 0:
                A2 = R / (2. * xa) * _equation1(xe, xa) - m1 * xe * xe / 2. - b1 * xe - R / (
                        2. * xa) * _equation1(xf,
                                              xa) + m1 * xf * xf / 2. + b1 * xf
            if xf < 0 < yf:
                A3 = -R / (2 * xa) * _equation1(xf, xa) + m1 * xf * xf / 2. + b1 * xf

            if xf < 0 and yf <= 0:
                A3 = pi * R * xa / 4.

            if yf2 > 0 >= xf2:
                A3 = A3 - (-R / (2. * xa) * _equation1(xf2, xa) + m2 * xf2 * xf2 / 2. + b2 * xf2)

            if yf < 0 and xf < 0:
                A4 = -xa / (2. * R) * _equation1(yf, R) - (yf * yf) / (2. * m1) + (b1 * yf / m1) + (
                        b1 * b1) / (
                             2. * m1) - (
                             b1 * b1 / m1)
            if yf < 0 and xf < 0:
                A1 = -xa / (2. * R) * _equation1(ye, R) + xe * (ye - b1) / 2.
            C = 0
            if xf < 0 < xf2:
                m = (yd - ye) / (xd - xe)
                b = yd - m * xd
                C = _triangle(xe, ye, xd, yd, xg, 0.) - xa / (2. * R) * _equation1(yd, R) - _equation1(ye,
                                                                                                       R) + (
                            yd * yd / 2. - yd * b) / m - (ye * ye / 2. - ye * b) / m

            if xf < 0 and xf <= 0 and yf2 > 0:
                if xe2 - xe < 0.0000001 and xe2 - xe > -0.0000001:
                    m = 0
                    b = ye
                    C = _triangle(xe, ye, xe2, ye2, xg, 0.) - 2 * (
                            xa / (2. * R) * _equation1(ye2, R) - xe * ye2)
                else:
                    m = (ye2 - ye) / (xe2 - xe)
                    b = ye - m * xe
                    C = _triangle(xe, ye, xe2, ye2, xg, 0.) - xa / (2. * R) * (
                            _equation1(ye2, R) - _equation1(ye, R)) + (
                                ye2 * ye2 / 2. - ye2 * b) / m - (ye * ye / 2. - ye * b) / m

            tic = pi * R * xa
            tic = tic + 2 * xb * (R * yd - yd * yd / 2.) / R
            tic = tic - xa / R * _equation1(yd, R)
            tic = tic - A1
            tic = tic - A2
            tic = tic - A3
            tic = tic - A4
            tic = tic - C
        return tib, tic


class AOPSpheroidTree(AOPTree, AOPSpecies):
    '''
    this class stores apparent optical properties(AOP) of a Cone Tree, these properties are sun and view dependent
    Args:
        tree: a tree object contains tree information such as cone height, cylinder height etc.
        geovi: an object contains information about view and illumination geometries
        tab: crown projection to viewer
        tic: sunlit crown projection to viewer
    '''

    def __init__(self, tree: SpheroidTree, geovi: GeoVI):
        AOPTree.__init__(self, tree, geovi)

    @AOPTree.tree.setter
    def tree(self, value: SpheroidTree):
        AOPTree.tree.fset(self, value)

    @property
    def tab(self):
        Tab = self._ta()
        return Tab

    @property
    def tac(self):
        return 0

    @property
    def tib(self):
        xi = self.geovi.xi
        if 0.000001 >= xi >= 0:
            Tib = self.tab
        else:
            Tib = self._ti()
        return Tib

    @property
    def tic(self):
        xi = self.geovi.xi
        if 0.000001 >= xi >= 0:
            Tic = self.tac
        else:
            Tic = 0
        return Tic

    @property
    def Vg_0(self):
        g_s = self._gs('VZA')
        return g_s

    @property
    def Vgc(self):
        return 0

    @property
    def Sg_0(self):
        g_s = self._gs('SZA')
        return g_s

    @property
    def Sgc(self):
        return 0

    @property
    def Vg_0_mean(self):
        g_s = self._gs('LAI')
        return g_s

    @property
    def Vgc_mean(self):
        return 0

    @property
    def Sv(self):
        V = self.tree.V
        Vg_0 = self.Vg_0
        VZA = self.geovi.VZA
        s = V / (Vg_0 * cos(VZA))
        return s

    @property
    def Ss(self):
        V = self.tree.V
        Sg_0 = self.Sg_0
        SZA = self.geovi.SZA
        s = V / (Sg_0 * cos(SZA))
        return s

    @property
    def H(self):
        if not hasattr(self, '_H'):
            SZA = self.geovi.SZA
            if SZA == pi or SZA == 0:
                self._H = 0
            else:
                Ha = self.tree.Ha
                Hb = self.tree.Hb
                Hc = self.tree.Hc
                self._H = (Ha + Hb + Hc / 3) / cos(SZA)
        return self._H

    '''subroutine to calculate crown projection to the viewer for diciduous(spheroid) shape'''

    def _ta(self):
        R = self.tree.R
        Hb = self.tree.Hb
        VZA = self.geovi.VZA
        tab = pi * R * (Hb / 2. * sin(VZA) + R * cos(VZA))
        return tab

    def _ti(self):
        '''subroutine to calculate sunlit crown proportion for diciduous(spheroid) shape'''
        R = self.tree.R
        Hb = self.tree.Hb
        VZA = self.geovi.VZA
        SZA = self.geovi.SZA
        phi = self.geovi.phi
        VZA_prime = atan(Hb / (2 * R) * tan(VZA))
        SZA_prime = atan(Hb / (2 * R) * tan(SZA))
        cs_prime = cos(SZA_prime) * cos(VZA_prime) + sin(SZA_prime) * sin(VZA_prime * cos(phi))
        tab = self.tab
        tib = tab * 0.5 * (1 + cs_prime)
        return tib

    def _gs(self, option: str):
        '''subroutine to calculate crown projection on the ground for diciduous(spheroid) shape'''
        if 'VZA' == option.upper():
            ZA = self.geovi.VZA
        elif 'SZA' == option.upper():
            ZA = self.geovi.SZA
        elif 'LAI' == option.upper():
            LAI = self.tree.LAI
            ZA = acos(0.537 + 0.025 * LAI)
        else:
            raise ValueError('Error: wrong option, it must be one of the three: VZA, SZA or LAI')
        R = self.tree.R
        Hb = self.tree.Hb
        b = Hb / 2
        if ZA == pi / 2: ZA = ZA - 0.0000000001
        ZA_prime = atan(b / R * tan(ZA))
        g_s = pi * R * R / (cos(ZA_prime))
        return g_s
