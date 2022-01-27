import time
from typing import Optional

import pandas as pd
from Domain import Domain
from AOPTree import GeoVI, AOPTree, AOPConeTree, AOPSpheroidTree
from Tree import Tree, ConeTree, SpheroidTree
from Leaf import Leaf, Broadleaf, Needle
from math import exp, cos, sin, pi, tan, acos, log, sqrt, atan, radians, degrees, asin
from scipy.special import comb
from scipy import integrate
import numpy as np


class AOPDomain(object):
    '''
    the class calculates apparent optical properties of a domain
    Attributes:
        geovi: an object contains information about view and illumination geometries
        domain: a class contains inherent properties of a domain
        AOP_tree: returns an AOP class inherited from AOPTree subclass, this parameter contains AOP of a tree
        Vg: viewed ground, this parameter derived from AOPTree.Vg_0 after resizing the tree(Eq.21)
        Sg: viewed ground, this parameter derived from AOPTree.Sg_0 after resizing the tree(Eq.21)
        PgapV: gap fraction of a tree crown viewed from viewer
        PgapS: gap fraction of a tree crown viewed from sun
        Pgap0: gap fraction of a tree crown viewed from nadir
        Gv: G(SZA)*Omega_E/gamma_E in Eq.22
        Gs: G(VZA)*Omega_E/gamma_E in Eq.22
        GFoliage: G(VZA) in Eq.22 when tree has no branches, it equals to A*VZA+C
        A, C: constants required to calculate GFoliage, Gv, Gs, PgapV, PgapS when tree has no branches
        OmegaT: clumping index of a tree, Eq.43
        E_r: mean gap between tree crowns when trees are subject to poisson distribution, Eq.41/42
        Wt: characteristic width of a tree crown projected on the ground,Eq.41
        Lt: clumping adjusted projected tree crown area index, Eq.42

    Methods:
        _resize_tree: adjust Vg and Sg with tree size
        _Pgap: calculate gap fraction according to option
        _G: calculate either Gv or Gs, depending on option
        _p_gap_ax_b: calculate gap fraction when tree has no branches

    '''

    def __init__(self, domain: Domain, geovi: GeoVI):
        self.geovi = geovi
        self.domain = domain

    @property
    def geovi(self):
        return self._geovi

    @geovi.setter
    def geovi(self, value):
        self._geovi = value

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, value):
        self._domain = value

    @property
    def AOP_tree(self):
        if not hasattr(self, '_AOP_tree'):
            tree = self.domain.tree
            geovi = self.geovi
            if type(tree) == ConeTree:
                self._AOP_tree = AOPConeTree(tree, geovi)
            elif type(tree) == SpheroidTree:
                self._AOP_tree = AOPSpheroidTree(tree, geovi)
            else:
                raise ValueError('tree type error: it must be ConeTree or SpheroidTree')
        return self._AOP_tree

    @property
    def Vg(self):
        return self._resize_tree('VZA')

    @property
    def Sg(self):
        return self._resize_tree('SZA')

    @property
    def GFoliage(self):  # G in Eq.22
        ge_choice = self.domain.tree.ge_choice
        za = self.geovi.VZA
        if 'NO_BRANCH' == ge_choice:  # if tree does not have branches
            a = self.A
            c = self.C
            return a * za + c
        else:  # if tree has branches
            alpha_l = self.domain.tree.alpha_l
            ratio = self.domain.tree.leaf.RATIO
            if alpha_l > 0:
                if za <= pi / 2 - alpha_l:
                    gl = cos(alpha_l) * cos(za)
                else:
                    xl = 1 / tan(alpha_l) / tan(za)
                    xl = acos(xl)
                    gl = cos(alpha_l) * cos(za) * (1. + 2. * (tan(xl) - xl) / pi)
                agl = acos(gl)
                gl = gl + sin(agl) * ratio
            else:
                gl = 0.5
            return gl

    @property
    def lo(self):  # lo in Eq.23
        LAI = self.domain.tree.LAI
        b = self.domain.area
        d = self.domain.n_tree
        za = self.geovi.SZA
        sg_0 = self.AOP_tree.Sg_0
        return LAI * b / (d * sg_0 * cos(za))

    @property
    def A(self):  # coefficient to calculate GFoliage, this method ensures it is a final constant
        return 0.0

    @property
    def C(self):  # coefficient to calculate GFoliage, this method ensures it is a final constant
        return 0.5

    @property
    def PgapV(self):  # gap fraction from viewer in one crown
        return self._Pgap('VZA')

    @property
    def PgapS(self):  # gap fraction from sun in one crown
        return self._Pgap('SZA')

    @property
    def Pgap0(self):  # gap fraction from nadir in one crown
        return self._Pgap('0')

    @property
    def PgapV_mean(self):  # mean gap fraction in one crown viewed from viewer
        return self._Pgap('LAI')

    @property
    def Gv(self):
        return self._G('VZA')

    @property
    def Gs(self):
        return self._G('SZA')

    @property
    def OmegaT(self):
        '''
        calculate tree clumping index
        @return:clumping index, i.e. Eq.43
        '''
        return log(self.Pig) / log(self.Pig_poisson)

    @property
    def E_r(self):
        '''
        calculate mean gap between tree crowns when trees are subject to poisson distribution
        @return:Eq.41/Eq.42
        '''
        return self._distance()
        # if self.domain.distribution == 'POISSON':
        #    return self._distance()
        # else:
        #    return None

    @property
    def Wt(self):
        '''
        characteristic width of a tree crown projected on the ground
        @return:Eq.41
        '''
        sg_0 = self.AOP_tree.Sg_0
        if sg_0 < 0:
            return 0
        else:
            return sqrt(sg_0)

    @property
    def Lt(self):
        '''
        clumping adjusted projected tree crown area index
        @return:Eq.42
        '''
        D = self.domain.n_tree
        B = self.domain.area
        OmegaT = self.OmegaT
        sg_0 = self.AOP_tree.Sg_0
        return D * sg_0 / B * OmegaT

    @property
    def Lambda_m(self):
        '''
        @return:Eq.52
        '''
        if not hasattr(self, '_Lambda_m'):
            H = self.AOP_tree.H
            xi = self.geovi.xi
            if xi < pi:
                return H * tan(xi)
                self._Lambda_m = H * tan(xi)
            else:
                self._Lambda_m = 0
        return self._Lambda_m

    @property
    def PSG0_VIEW(self):  # mean gap fraction in one crown viewed from viewer
        Vg = self.Vg
        D = self.domain.n_tree  # number of trees in the domain
        B = self.domain.area  # domain size
        PgapV = self.PgapV
        value = 1 - Vg * D / B * (1 - PgapV)
        if value < 0:
            value = 0
        return value

    @property
    def PSG0_SUN(self):  # mean gap fraction in one crown viewed from viewer
        Sg = self.Sg
        D = self.domain.n_tree  # number of trees in the domain
        B = self.domain.area  # domain size
        PgapS = self.PgapS
        value = 1 - Sg * D / B * (1 - PgapS)
        if value < 0:
            value = 0
        return value

    @property
    def PSG_HOT0(self):  # serve as a hotspot kernel
        R = self.domain.tree.R
        D = self.domain.n_tree  # number of trees in the domain
        B = self.domain.area  # domain size
        Pgap0 = self.Pgap0
        value = 1 - pi * R ** 2 * D / B * (1 - Pgap0)
        return value

    @property
    def PVG(self):  # Eq.26
        '''
        @return: overlap_v1ping prob of Vg,i.e. Eq.26, this param does not consider vertical overlap
        '''
        return self._overlap_v1('VZA', 'Pvg')

    @property
    def Pvg(self):
        '''
        @return: overlap_v1ping prob of Vg,i.e. Eq.26, this param considers vertical overlap
        '''
        return self._fo('VZA', 'Pvg')

    @property
    def Pvc(self):
        '''
        @return: overlap_v1ping prob of Vgc,i.e. Eq.27
        '''
        shape = self.domain.tree.shape
        if 'CONE_CYLINDER' == shape:
            return self._overlap_v1('VZA', 'Pvc')
        else:
            return 0

    @property
    def PV(self):  # Eq. 29
        '''
        @return: overlap_v1ping prob of Vgb,i.e. Eq.29, this param does not consider vertical overlapping
        '''
        return self._overlap_v1('VZA', 'Pv')

    @property
    def Pv(self):  # Eq. 29
        '''
        @return: overlap_v1ping prob of Vgb,i.e. Eq.29, this param considers vertical overlapping
        '''
        return self._fo('VZA', 'Pv')

    @property
    def PIG(self):
        '''
        @return: overlap_v1ping prob of Sg, this param does not consider vertical overlapping
        '''
        return self._overlap_v1('SZA', 'Pig')

    @property
    def Pig(self):
        '''
        @return: overlap_v1ping prob of Sg, this param considers vertical overlapping
        '''
        return self._fo('SZA', 'Pig')

    @property
    def Pig_poisson(self):
        '''
        @return: overlap_v1ping prob of Sg
        '''
        return self._overlap_v1('SZA', 'Pig_poisson')

    @property
    def PS(self):
        '''
        @return: overlap_v1ping prob of Sgb, this param does not consider vertical overlapping
        '''
        if not hasattr(self, '_PS'):
            # self._PS = self._overlap_v1('SZA', 'Ps')
            self._PS = self.Pig * self.Pvg
        return self._PS

    @property
    def Ps(self):
        '''
        @return: overlap_v1ping prob of Sgb, this param considers vertical overlapping
        '''
        return self._fo('SZA', 'Ps')

    @property
    def Psc(self):
        '''
        @return: overlapping prob of Sgc
        '''
        shape = self.domain.tree.shape
        if 'CONE_CYLINDER' == shape:
            return self._overlap_v1('SZA', 'Psc')
        else:
            return 0

    @property
    def Pvg_mean(self):
        return self._overlap_v1('LAI')

    @property
    def Pvg_nadir(self):
        return self._overlap_v1('NADIR')

    @property
    def Ptreev(self):
        return self._overlap_v1('VZA', 'Ptreev')

    @property
    def Ptrees(self):
        return self._overlap_v1('SZA', 'Ptrees')

    @property
    def Pti(self):
        '''
        prob of seeing illuminated tree crowns
        @return: Eq.39
        '''
        r = self.domain.tree.R
        e_r = self.E_r
        phi = self.geovi.phi
        pv = self.Pv
        pvc = self.Pvc
        ps = self.Ps
        psc = self.Psc
        tic = self.AOP_tree.tic
        tib = self.AOP_tree.tib
        tab = self.AOP_tree.tab
        tac = self.AOP_tree.tac
        shape = self.domain.tree.shape

        x = atan(2 * r / e_r)
        if x < 0: x = -x
        f = 1 - 1. * phi / x
        qv = 1 - pv
        qs = 1 - ps
        qvc = 1 - pvc
        qsc = 1 - psc
        if pvc == 1: qvc = 0.000000000001
        if qs <= qv:
            aa = qs
        else:
            aa = qv
        if qsc <= qvc:
            aac = qsc
        else:
            aac = qvc
        if f < 0: f = 0
        if f > 1: f = 1
        if phi > x: f = 0

        if shape == 'CONE_CYLINDER':
            pti = ((qvc * qsc + (aac - qvc * qsc) * f) * tic + (qv * qs + (aa - qv * qs) * f) * tib) / (
                    tac * qvc + qv * tab)
        else:
            pti = (tib + tic) / (tab + tac)
        if pti > 1:
            raise ValueError('There must something wrong with Pti, it is larger than 1')
        if pti < 0:
            raise ValueError('There must something wrong with Pti, it is larger than 1')
        return pti

    @property
    def Viewed_shadow(self):
        return self._nadir('VIEWED_SHADOW')

    @property
    def Fd(self):
        return self._nadir('FD')

    @property
    def PT(self):
        if not hasattr(self, '_PT'):
            pt, fs = self._ptg_sub('CANOPY', 40000, 0.001)
            if pt < 0:
                pt = 0
            self._PT = pt
        return self._PT

    @property
    def Fs(self):
        pt, fs = self._ptg_sub('CANOPY', 40000, 0.001)
        return fs

    @property
    def ZG(self):
        if not hasattr(self, '_ZG'):
            zg, fn = self._ptg_sub('NADIR', 120000, 0.1)
            self._ZG = zg
        return self._ZG

    @property
    def Fn(self):
        zg, fn = self._ptg_sub('NADIR', 120000, 0.1)
        return fn

    @property
    def PG(self):
        if not hasattr(self, '_PG'):
            pg, ft = self._ptg_sub('GROUND', 20000, 0.01)
            self._PG = pg
        return self._PG

    @property
    def Ft(self):
        pg, ft = self._ptg_sub('GROUND', 20000, 0.01)
        return ft

    @property
    def OmegaTotal(self):
        num = log(self.Pvg_mean)
        LAI = self.domain.tree.LAI
        dem = -0.5 * LAI / (0.537 + 0.025 * LAI)
        return num / dem

    @property
    def Medium(self):
        cold = self.Pvg * (1 - self.Pig)
        hot = self.Viewed_shadow
        return hot - cold

    @property
    def H(self):
        return 1 / self.mu

    @property
    def mu(self):
        LAI = self.domain.tree.LAI
        b = self.domain.area
        d = self.domain.n_tree
        v = self.domain.tree.V
        return LAI * b / (d * v)

    @property
    def Ls(self):
        LAI_HOT_SPOT = 1
        return self.Gs * LAI_HOT_SPOT

    @property
    def Lo_90(self):
        lo_90, cs, cv = self._ls()
        return lo_90

    @property
    def Cs(self):
        if not hasattr(self, '_Cs'):
            lo_90, cs, cv = self._ls()
            self._Cs = cs
        return self._Cs

    @property
    def Cv(self):
        lo_90, cs, cv = self._ls()
        return cv

    @property
    def q1(self):
        cs = self.Cs
        cv = self.Cv
        lo_90 = self.Lo_90
        value = (1. - exp(-(cs * lo_90 + cv * lo_90))) * cs * cv / (cs + cv)  # Eq.56
        if value > 1:
            raise ValueError('q1 is larger than 1. Impossible!')
        return value

    @property
    def q2(self):
        cs = self.Cs
        cv = self.Cv
        lo_90 = self.Lo_90
        value = (exp(-cs * lo_90) - exp(-cv * lo_90)) * cs * cv / (cv - cs)  # Eq.61
        if value > 1:
            raise ValueError('q2 is larger than 1. Impossible!')
        return value

    @property
    def QQ1(self):
        return self._q('QQ1')

    @property
    def QQ2(self):
        return self._q('QQ2')

    @property
    def QQ1B(self):
        return self._q('QQ1B')

    @property
    def QQ2B(self):
        return self._q('QQ2B')

    @property
    def PT_Cold(self):
        qq1 = self.QQ1
        pti = self.Pti
        qq2 = self.QQ2
        value = qq1 * pti + (1 - pti) * qq2
        return value

    @property
    def ZT(self):
        if not hasattr(self, '_ZT'):
            zt = 1 - self.Pvg - self.PT
            self._ZT = zt
        return self._ZT

    def ro(self) -> np.array:  # simulate canopy reflectance
        self._ZG = self.Pvg - self.PG  # equation below Eq.72
        ZG = self.ZG
        PT = self.PT
        xi = self.geovi.xi
        if ZG < 0:
            self._ZG = 0
            self._PG = self.Pvg
            if xi > 0.0001:
                raise ValueError(
                    'BRDF may not be calculated correctly\n ZG at VZA = %5.1f deg.' % degrees(self.geovi.VZA))
        ZT = self.ZT
        if ZT < 0:
            self._ZT = 0
            self._PT = 1 - self.Pvg
            if xi > 0.0001:
                raise ValueError(
                    'BRDF may not be calculated correctly\n PT at VZA = %5.1f deg.' % degrees(self.geovi.VZA))
        BACK_FILE = "../Mod_5scale/soil_reflectance.txt"
        BACKGROUND_REF = np.array(pd.read_csv(BACK_FILE, delim_whitespace=True).values.tolist())
        value = self._multiple_scattering(BACKGROUND_REF)
        return value

    def _q(self, return_type: str):
        xi = self.geovi.xi
        cp = self.domain.tree.leaf.cp
        cs = self.Cs
        cv = self.Cv
        lo_90 = self.Lo_90
        gv = self.Gv
        LAI = self.domain.tree.LAI
        b = self.domain.area
        d = self.domain.n_tree
        vg_0 = self.AOP_tree.Vg_0
        SZA = self.geovi.SZA
        VZA = self.geovi.VZA
        ptreev = self.Ptreev
        pgapv = self.PgapV
        gamma_e = self.domain.tree.gamma_E
        __NN = self.domain.MAX_TREE_PER_QUADRAT
        qq1, qq2 = 0, 0
        DEL = 1. - xi * cp / pi
        if DEL < 0: DEL = 0
        if DEL > 1: DEL = 1
        del_broadleaf = xi * cp / pi - cp * 60 / 180
        if del_broadleaf < 0: del_broadleaf = 0
        if del_broadleaf > 1: del_broadleaf = 1
        q1 = self.q1
        if sqrt((cs - cv) * (cs - cv)) < 0.0000001:
            cs = cs - 0.0000001
            self._Cs = cs
        q2 = self.q2
        a = gv * LAI * b / (d * vg_0 * cos(VZA) * cos(VZA)) / cos(SZA)
        for i in range(1, __NN):
            pav_i = 0
            if exp(-(i - 1) * a) < 0.000000000000000000001:
                break
            for j in range(i, __NN):
                pav_i = pav_i + ptreev[j]
                if ptreev[j] < 0.00000000000000000001 and j > __NN / 2.:
                    break
            qq1 = qq1 + 1. * q1 * pav_i * pow(pgapv, i - 1.) * exp(-(i - 1) * a)
            qq2 = qq2 + 1. * q2 * pav_i * pow(pgapv, i - 1.) * exp(-(i - 1) * a)
        if qq1 > 1.:
            raise ValueError('qq1 is larger than 1. Impossible!')
        if qq2 > 1.:
            raise ValueError('qq2 is larger than 1. Impossible!')
        if gamma_e > 1:
            qq1b = qq1 * (1 - DEL)
            qq2b = qq2 * (1 - DEL)
        else:
            qq1b = qq1 * del_broadleaf
            qq2b = qq2 * del_broadleaf
        qq1 = qq1 * DEL
        qq2 = qq2 * DEL
        if 'QQ1' == return_type.upper():
            return qq1
        elif 'QQ2' == return_type.upper():
            return qq2
        elif 'QQ1B' == return_type.upper():
            return qq1b
        elif 'QQ2B' == return_type.upper():
            return qq2b
        else:
            raise ValueError("bad request for returns, it must be QQ1, QQ2, QQ1B or QQ2B (case insensitive)")

    def _ls(self):
        '''compute cv, cs,lo_90, mu'''
        LAI = self.domain.tree.LAI
        b = self.domain.area
        d = self.domain.n_tree
        v = self.domain.tree.V
        gs = self.Gs
        gv = self.Gv
        ss = self.AOP_tree.Ss
        sv = self.AOP_tree.Sv
        r = self.domain.tree.R
        shape = self.domain.tree.shape
        mu = self.mu
        h = 1 / mu
        self._AOP_tree._H = h
        if shape == 'CONE_CYLINDER':
            lo_90 = LAI * b / (v * d) * pi * r / 2.
        else:
            lo_90 = LAI * b / (v * d)
        cs = gs * ss * mu / lo_90
        cv = gv * sv * mu / lo_90
        return lo_90, cs, cv

    def _ptg_sub(self, option: str, max_integral: float, increment: float):
        in1, in2, flag_in1, flag_in2 = 0., 0., 0., 0.
        i = 0
        f_thelta, lt, cold, hot, w, ptg, f, H, XI, lambda_m = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        d = self.domain.n_tree
        b = self.domain.area
        r = self.domain.tree.R
        fd = self.Fd
        if 'CANOPY' == option.upper():
            w = self.domain.tree.leaf.ws
            lt = self.Ls
            cold = self.PT_Cold
            hot = 1 - self.Pig
            XI = self.geovi.xi
            H = self.H
            lambda_m = H * tan(XI)
        elif 'GROUND' == option.upper():
            w = self.Wt
            lt = self.Lt
            cold = self.PS
            hot = self.Pig
            XI = self.geovi.xi
            H = self.AOP_tree.H
            lambda_m = self.Lambda_m
        elif 'NADIR' == option.upper():
            w = sqrt(pi * r ** 2)
            lt = self.OmegaT * pi * (r ** 2) * d / b
            cold = self.Pvg * (1 - self.Pig)
            hot = self.Viewed_shadow
            medium = hot - cold
            if medium < 0: hot = cold
            XI = self.geovi.VZA
            H = self.domain.tree.Hc / 3. + self.domain.tree.Hb + self.domain.tree.Ha
            lambda_m = H * tan(XI)
        else:
            raise ValueError('Bad option, it must be GROUND, NADIR or CANOPY (case insensitive)')

        if XI < pi / 2.:
            i = np.linspace(0, max_integral, max_integral + 1)
            i_tmp = lambda_m + increment * i
            i_tmp1 = i_tmp[np.where(i_tmp != 0)]
            in1_e = np.exp(-lt * (1 + i_tmp1 / w)) / np.arctan(i_tmp1 / H)
            in1_e = in1_e[np.where(in1_e >= 0.00000000001)]
            in1 = np.sum(in1_e)
            in2_e = np.exp(-lt * (1 + i_tmp / w))
            in2_e = in2_e[np.where(in2_e >= 0.00000000001)]
            in2 = np.sum(in2_e)

            if in1 > 10000 or in1 < -10000:
                in1 = 0
                if XI > 0.00001:
                    print("possible problem with hotspot kernel at xi = {%5.3f}" % degrees(XI))

            if in2 > 10000 or in2 < -10000:
                in2 = 0
                if XI > 0.00001:
                    print("possible problem with hotspot kernel at xi = {%5.3f}" % degrees(XI))
            f_thelta = 0
            if in2 != 0: f_thelta = 1 - in1 / in2 * XI
            f = f_thelta
            ptg = (hot - cold) * f_thelta + cold
        else:
            ptg = cold
            f = 0

        if f_thelta < 0 or f_thelta > 1: ptg = cold

        if 'CANOPY' == option.upper():
            pt = ptg
            fs = f
            return pt, fs
        elif 'GROUND' == option.upper():
            pg = ptg
            ft = f
            return pg, ft
        else:
            zg = ptg
            if self.PS > self.Pvg - zg: self._PS = self.Pvg - zg
            if self.Pvg - zg < 0:
                self._PS = 0
                raise ValueError('Pvg is larger than ZG. Impossible!')
            fn = f * fd
            return zg, fn

    def _nadir(self, option: str):
        '''works like the hotspot function, but is applied to the shaded background viewed
        the original pig*pvg equation can induce and underestimation of the shaded background'''
        omega_t = self.OmegaT
        r = self.domain.tree.R
        d = self.domain.n_tree
        b = self.domain.area
        h = self.AOP_tree.H
        hc = self.domain.tree.Hc
        ha = self.domain.tree.Ha
        SZA = self.geovi.SZA
        e_r = self.E_r
        sg_0 = self.AOP_tree.Sg_0
        pig = self.Pig

        fd1, fd2, fd3 = 0, 0, 0
        i, lt, wt, DIST, total = 0., 0., 0., 0., 0.
        lt = omega_t * pi * (r ** 2) * d / b
        wt = pi * r * r

        i = 0
        while i < 100:
            DIST = i * exp(-lt * (1 + i / wt))
            if i > h / tan(SZA):
                fd2 = fd2 + exp(-lt * (1 + i / wt))
                total = total + exp(-lt * (1 + i / wt))
            else:
                fd2 = fd2 + DIST / (h / tan(SZA))
                total = total + exp(-lt * (1 + i / wt))
            i = i + 0.1
        fd2 = fd2 / total

        if ha / tan(SZA) > 2 * r:
            fd1 = 1
        else:
            fd1 = ha / (tan(SZA) * 2 * r)
        if ha / tan(SZA) > e_r: fd1 = 0
        fd3 = (pi * (r ** 2) * (fd1 - fd2) + sg_0 * fd2) / sg_0
        if fd3 > 1:
            raise ValueError('Fd3 is larger than 1, that\'s impossible')
        if fd3 < 0:
            raise ValueError('Fd3 is smaller than 0, that\'s impossible')
        viewed_shadow = (1 - pig) * fd3
        fd = fd3
        if 'VIEWED_SHADOW' == option.upper():
            return viewed_shadow
        elif 'FD' == option.upper():
            return fd
        else:
            raise ValueError('Wrong option, it must be VIEWED_SHADOW or FD (case insensitive)')

    def _distance(self):
        R = self.domain.tree.R
        D = self.domain.n_tree
        B = self.domain.area
        Lt = self.OmegaT * pi * R ** 2 * D / B
        Wt = sqrt(pi * R ** 2)
        return Wt / Lt

    def _resize_tree(self, option: str):  # adjust Vg and Sg accroding to tree size
        D = self.domain.n_tree
        n = self.domain.n_quadrat
        aop_tree = self.AOP_tree
        d = D / n
        vg = 0
        if 'VZA' == option.upper():
            vg_0 = aop_tree.Vg_0
        elif 'SZA' == option.upper():
            vg_0 = aop_tree.Sg_0
        else:
            raise ValueError('Wrong option, it must be VZA or SZA (case insensitive)')
        px = self.domain.Px  # get poisson or neyman distribution arrays
        for i in range(self.domain.MAX_TREE_PER_QUADRAT):
            if i > d:
                vg = vg + px[i] * vg_0 * d / i
            else:
                vg = vg + px[i] * vg_0
        return vg

    def _G(self, option: str):
        ge_choice = self.domain.tree.ge_choice
        if option.upper() not in ['SZA', 'VZA']:
            raise ValueError('Wrong option: it must be SZA or VZA')
        if 'NO_BRANCH' == ge_choice:
            gv, pgapv = self._p_gap_ax_b(option)
            return gv
        else:
            gv, pgapv = self._p_gap_branch(option)
            return gv

    def _Pgap(self, option):
        ge_choice = self.domain.tree.ge_choice
        if 'NO_BRANCH' == ge_choice:
            gv, pgapv = self._p_gap_ax_b(option)
            return pgapv
        else:
            gv, pgapv = self._p_gap_branch(option)
            return pgapv

    def _p_gap_ax_b(self, option: str):
        a = self.A
        c = self.C
        aop_tree = self.AOP_tree
        omega_e = self.domain.tree.Omega_E  # clumping index
        gamma_e = self.domain.tree.gamma_E  # needle to shoot ratio
        LAI = self.domain.tree.LAI  # leaf area index
        b = self.domain.area  # domain size
        d = self.domain.n_tree  # number of trees in the domain

        if 'VZA' == option.upper():
            za = self.geovi.VZA
            vg = aop_tree.Vg_0  # viewed ground
        elif 'SZA' == option.upper():
            za = self.geovi.SZA
            vg = aop_tree.Sg_0  # sunlit ground
        elif '0' == option.upper():
            za = 0
            r = self.domain.tree.R
            vg = pi * r ** 2
        elif 'LAI' == option.upper():
            za = cos(0.537 + 0.025 * LAI)
            vg = aop_tree.Vg_0_mean
        else:
            raise ValueError('Wrong option, it must be VZA, SZA, 0 or LAI (case insensitive)')

        gv = (a * za + c) * omega_e / gamma_e
        pgapv = exp(-gv * LAI * b / (d * vg * cos(za)))
        return gv, pgapv

    def _p_gap_branch(self, option):
        LAI = self.domain.tree.LAI
        aop_tree = self.AOP_tree
        sv = aop_tree.Sv
        ss = aop_tree.Ss
        vg_0_mean = aop_tree.Vg_0_mean
        sg_0 = aop_tree.Sg_0
        vg_0 = aop_tree.Vg_0
        b = self.domain.area
        v = self.domain.tree.V
        r = self.domain.tree.R
        gamma_e = self.domain.tree.gamma_E
        d = self.domain.n_tree
        hc = self.domain.tree.Hc
        hb = self.domain.tree.Hb
        alpha_l = self.domain.tree.alpha_l
        alpha_b = self.domain.tree.alpha_b
        ll = self.domain.tree.Ll
        rb = self.domain.tree.Rb
        ratio = self.domain.tree.leaf.RATIO
        # xb, xl, lb, za, gb, gl, pbj, pj = 0., 0., 0., 0., 0., 0., 0., 0.
        # cos_zab, p, mub, mub, beta, s, agb, agl = 0., 0., 0., 0., 0., 0., 0., 0.
        # j = 0
        lb = LAI / ll
        mub = lb * b / (v * d)
        mul = ll * b / (v * d)

        if 'VZA' == option.upper():
            za = self.geovi.VZA
            s = sv
        elif 'SZA' == option.upper():
            za = self.geovi.SZA
            s = ss
        elif '0' == option.upper():
            za = 0
            s = 1 / 3 * hc + hb
        elif 'LAI' == option.upper():
            za = acos(0.537 + 0.025 * LAI)
            s = v / (vg_0_mean * (0.537 + 0.025 * LAI))
        else:
            raise ValueError('Wrong option: it must be VZA,SZA,0 or LAI')

        if alpha_l > 0:
            if za <= pi / 2 - alpha_l:
                gl = cos(alpha_l) * cos(za)
            else:
                xl = 1 / tan(alpha_l) / tan(za)
                xl = acos(xl)
                gl = cos(alpha_l) * cos(za) * (1. + 2. * (tan(xl) - xl) / pi)
            agl = acos(gl)
            gl = gl + sin(agl) * ratio
        else:
            gl = 0.5
            agl = acos(gl)

        if alpha_b > 0:
            if za <= pi / 2 - alpha_b:
                gb = cos(alpha_b) * cos(za)
            else:
                xb = 1 / tan(alpha_b) / tan(za)
                xb = acos(xb)
                gb = cos(alpha_b) * cos(za) * (1. + 2. * (tan(xb) - xb) / pi)
            agb = acos(gb)
            gb = gb + sin(agb) * cos(alpha_b) * rb / r
        else:
            gb = 0.5

        pl1 = 0
        beta = 0
        while beta < pi:
            if alpha_b > 0:
                cos_zab = sin(za) * sin(alpha_b) * cos(beta) + cos(za) * cos(alpha_b)
                agb = acos(cos_zab)
                cos_zab = cos_zab + sin(agb) * cos(alpha_b) * rb * r
            else:
                cos_zab = 1
            if cos_zab < 0: cos_zab = -cos_zab
            agl = gl * ll / (gamma_e * cos_zab)
            pl1 = pl1 + 1 / pi * exp(-agl) * pi / 100.
            beta = beta + pi / 100

        pbj = exp(-gb * mub * s)
        p = pbj

        j = 1
        while j <= s * 5:
            pbj = pbj * gb * mub * s / j
            pj = pbj * pow(pl1, j)
            if pj <= 1.: p = p + pj
            j = j + 1

        if 'VZA' == option.upper():
            pgapv = p
            gv = cos(za) * vg_0 * d * log(1 / p) / (b * LAI)
            return gv, pgapv
        elif 'SZA' == option.upper():
            pgaps = p
            gs = cos(za) * sg_0 * d * log(1 / p) / (b * LAI)
            return gs, pgaps
        elif '0' == option.upper():
            gv = 0
            pgap0 = p
            return gv, pgap0
        else:
            gv = 0
            pgapv_mean = p
            return gv, pgapv_mean

    def _fo(self, option: str, return_type: str):
        r = self.domain.tree.R
        fo = 0
        svg0 = r * r * pi
        fr = self.domain.Fr
        # psvg0 = psg_hot0
        psvg0 = self.PSG_HOT0
        pvg_nadir = self.Pvg_nadir
        if 'VZA' == option:
            svg = self.AOP_tree.Vg_0
            theta = self.geovi.VZA
        elif 'SZA' == option:
            svg = self.AOP_tree.Sg_0
            theta = self.geovi.SZA
        else:
            raise ValueError('bad option in _fo function, it must be VZA or SZA')

        if svg0 != 0:
            fo = fr * exp(-(svg - svg0) / svg0 * 2 * theta / pi)
        if fo < 0:
            raise ValueError('possible problem with _fo function')
        if 'VZA' == option:
            if 'Pvg' == return_type:
                PVG = self.PVG
                Pvg = PVG + (psvg0 - pvg_nadir) * fo
                return Pvg
            elif 'Pv' == return_type:
                PV = self.PV
                Pv = PV * (1 - fo)
                return Pv
            else:
                raise ValueError('Wrong request for return types under current option VZA, it must be Pvg or Pv')
        elif 'SZA' == option:
            if 'Pig' == return_type:
                PIG = self.PIG
                Pig = PIG + (psvg0 - pvg_nadir) * fo
                return Pig
            elif 'Ps' == return_type:
                PS = self.PS
                Ps = PS * (1 - fo)
                return Ps
            else:
                raise ValueError('Wrong request for return types under current option SZA, it must be Pig or Ps')
        else:
            raise ValueError('bad option in _fo function, it must be VZA or SZA')

    def _overlap_v1(self, option: str, return_type: Optional[str] = None):
        '''
        subroutine to calculate gap fraction (Pvg & Pig), Pv|Ps, Pvc|Psc
        @param option: option to determine what kind of returns
        @param return_type: request for return type, default: None
        @param restrain: restraints when option:SZA, default: None, if it equals to 'POISSON', Px will be replaced By px_poisson
        @return: psv: overlap probability
        @return: psvc: cone overlap probability
        '''
        b = self.domain.area  # domain area
        d = self.domain.n_tree  # total number of trees in the domain
        n = self.domain.n_quadrat  # number of quadrats in the domain
        r = self.domain.tree.R  # crown radius
        NN = self.domain.MAX_TREE_PER_QUADRAT  # maximum amount of trees that is allowed in a quadrat
        A = b / n  # quadrat size
        i_0 = d / n  # number of trees in a qudrat
        # px = self.domain.Px  # an array of poisson or neyman distribution probabilities
        if 'SZA' == option and 'Pig_poisson' == return_type:
            px = self.domain.Px_poisson  # an array of poisson or neyman distribution probabilities
        else:
            px = self.domain.Px  # an array of poisson or neyman distribution probabilities

        if 'VZA' == option.upper():
            pgap = self.PgapV  # gap fraction in one crown, viewed from viewer
            a = self.AOP_tree.Vg_0  # tree projection on the ground, viewed from viewer
            ac = self.AOP_tree.Vgc  # tree cone projection on the ground, viewed from viewer
        elif 'SZA' == option.upper():
            pgap = self.PgapS  # gap fraction in one crown, viewed from sun
            a = self.AOP_tree.Sg_0  # tree projection on the ground, viewed from sun
            ac = self.AOP_tree.Sgc  # tree cone projection on the ground, viewed from sun
        elif 'LAI' == option.upper():
            pgap = self.PgapV_mean  # mean gap fraction in one crown, evaluated by LAI
            a = self.AOP_tree.Vg_0_mean  # tree projection on the ground, evaluated by LAI
            ac = 0
        else:
            pgap = self.Pgap0  # gap fraction from nadir view
            a = pi * (r ** 2)  # tree projection on the ground ,from nadir view
            ac = 0

        p = 0
        pc = 0
        psv = 0
        psvc = 0
        ptree = [0] * NN
        ptreec = [0] * NN
        ptj = [0] * NN
        ptjc = [0] * NN
        ptj1 = 0
        ptjc1 = 0

        for j in range(0, NN):
            pt = 0
            ptc = 0
            for i in range(j, NN):
                if i >= 1 and ptj[i - 1] < 0.0000001 and i > i_0:
                    ptj[i] = 0
                    ptjc[i] = 0
                else:
                    aa = a
                    aac = ac
                    if i > i_0:
                        aa = a * i_0 / i
                    if a > A:
                        aa = A
                    if ac > A:
                        aac = A
                    ptj[i] = px[i] * comb(i + j - 1, i - 1) * pow(1 - aa / A, i) * pow(aa / A, j)
                    ptjc[i] = px[i] * comb(i + j - 1, i - 1) * pow(1 - aac / A, i) * pow(aac / A, j)

                ptj[0] = px[0]
                ptree[0] = ptj[0]
                pt = ptj[i] + pt
                ptjc[0] = px[0]
                ptreec[0] = ptjc[0]
                ptc = ptjc[i] + ptc

            if j == 1:
                ptj1 = pt
                ptjc1 = ptc

            p = p + pt * pow(pgap, j * 1.)
            pc = pc + ptc * pow(pgap, j * 1.)
            ptree[j] = pt
            ptreec[j] = ptc
            # if 'VZA' == option:
            #    ptreev[j] = ptree[j]
            # if 'SZA' == option:
            #    ptrees[j] = ptree[j]
        psv = 1 - p - ptj1 * (1 - pgap)
        psvc = 1 - pc - ptjc1 * (1 - pgap)
        psv = (psv * a - psvc * ac) / (a - ac)

        if p <= 0 or p > 1:
            p = 0
            psv = 1
        if pc <= 0 or pc > 1:
            pc = 0
            psvc = 1
        if a / A >= 1:
            psv = 1
            p = 1
        if ac / A >= 1:
            psvc = 1
            pc = 1

        if psv >= 1: psv = 1
        if p < 0.00001: psv = 1
        if psvc >= 1: psvc = 1
        if pc < 0.00001: psvc = 1
        if ac <= 0: psvc = 0

        if 'VZA' == option.upper():
            if return_type is None:
                raise ValueError('return type cannot be None under current option VZA')
            elif 'Pvg' == return_type:
                return p
            elif 'Pv' == return_type:
                return psv
            elif 'Pvc' == return_type:
                return psvc
            elif 'Ptreev' == return_type:
                return ptree
            else:
                raise ValueError('Wrong request of return type under current option VZA: it must be Pvg, Pv or Pvc')
        elif 'SZA' == option.upper():
            if return_type is None:
                raise ValueError('return type cannot be None under current option VZA')
            elif 'Pig' == return_type or 'Pig_poisson' == return_type:
                return p
            elif 'Ps' == return_type:
                return psv
            elif 'Psc' == return_type:
                return psvc
            elif 'Ptrees' == return_type:
                return ptree
            else:
                raise ValueError(
                    'Wrong request of return type under current option SZA: it must be Pig, Pig_poisson, Ps or Psc')
        elif 'LAI' == option.upper():
            if return_type is None or 'Pvg_mean' == return_type:
                return p
            else:
                raise ValueError('Wrong request of return type under current option LAI: it must be None or Pvg_mean')
        elif 'NADIR' == option.upper():
            if return_type is None or 'Pvg_nadir' == return_type:
                return p
            else:
                raise ValueError(
                    'Wrong request of return type under current option NADIR: it must be None or Pvg_nadir')
        else:
            raise ValueError('Wrong option! it must be one of the followings: VZA, SZA, LAI, NADIR')

    def _multiple_scattering(self, background_ref: np.array):
        '''this function computes the amount of eletromagnetic radiation reaching the four
        components(sunlit,shaded background and foliage) due to mutliple scattering.
        It serves as the basis for the hyperspectral mode of 5-scale
        Reference:Chen, J. M., & Leblanc, S. G. (2001). Multiple-scattering scheme useful
        for geometric optical modeling. IEEE Transactions on Geoscience and Remote Sensing
        , 39(5), 1061-1071.  '''
        qq1b = self.QQ1B
        qq2b = self.QQ2B
        pti = self.Pti
        pg = self.PG
        pt = self.PT
        zg = self.ZG
        zt = self.ZT
        ft = self.Ft
        xi = self.geovi.xi
        pig = self.Pig
        delta_LAI = self.domain.tree.DeltaLAI
        ha = self.domain.tree.Ha
        hb = self.domain.tree.Hb
        hc = self.domain.tree.Hc
        cp = self.domain.tree.leaf.cp
        gv = self.Gv
        sv = self.AOP_tree.Sv
        mu = self.mu
        cs = self.Cs
        gamma_e = self.domain.tree.gamma_E
        shape = self.domain.tree.shape
        lo_90 = self.Lo_90
        r = self.domain.tree.R
        b = self.domain.area
        d = self.domain.n_tree
        e_r = self.E_r
        SZA = self.geovi.SZA
        VZA = self.geovi.VZA
        LAI = self.domain.tree.LAI
        foliage_ref = np.array(self.domain.tree.leaf.DHR).flatten()
        foliage_trans = np.array(self.domain.tree.leaf.DHT).flatten()
        wave = np.array(self.domain.tree.leaf.wv).flatten()
        pvg_mean = self.Pvg_mean
        cv_mean, f_s_trest, f_s_t, f_T_T, F_G_T, F_g_T, f_zg_t, f_t_t, f_zt_t, f_t_zt = \
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        f_s_g, f_t_g, f_tt2, f_tt3, f_tt_zt, f_tt_g, hi, hj, li, lj = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        num, omega_total, pgapv_mean, p_tot, q1_mean, q2_mean, q1b_mean, q2b_mean, rr, thelta = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        intensity = atan((hb + hc) / (2 * r))
        intensity = intensity - SZA
        if intensity < 0: intensity = -intensity
        intensity = cos(intensity)

        # should intersect wave with the wavelengths of background_foliage
        wv_back = background_ref[:, 0]
        wv_back = wv_back.flatten()
        background_ref = background_ref[:, 1]
        background_ref = background_ref.flatten()
        wv_in1 = np.in1d(wave, wv_back)  # indices of intersected wavelengths
        wv_in2 = np.in1d(wv_back, wave)  # indices of intersected wavelengths
        wave = wave[wv_in1]
        foliage_ref = foliage_ref[wv_in1]
        foliage_trans = foliage_trans[wv_in1]
        background_ref = background_ref[wv_in2]
        wave_index_min = 0
        wave_index_max = len(wave)
        lambda0 = wave[wave_index_min:wave_index_max] / 1000
        rg = background_ref[wave_index_min:wave_index_max]
        rt = foliage_ref[wave_index_min:wave_index_max]
        tt = foliage_trans[wave_index_min:wave_index_max]
        tau_r = 0.008569 * pow(lambda0, -4) * (1 - 0.0113 * pow(lambda0, -2) + 0.00013 * pow(lambda0, -4))
        rr = (3 * tau_r + (2 - 3 * cos(SZA)) * (1 - np.exp(-tau_r / cos(SZA)))) / (4 + 3 * tau_r)
        fd = ((1 - rr) - np.exp(-tau_r / cos(SZA))) / (1 - rr)
        if LAI > 0:
            num = log(pvg_mean)
            dem = (-0.5 * LAI / (0.537 + 0.025 * LAI))
            omega_total = num / dem
            if omega_total > 1:
                raise ValueError('OMEGA_TOTAL is larger than 1, this is ridiculous')
            cv = 0.5 * omega_total / cos(VZA)
            thelta_half_mean = atan((ha + 0.5 * (hb + hc / 3.)) / (0.5 * e_r))
            q1_mean = self._q1(pi / 2 + thelta_half_mean, 0)
            del1 = 1 - cp * GeoVI.XI(SZA, pi / 2 + thelta_half_mean, 0) / pi
            q1b_mean = q1_mean * (1 - del1) / del1
            if q1b_mean > 1: q1b_mean = 1
            if q1b_mean < 0: q1b_mean = 0
            cv_mean = gv * sv * mu / lo_90
            tg = tan(pi / 2 + thelta_half_mean)
            if tg < 0: tg = -tg
            vg_mean = 2 * tg * r * (hb + hc / 3) + pi * (r ** 2)
            if vg_mean < 0: vg_mean = 0
            pgapv_mean = exp(-cv_mean * LAI * b / (d * vg_mean * cos(thelta_half_mean)))
            del1 = 1 - cp * (cos(SZA) * cos(thelta_half_mean) + sin(thelta_half_mean) * sin(SZA)) / pi
            if del1 < 0: del1 = -del1
            q2_mean = cs * cv_mean / (cv_mean - cs) * (exp(-cs * lo_90) - exp(-cv_mean * lo_90)) / (1 - pgapv_mean)
            q2b_mean = (1 - del1) * q2_mean
            q2_mean = q2_mean * del1
            if q2b_mean > 1: q2b_mean = 1
            if q2b_mean < 0: q2b_mean = 0
            if 'SPHEROID' != shape:
                tt = np.power(tt, gamma_e)

            F_G_T = 0
            li = delta_LAI
            while li <= LAI:
                # for li in range(delta_LAI, LAI, delta_LAI):
                arg4 = 0.5 * omega_total * li / cos(asin(0.537 + 0.025 * li))
                f_s_t = f_s_t + 0.5 * exp(-arg4) * exp(-cv * li) * delta_LAI
                f_s_trest = f_s_trest + 0.5 * exp(-arg4) * delta_LAI
                arg5 = 0.5 * omega_total * (LAI - li) / cos(asin(0.537 + 0.025 * (LAI - li)))
                F_G_T = F_G_T + 0.5 * exp(-arg5) * delta_LAI
                p_tot = p_tot + exp(-cv * li) * delta_LAI
                f_tt2 = 0
                f_tt3 = 0
                hi = (hb + hc / 3) - li * (hb + hc / 3) / LAI
                inc_thelta = delta_LAI
                thelta_min = (hi - inc_thelta) / e_r
                thelta_min = 0.5 * (pi / 2 + atan(thelta_min))
                thelta_max = ((hb + hc / 3) - hi) / e_r
                thelta_max = 0.5 * (pi / 2 + atan(thelta_max))

                if thelta_min > thelta_max:
                    thelta_min, thelta_max = thelta_max, thelta_min
                    # thelta_tmp = thelta_min
                    # thelta_min = thelta_max
                    # thelta_max = thelta_tmp

                thelta = thelta_min
                while thelta <= thelta_max:
                    # for thelta in range(thelta_min, thelta_max, inc_thelta):
                    thelta_h = pi - 2 * thelta
                    f_tt2 = f_tt2 + self._q1(thelta_h, 0) * cos(thelta) * sin(thelta) * inc_thelta * 10 * pi / 180
                    del1 = 0
                    del2 = 0
                    for p in range(0, 180, 10):
                        del1 = del1 + 1 - cp * (
                                cos(SZA) * cos(thelta_h) + sin(thelta_h) * sin(SZA) * cos(p * pi / 180)) / pi
                        del2 = del2 + cp * (
                                cos(SZA) * cos(thelta_h) + sin(thelta_h) * sin(SZA) * cos(p * pi / 180)) / pi
                    f_tt2 = f_tt2 * del1 * 10 * pi / 180
                    f_tt3 = f_tt2 * del2 * 10 * pi / 180
                    thelta = thelta + inc_thelta

                f_t_zt = f_t_zt + f_tt2 * delta_LAI
                f_tt_zt = f_tt_zt + f_tt3 * delta_LAI
                li = li + delta_LAI

            f_t_zt = f_t_zt / LAI
            f_tt_zt = f_tt_zt / LAI
            f_zt_t = f_t_zt
            f_s_t = f_s_t / p_tot
            f_s_trest = f_s_trest / LAI

            F_G_T = F_G_T / LAI
            F_g_T = F_G_T * pig
            f_zg_t = F_G_T * (1 - pig)

            arg3 = 0.5 * omega_total * LAI / (0.537 + 0.025 * LAI)  # sky view factor from ground
            f_s_g = exp(-arg3)

            f_t_g = 0.5 * (q1_mean + q2_mean) * (1 - f_s_g)  # sunlit trees view factor from ground
            f_tt_g = 0.5 * (q1b_mean + q2b_mean) * (1 - f_s_g)

            f_t_t = 1 - f_s_trest - F_G_T - f_zt_t
            f_T_T = 1 - F_G_T - f_s_trest

            if f_t_t < 0:
                raise ValueError('F_t_t is smaller than 0')
            if f_s_t < 0:
                raise ValueError('F_s_t is smaller than 0')
            if f_zt_t < 0:
                f_zt_t = 0
                raise ValueError('F_zt_t is smaller than 0')
            if f_t_g < 0:
                f_t_g = 0
                raise ValueError('F_t_G is smaller than 0')
            if F_G_T < 0:
                F_G_T = 0
                raise ValueError('F_G_T is smaller than 0')
            # calculate shaded reflectivities
            rt_2nd = rt * ((rt + tt * ft) * f_t_t + rg * F_g_T + fd * f_s_t)
            #temp = rt[0] * (rt[0] * f_t_zt+ tt[0] * f_tt_zt + rg[0] * F_g_T + fd[0] * f_s_t)
            rtz_2nd = rt * (rt * f_t_zt + tt * f_tt_zt + rg * F_g_T + fd * f_s_t)
            rg_2nd = rg * (rt * f_t_g + tt * f_tt_g + fd * f_s_g)
            rgt_2nd = (rg_2nd * F_G_T + rtz_2nd * (1 - F_G_T - f_t_t - f_s_trest) + rt_2nd * (f_t_t + f_zt_t)) / (
                    1. - f_s_trest)
            rgt_3rd = 0.5 * (rt + tt) * (1 - f_s_trest) * rgt_2nd
            rt_ms = rt_2nd + rgt_3rd / (1 - (rt + tt) / 2 * (1 - f_s_trest))
            rzt_ms = rtz_2nd + rgt_3rd / (1 - (rt + tt) / 2 * (1 - f_s_trest))
            rg_ms = rg_2nd + rg * rgt_2nd * (1 - f_s_g) / (1 - (rt + tt) / 2 * (1 - f_s_trest))
            tz = rzt_ms
            gz = rg_ms

            r0 = pg * (rg * (1 - xi * cp / pi) + rg_ms) + pt * (rt * intensity + rt_ms) + zg * gz + zt * tz + tt * (
                    qq2b * (1 - pti) + qq1b * pti)
        else:
            r0 = (1 + fd) * rg
        return r0

    def _q1(self, thelta, phi):
        '''calcualte appr. of Q1tot for MS Scheme'''
        SZA = self.geovi.SZA
        cp = self.domain.tree.leaf.cp
        gv = self.Gv
        cs = self.Cs
        hb = self.domain.tree.Hb
        hc = self.domain.tree.Hc
        r = self.domain.tree.R
        LAI = self.domain.tree.LAI
        b = self.domain.area
        d = self.domain.n_tree
        lo_90 = self.Lo_90

        if thelta < 0: thelta = -thelta
        diff = cos(SZA) * cos(thelta) + sin(thelta) * cos(phi) * sin(SZA)
        if diff < 0: diff = -diff
        del0 = 1 - diff * cp / pi
        if del0 < 0: del0 = -del0
        if del0 > 1: del0 = 1
        if thelta > pi / 2: thelta = pi - thelta
        if thelta == 0: thelta = 0.0000000001
        if sqrt((thelta - pi / 2) * (thelta - pi / 2)) < 0: thelta = pi / 2 - 0.0000001
        cv = gv / sin(thelta)
        vg_thelta = 2 * tan(thelta) * r * (hb + hc) + pi * (r ** 2)
        pgapv_thelta = exp(-cv * LAI * b / (d * vg_thelta * cos(thelta)))
        q1_thelta = (1. - exp(-(cs * lo_90 + cv * lo_90))) * cs * cv / (cs + cv) / (1 - pgapv_thelta)
        q1_thelta = q1_thelta * del0
        if q1_thelta > 1:
            raise ValueError('Q1_thelta is larger than 1, impossible!')
        elif q1_thelta < 0:
            raise ValueError('Q1_thelta is smaller than 0, impossible!')
        else:
            return q1_thelta


if __name__ == '__main__':
    leaf = Needle(diameter=40, thickness=1.6, xu=0.045, baseline=0.0005, albino=2, Cab=200, Cl=40, Cp=1, Cw=100)
    tree = ConeTree(leaf=leaf, R=1, alpha=13, Ha=1, Hb=5, LAI=3.5, Omega_E=0.8, gamma_E=1, ge_choice='BRANCH',
                    alpha_b=25, alpha_l=-1)
    domain = Domain(tree=tree, area=10000, n_tree=6000, n_quadrat=40, Fr=0.0, m2=0)
    geovi = GeoVI(SZA=20, VZA=0, phi=0)
    aop_domain = AOPDomain(geovi=geovi, domain=domain)
    # aop_domain.compare()
    ro = aop_domain.ro()
    print(ro)
