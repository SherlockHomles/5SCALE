import os
import time
from typing import Optional

import pandas as pd
from Domain_v1 import Domain
from AOPTree_v1 import GeoVI, AOPConeTree, AOPSpheroidTree
from Tree_v1 import ConeTree, SpheroidTree
from Leaf_v1 import Needle, Broadleaf
from math import exp, cos, sin, pi, tan, acos, log, sqrt, atan, degrees, asin
from scipy.special import comb
import numpy as np
import matplotlib.pyplot as plt


class AOPDomain(object):
    '''
    the class calculates apparent optical properties(AOP, i.e. view and sun geometries dependent) of a domain
    Attributes:
        geovi: an object contains information about view and illumination geometries ( perhaps a better way is to define
                a class for viewer and sun separately)
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
        lambda_m: Eq.52
        PSG0_VIEW: mean gap fraction in one crown viewed from viewer
        PSG0_SUN: mean gap fraction in one crown viewed from sun
        PVG:overlapping prob of Vg,i.e. Eq.26, this param does not consider vertical overlap
        Pvg:overlapping prob of Vg,i.e. Eq.26, this param considers vertical overlap
        Pvc: overlapping prob of Vgc,i.e. Eq.27
        PV: overlapping prob of Vgb,i.e. Eq.29, this param does not consider vertical overlapping
        Pv: overlapping prob of Vgb,i.e. Eq.29, this param considers vertical overlapping
        PIG: overlapping prob of Sg, this param does not consider vertical overlapping
        Pig: overlapping prob of Sg, this param considers vertical overlapping

    Methods:
        _resize_tree: adjust Vg and Sg with tree size
        _Pgap: calculate gap fraction according to option
        _G: calculate either Gv or Gs, depending on option
        _p_gap_ax_b: calculate gap fraction when tree has no branches

    '''

    def __init__(self, domain: Domain, geovi: GeoVI) -> None:
        self.__geovi = geovi
        self.__domain = domain
        tree = self.__domain.tree
        geovi = self.__geovi
        self.__A = 0.0
        self.__C = 0.5
        if type(tree) == ConeTree:
            self.__AOP_tree = AOPConeTree(tree, geovi)
        elif type(tree) == SpheroidTree:
            self.__AOP_tree = AOPSpheroidTree(tree, geovi)
        else:
            raise ValueError('tree type error: it must be ConeTree or SpheroidTree')
        self.__wave, self.__Rc = self._ro()

    @property
    def geovi(self):
        return self.__geovi

    @property
    def domain(self):
        return self.__domain

    @property
    def wave(self):
        return self.__wave

    @property
    def Rc(self):
        return self.__Rc

    @property
    def ZG(self):
        return self.__ZG

    @property
    def PG(self):
        return self.__PG

    @property
    def ZT(self):
        return self.__ZT

    @property
    def PT(self):
        return self.__PT

    @property
    def q1(self):
        cs = self.__Cs
        cv = self.__Cv
        lo_90 = self.__Lo_90
        value = (1. - exp(-(cs * lo_90 + cv * lo_90))) * cs * cv / (cs + cv)  # Eq.56
        if value > 1:
            raise ValueError('q1 is larger than 1. Impossible!')
        return value

    @property
    def q2(self):
        cs = self.__Cs
        cv = self.__Cv
        lo_90 = self.__Lo_90
        value = (exp(-cs * lo_90) - exp(-cv * lo_90)) * cs * cv / (cv - cs)  # Eq.61
        if value > 1:
            raise ValueError('q2 is larger than 1. Impossible!')
        return value

    def _ro(self) -> np.array:  # simulate canopy reflectance
        self.__Vg = self._resize_tree('VZA')
        ge_choice = self.__domain.tree.ge_choice
        R = self.__domain.tree.R
        D = self.__domain.n_tree  # number of trees in the domain
        B = self.__domain.area  # domain size
        LAI = self.__domain.tree.LAI
        if 'NO_BRANCH' == ge_choice:
            self.__Gv, self.__PgapV, self.__GFoliage = self._p_gap_ax_b('VZA')
            self.__PSG0_VIEW = 1 - self.__Vg * D / B * (1 - self.__PgapV)
            self.__PgapV_mean = self._p_gap_ax_b('LAI')
            self.__Gs, self.__PgapS, self.__Lo = self._p_gap_ax_b('SZA')
            self.__Sg = self._resize_tree('SZA')
            self.__PSG0_SUN = 1 - self.__Sg * D / B * (1 - self.__PgapS)
            self.__Pgap0 = self._p_gap_ax_b('0')
        else:
            self.__Gv, self.__PgapV, self.__GFoliage = self._p_gap_branch('VZA')
            self.__PSG0_VIEW = 1 - self.__Vg * D / B * (1 - self.__PgapV)
            self.__PgapV_mean = self._p_gap_branch('LAI')
            self.__Gs, self.__PgapS, self.__Lo = self._p_gap_branch('SZA')
            self.__Sg = self._resize_tree('SZA')
            self.__PSG0_SUN = 1 - self.__Sg * D / B * (1 - self.__PgapS)
            self.__Pgap0 = self._p_gap_branch('0')
        self.__PSG_HOT0 = 1 - pi * R ** 2 * D / B * (1 - self.__Pgap0)
        if self.__PSG0_SUN < 0:
            self.__PSG0_SUN = 0
        if self.__PSG0_VIEW < 0:
            self.__PSG0_VIEW = 0
        self.__Pig_poisson, self.__Pv, self.__Pvc, self.__Ptreev = self._overlap_v1('SZA', dist='POISSON')
        if self.__Pig_poisson > 1 or self.__Pig_poisson < 0:
            self.__Pig_poisson = 0
        self.__Pvg_mean = self._overlap_v1('LAI')
        self.__Pig, self.__Ps, self.__Psc, self.__Ptrees = self._overlap_v1('SZA')
        self.__Pvg_nadir = self._overlap_v1('NADIR')
        self.__PIG, self.__Pig, self.__Ps = self._fo('SZA')
        self.__OmegaT = log(self.__Pig) / log(self.__Pig_poisson)
        self.__Pvg = 0
        self.__Pvg, self.__Pv, self.__Pvc, self.__Ptreev = self._overlap_v1('VZA')
        self.__PVG, self.__Pvg, self.__Pv = self._fo('VZA')
        if self.__Pvg < 0:
            raise ValueError('Pvg should larger than 0')
        self.__E_r = self._distance()
        self.__PS, self.__Wt, self.__H, self.__Lt, self.__Lambda_m = self._ps()
        self.__Pti = self._pq_sub()
        if self.__Pti > 1:
            self.__Pti = 1
        if self.__Pti < 0:
            raise ValueError('There must be something wrong with Pti, it is smaller than 0')
        self.__Viewed_shadow, self.__Fd = self._nadir()
        self.__ZG, self.__Fn = self._ptg_sub('NADIR', 120000, 0.1)
        self.__PG, self.__Ft = self._ptg_sub('GROUND', 20000, 0.01)
        num = log(self.__Pvg_mean)
        dem = -0.5 * LAI / (0.537 + 0.025 * LAI)
        self.__OmegaTotal = num / dem
        self.__Lo_90, self.__Cs, self.__Cv = self._ls()
        self.__QQ1, self.__QQ2, self.__QQ1B, self.__QQ2B = self._q()
        self.__PT_Cold = self.__QQ1 * self.__Pti + (1 - self.__Pti) * self.__QQ2
        self.__PT, self.__Fs = self._ptg_sub('CANOPY', 40000, 0.001)

        self.__ZG = self.__Pvg - self.__PG  # equation below Eq.72
        if self.__PT < 0:
            self.__PT = 0
        if self.__ZG < 0:
            self.__ZG = 0
            self.__PG = self.__Pvg
        self.__ZT = (1 - self.__Pvg) - self.__PT
        if self.__ZT < 0:
            self.__ZT = 0
            self.__PT = 1 - self.__Pvg
        #    if xi > 0.0001:
        #        raise ValueError(
        #            'BRDF may not be calculated correctly\n PT at VZA = %5.1f deg.' % degrees(self.__geovi.VZA))
        tmp_path = os.path.dirname(os.path.abspath(__file__))
        BACK_FILE = os.path.join(tmp_path, 'soil_reflectance.txt')
        BACKGROUND_REF = np.array(pd.read_csv(BACK_FILE, delim_whitespace=True).values.tolist())
        wave, value = self._multiple_scattering(BACKGROUND_REF)
        return wave, value

    def _ps(self):
        SZA = self.__geovi.SZA
        sg_0 = self.__AOP_tree.Sg_0
        Ha = self.__AOP_tree.tree.Ha
        Hb = self.__AOP_tree.tree.Hb
        Hc = self.__AOP_tree.tree.Hc
        D = self.__domain.n_tree  # number of trees in the domain
        B = self.__domain.area  # domain size
        xi = self.__geovi.xi
        PS = self.__Pig * self.__Pvg
        if sg_0 < 0:
            Wt = 0
        else:
            Wt = sqrt(sg_0)

        if SZA == pi:
            H = 0
        else:
            H = 1 / cos(SZA) * (Hc / 3 + Hb + Ha)

        Lt = 1 * D * sg_0 / B * self.__OmegaT
        if xi < pi:
            lambda_m = H * tan(xi)
        else:
            lambda_m = 0

        return PS, Wt, H, Lt, lambda_m

    def _pq_sub(self):
        '''
        prob of seeing illuminated tree crowns
        @return: Eq.39
        '''
        r = self.__domain.tree.R
        e_r = self.__E_r
        phi = self.__geovi.phi
        pv = self.__Pv
        pvc = self.__Pvc
        ps = self.__Ps
        psc = self.__Psc
        tic = self.__AOP_tree.tic
        tib = self.__AOP_tree.tib
        tab = self.__AOP_tree.tab
        tac = self.__AOP_tree.tac
        shape = self.__domain.tree.shape

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
            self.__Psc = 0
            self.__Pvc = 0
            pti = (tib + tic) / (tab + tac)
        return pti

    def _q(self):
        xi = self.__geovi.xi
        cp = self.__domain.tree.leaf.cp
        cs = self.__Cs
        cv = self.__Cv
        lo_90 = self.__Lo_90
        gv = self.__Gv
        LAI = self.__domain.tree.LAI
        b = self.__domain.area
        d = self.__domain.n_tree
        vg_0 = self.__AOP_tree.Vg_0
        SZA = self.__geovi.SZA
        VZA = self.__geovi.VZA
        ptreev = self.__Ptreev
        pgapv = self.__PgapV
        gamma_e = self.__domain.tree.gamma_E
        __NN = self.__domain.MAX_TREE_PER_QUADRAT
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
            self.__Cs = cs
        q2 = self.q2
        a = gv * LAI * b / (d * vg_0 * cos(VZA)) * cos(VZA) / cos(SZA)
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
        return qq1, qq2, qq1b, qq2b

    def _ls(self):
        '''compute cv, cs,lo_90, mu'''
        LAI = self.__domain.tree.LAI
        b = self.__domain.area
        d = self.__domain.n_tree
        v = self.__domain.tree.V
        gs = self.__Gs
        gv = self.__Gv
        ss = self.__AOP_tree.Ss
        sv = self.__AOP_tree.Sv
        r = self.__domain.tree.R
        shape = self.__domain.tree.shape
        self.__mu = LAI * b / (d * v)
        self.__Ls = self.__Gs * 1.0
        self.__H = 1 / self.__mu
        self.__Lambda_m = self.__H * tan(self.__geovi.xi)
        mu = self.__mu
        if shape == 'CONE_CYLINDER':
            lo_90 = LAI * b / (v * d) * pi * r / 2.
        else:
            lo_90 = LAI * b / (v * d)
        cs = gs * ss * mu / lo_90
        cv = gv * sv * mu / lo_90
        return lo_90, cs, cv

    def _ptg_sub(self, option: str, max_integral: float, increment: float):
        i = 0
        f_thelta, lt, cold, hot, w, ptg, f, H, XI, lambda_m = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        d = self.__domain.n_tree
        b = self.__domain.area
        r = self.__domain.tree.R
        fd = self.__Fd
        if 'CANOPY' == option.upper():
            w = self.__domain.tree.leaf.ws
            lt = self.__Ls
            cold = self.__PT_Cold
            hot = 1 - self.__Pig
            XI = self.__geovi.xi
            H = self.__H
            lambda_m = H * tan(XI)
        elif 'GROUND' == option.upper():
            w = self.__Wt
            lt = self.__Lt
            cold = self.__PS
            hot = self.__Pig
            XI = self.__geovi.xi
            H = self.__AOP_tree.H
            lambda_m = self.__Lambda_m
        elif 'NADIR' == option.upper():
            w = sqrt(pi * r ** 2)
            lt = self.__OmegaT * pi * (r ** 2) * d / b
            cold = self.__Pvg * (1 - self.__Pig)
            hot = self.__Viewed_shadow
            medium = hot - cold
            if medium < 0: hot = cold
            XI = self.__geovi.VZA
            H = self.__domain.tree.Hc / 3. + self.__domain.tree.Hb + self.__domain.tree.Ha
            lambda_m = H * tan(XI)
        else:
            raise ValueError('Bad option, it must be GROUND, NADIR or CANOPY (case insensitive)')

        if XI < pi / 2.:
            i = np.arange(0, max_integral)
            i_tmp = lambda_m + increment * i
            i_tmp1 = i_tmp[np.where(i_tmp != 0)]
            in1_e = np.exp(-lt * (1 + i_tmp1 / w)) / np.arctan(i_tmp1 / H)
            # in1_e = in1_e[np.where(in1_e >= 0.00000000001)]
            in1 = np.sum(in1_e)
            in2_e = np.exp(-lt * (1 + i_tmp / w))
            # in2_e = in2_e[np.where(in2_e >= 0.00000000001)]
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
            if self.__PS > self.__Pvg - zg: self.__PS = self.__Pvg - zg
            if self.__Pvg - zg < 0:
                self.__PS = 0
                # raise ValueError('Pvg is larger than ZG. Impossible!')
            fn = f * fd
            return zg, fn

    def _nadir(self):
        '''works like the hotspot function, but is applied to the shaded background viewed
        the original pig*pvg equation can induce and underestimation of the shaded background'''
        omega_t = self.__OmegaT
        r = self.__domain.tree.R
        d = self.__domain.n_tree
        b = self.__domain.area
        h = self.__AOP_tree.H
        hc = self.__domain.tree.Hc
        ha = self.__domain.tree.Ha
        SZA = self.__geovi.SZA
        e_r = self.__E_r
        sg_0 = self.__AOP_tree.Sg_0
        pig = self.__Pig

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
        return viewed_shadow, fd

    def _distance(self):
        R = self.__domain.tree.R
        D = self.__domain.n_tree
        B = self.__domain.area
        Lt = self.__OmegaT * pi * R ** 2 * D / B
        Wt = sqrt(pi * R ** 2)
        return Wt / Lt

    def _resize_tree(self, option: str):  # adjust Vg and Sg accroding to tree size
        D = self.__domain.n_tree
        n = self.__domain.n_quadrat
        aop_tree = self.__AOP_tree
        d = D / n
        vg = 0
        if 'VZA' == option.upper():
            vg_0 = aop_tree.Vg_0
        elif 'SZA' == option.upper():
            vg_0 = aop_tree.Sg_0
        else:
            raise ValueError('Wrong option, it must be VZA or SZA (case insensitive)')
        px = self.__domain.Px  # get poisson or neyman distribution arrays
        for i in range(self.__domain.MAX_TREE_PER_QUADRAT):
            if i > d:
                vg = vg + px[i] * vg_0 * d / i
            else:
                vg = vg + px[i] * vg_0
        return vg

    def _p_gap_ax_b(self, option: str):
        a = self.__A
        c = self.__C
        aop_tree = self.__AOP_tree
        omega_e = self.__domain.tree.Omega_E  # clumping index
        gamma_e = self.__domain.tree.gamma_E  # needle to shoot ratio
        LAI = self.__domain.tree.LAI  # leaf area index
        b = self.__domain.area  # domain size
        d = self.__domain.n_tree  # number of trees in the domain

        if 'VZA' == option.upper():
            za = self.__geovi.VZA
            vg = aop_tree.Vg_0  # viewed ground
            gv = (a * za + c) * omega_e / gamma_e
            pgapv = exp(-gv * LAI * b / (d * vg * cos(za)))
            gfoliage = a * za + co
            return gv, gapv, gfoliage
        elif 'SZA' == option.upper():
            za = self.__geovi.SZA
            vg = aop_tree.Sg_0  # sunlit ground
            gs = (a * za + c) * omega_e / gamma_e
            pgaps = exp(-gs * LAI * b / (d * vg * cos(za)))
            sg_0 = self.AOP_tree.Sg_0
            lo = LAI * b / (d * sg_0 * cos(za))
            return gs, pgaps, lo
        elif '0' == option.upper():
            za = 0
            r = self.__domain.tree.R
            vg = pi * r ** 2
            gs = (a * za + c) * omega_e / gamma_e
            pgap0 = exp(-gs * LAI * b / (d * vg * cos(za)))
            return pgap0
        elif 'LAI' == option.upper():
            za = cos(0.537 + 0.025 * LAI)
            vg = aop_tree.Vg_0_mean
            gs = (a * za + c) * omega_e / gamma_e
            pgapv_mean = exp(-gs * LAI * b / (d * vg * cos(za)))
            return pgapv_mean
        else:
            raise ValueError('Wrong option, it must be VZA, SZA, 0 or LAI (case insensitive)')

    def _p_gap_branch(self, option):
        LAI = self.__domain.tree.LAI
        aop_tree = self.__AOP_tree
        sv = aop_tree.Sv
        ss = aop_tree.Ss
        vg_0_mean = aop_tree.Vg_0_mean
        sg_0 = aop_tree.Sg_0
        vg_0 = aop_tree.Vg_0
        b = self.__domain.area
        v = self.__domain.tree.V
        r = self.__domain.tree.R
        gamma_e = self.__domain.tree.gamma_E
        d = self.__domain.n_tree
        hc = self.__domain.tree.Hc
        hb = self.__domain.tree.Hb
        alpha_l = self.__domain.tree.alpha_l
        alpha_b = self.__domain.tree.alpha_b
        ll = self.__domain.tree.Ll
        rb = self.__domain.tree.Rb
        ratio = self.__domain.tree.leaf.RATIO
        lb = LAI / ll
        mub = lb * b / (v * d)
        mul = ll * b / (v * d)

        if 'VZA' == option.upper():
            za = self.__geovi.VZA
            s = sv
        elif 'SZA' == option.upper():
            za = self.__geovi.SZA
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
        gfoliage = gl
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
            return gv, pgapv, gfoliage
        elif 'SZA' == option.upper():
            pgaps = p
            gs = cos(za) * sg_0 * d * log(1 / p) / (b * LAI)
            lo = LAI * b / (d * sg_0 * cos(za))
            return gs, pgaps, lo
        elif '0' == option.upper():
            pgap0 = p
            return pgap0
        else:
            pgapv_mean = p
            return pgapv_mean

    def _fo(self, option: str):
        r = self.__domain.tree.R
        fo = 0
        svg0 = r * r * pi
        fr = self.__domain.Fr
        # psvg0 = psg_hot0
        psvg0 = self.__PSG_HOT0
        pvg_nadir = self.__Pvg_nadir
        if 'VZA' == option:
            svg = self.__AOP_tree.Vg_0
            theta = self.__geovi.VZA
        elif 'SZA' == option:
            svg = self.__AOP_tree.Sg_0
            theta = self.__geovi.SZA
        else:
            raise ValueError('bad option in _fo function, it must be VZA or SZA')

        if svg0 != 0:
            fo = fr * exp(-(svg - svg0) / svg0 * 2 * theta / pi)
        if fo < 0:
            raise ValueError('possible problem with _fo function')
        if 'VZA' == option:
            PVG = self.__Pvg
            Pvg = self.__Pvg + (psvg0 - pvg_nadir) * fo
            Pv = self.__Pv * (1 - fo)
            return PVG, Pvg, Pv
        elif 'SZA' == option:
            PIG = self.__Pig
            Pig = self.__Pig + (psvg0 - pvg_nadir) * fo
            Ps = self.__Ps * (1 - fo)
            return PIG, Pig, Ps
        else:
            raise ValueError('bad option in _fo function, it must be VZA or SZA')

    def _overlap_v1(self, option: str, dist=None):
        '''
        subroutine to calculate gap fraction (Pvg & Pig), Pv|Ps, Pvc|Psc
        @param option: option to determine what kind of returns
        @param return_type: request for return type, default: None
        @param restrain: restraints when option:SZA, default: None, if it equals to 'POISSON', Px will be replaced By px_poisson
        @return: psv: overlap probability
        @return: psvc: cone overlap probability
        '''
        b = self.__domain.area  # domain area
        d = self.__domain.n_tree  # total number of trees in the domain
        n = self.__domain.n_quadrat  # number of quadrats in the domain
        r = self.__domain.tree.R  # crown radius
        NN = self.__domain.MAX_TREE_PER_QUADRAT  # maximum amount of trees that is allowed in a quadrat
        A = b / n  # quadrat size
        i_0 = d / n  # number of trees in a qudrat
        # px = self.__domain.Px  # an array of poisson or neyman distribution probabilities
        if 'SZA' == option and dist is not None:
            px = self.__domain.Px_poisson  # an array of poisson or neyman distribution probabilities
        else:
            px = self.__domain.Px  # an array of poisson or neyman distribution probabilities

        if 'VZA' == option.upper():
            pgap = self.__PgapV  # gap fraction in one crown, viewed from viewer
            a = self.__AOP_tree.Vg_0  # tree projection on the ground, viewed from viewer
            ac = self.__AOP_tree.Vgc  # tree cone projection on the ground, viewed from viewer
        elif 'SZA' == option.upper():
            pgap = self.__PgapS  # gap fraction in one crown, viewed from sun
            a = self.__AOP_tree.Sg_0  # tree projection on the ground, viewed from sun
            ac = self.__AOP_tree.Sgc  # tree cone projection on the ground, viewed from sun
        elif 'LAI' == option.upper():
            pgap = self.__PgapV_mean  # mean gap fraction in one crown, evaluated by LAI
            a = self.__AOP_tree.Vg_0_mean  # tree projection on the ground, evaluated by LAI
            ac = 0
        else:
            pgap = self.__Pgap0  # gap fraction from nadir view
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
        ptreer = [0] * NN

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

            if pgap == 0 and j == 0:
                p = pt
                pc = ptc
            else:
                p = p + pt * pow(pgap, j * 1.)
                pc = pc + ptc * pow(pgap, j * 1.)
                ptree[j] = pt
                ptreec[j] = ptc
                ptreer[j] = ptree[j]
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
            Pvg, Pv, Pvc, Ptreev = p, psv, psvc, ptreer
            return Pvg, Pv, Pvc, Ptreev
        elif 'SZA' == option.upper():
            Pig, Ps, Psc, Ptrees = p, psv, psvc, ptreer
            return Pig, Ps, Psc, Ptrees
        elif 'LAI' == option.upper():
            Pvg_mean = p
            return Pvg_mean
        elif 'NADIR' == option.upper():
            Pvg_nadir = p
            return Pvg_nadir
        else:
            raise ValueError('Wrong option! it must be one of the followings: VZA, SZA, LAI, NADIR')

    def _multiple_scattering(self, background_ref: np.array):
        '''this function computes the amount of eletromagnetic radiation reaching the four
        components(sunlit,shaded background and foliage) due to mutliple scattering.
        It serves as the basis for the hyperspectral mode of 5-scale
        Reference:Chen, J. M., & Leblanc, S. G. (2001). Multiple-scattering scheme useful
        for geometric optical modeling. IEEE Transactions on Geoscience and Remote Sensing
        , 39(5), 1061-1071.  '''
        qq1b = self.__QQ1B
        qq2b = self.__QQ2B
        pti = self.__Pti
        pg = self.__PG
        pt = self.__PT
        zg = self.__ZG
        zt = self.__ZT
        ft = self.__Ft
        xi = self.__geovi.xi
        pig = self.__Pig
        delta_LAI = self.__domain.tree.DeltaLAI
        ha = self.__domain.tree.Ha
        hb = self.__domain.tree.Hb
        hc = self.__domain.tree.Hc
        cp = self.__domain.tree.leaf.cp
        gv = self.__Gv
        sv = self.__AOP_tree.Sv
        mu = self.__mu
        cs = self.__Cs
        gamma_e = self.__domain.tree.gamma_E
        shape = self.__domain.tree.shape
        lo_90 = self.__Lo_90
        r = self.__domain.tree.R
        b = self.__domain.area
        d = self.__domain.n_tree
        e_r = self.__E_r
        SZA = self.__geovi.SZA
        VZA = self.__geovi.VZA
        LAI = self.__domain.tree.LAI
        foliage_ref = np.array(self.__domain.tree.leaf.DHR).flatten()
        foliage_trans = np.array(self.__domain.tree.leaf.DHT).flatten()
        wave = np.array(self.__domain.tree.leaf.wv).flatten()
        pvg_mean = self.__Pvg_mean
        cv_mean, f_s_trest, f_s_t, f_T_T, F_G_T, F_g_T, f_zg_t, f_t_t, f_zt_t, f_t_zt = \
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        f_s_g, f_t_g, f_tt2, f_tt3, f_tt_zt, f_tt_g, hi, hj, li, lj = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        num, omega_total, pgapv_mean, p_tot, q1_mean, q2_mean, q1b_mean, q2b_mean, rr, thelta = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        intensity = atan((hb + hc) / (2 * r))
        intensity = intensity - SZA
        if intensity < 0: intensity = -intensity
        intensity = cos(intensity)

        # intersect wave with the wavelengths of background_foliage
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
            q1_mean = self._q1(np.array([pi / 2 + thelta_half_mean]), 0)
            q1_mean = q1_mean[0]
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

            li = delta_LAI + np.arange(0, (LAI - delta_LAI) // delta_LAI + 1) * delta_LAI
            arg4 = 0.5 * omega_total * li / np.cos(np.arcsin(0.537 + 0.025 * li))
            f_s_t = np.sum(0.5 * np.exp(-arg4) * np.exp(-cv * li) * delta_LAI)
            f_s_trest = np.sum(0.5 * np.exp(-arg4) * delta_LAI)
            arg5 = 0.5 * omega_total * (LAI - li) / np.cos(np.arcsin(0.537 + 0.025 * (LAI - li)))
            F_G_T = np.sum(0.5 * np.exp(-arg5) * delta_LAI)
            p_tot = np.sum(np.exp(-cv * li) * delta_LAI)
            hi = (hb + hc / 3) - li * (hb + hc / 3) / LAI
            inc_thelta = delta_LAI
            thelta_min = (hi - inc_thelta) / e_r
            thelta_min = 0.5 * (pi / 2 + np.arctan(thelta_min))
            thelta_max = ((hb + hc / 3) - hi) / e_r
            thelta_max = 0.5 * (pi / 2 + np.arctan(thelta_max))
            flag = thelta_min > thelta_max
            temp = thelta_min.copy()
            thelta_min[flag] = thelta_max[flag]
            thelta_max[flag] = temp[flag]
            f_tt2, f_tt3 = [], []
            for min, max in list(zip(thelta_min, thelta_max)):
                ub = (max - min) // inc_thelta + 1
                thelta = min + np.arange(0, ub) * inc_thelta
                thelta_h = pi - 2 * thelta
                # do not enter the view of a np.ndarray into _q1, othersize the array will change, instead used deep copy
                in3 = self._q1(thelta_h.copy(), 0) * np.cos(thelta) * np.sin(thelta) * inc_thelta * 10 * pi / 180
                p = np.radians(np.arange(0, 180, 10))
                in1 = np.dot(np.cos(thelta_h[:, np.newaxis]), np.ones_like(p[np.newaxis, :]))
                in2 = np.dot(np.sin(thelta_h[:, np.newaxis]), np.cos(p[np.newaxis, :]))
                step = 10 * pi / 180
                del1 = np.sum(1 - cp * (cos(SZA) * in1 + sin(SZA) * in2) / pi, axis=1).flatten() * step
                del2 = np.sum(cp * (cos(SZA) * in1 + sin(SZA) * in2) / pi, axis=1).flatten() * step
                DEL = np.cumprod(del1[::-1])
                elm = np.sum(DEL * in3[::-1])
                f_tt2.append(elm)
                f_tt3.append(elm * del2[-1])
            f_t_zt = np.sum(np.array(f_tt2) * delta_LAI)
            f_tt_zt = np.sum(np.array(f_tt3) * delta_LAI)

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
        return wave, r0

    def _q1(self, thelta: np.ndarray, phi):
        '''calcualte appr. of Q1tot for MS Scheme'''
        SZA = self.__geovi.SZA
        cp = self.__domain.tree.leaf.cp
        gv = self.__Gv
        cs = self.__Cs
        hb = self.__domain.tree.Hb
        hc = self.__domain.tree.Hc
        r = self.__domain.tree.R
        LAI = self.__domain.tree.LAI
        b = self.__domain.area
        d = self.__domain.n_tree
        lo_90 = self.__Lo_90

        thelta[thelta < 0] = -thelta[thelta < 0]
        diff = np.cos(SZA) * np.cos(thelta) + np.sin(thelta) * np.cos(phi) * np.sin(SZA)
        diff[diff < 0] = -diff[diff < 0]
        del0 = 1 - diff * cp / pi
        del0[del0 < 0] = -del0[del0 < 0]
        del0[del0 > 1] = 1
        thelta[thelta > pi / 2] = pi - thelta[thelta > pi / 2]
        thelta[thelta == 0] = 0.0000000001
        flag = np.sqrt((thelta - pi / 2) * (thelta - pi / 2)) < 0
        thelta[flag] = pi / 2 - 0.0000001
        cv = gv / np.sin(thelta)
        vg_thelta = 2 * np.tan(thelta) * r * (hb + hc) + pi * (r ** 2)
        pgapv_thelta = np.exp(-cv * LAI * b / (d * vg_thelta * np.cos(thelta)))
        q1_thelta = (1. - np.exp(-(cs * lo_90 + cv * lo_90))) * cs * cv / (cs + cv) / (1 - pgapv_thelta)
        q1_thelta = q1_thelta * del0
        flag1 = np.where(q1_thelta > 1, 1, 0)
        count1 = np.sum(flag1)
        flag2 = np.where(q1_thelta < 0, 1, 0)
        count2 = np.sum(flag2)
        if count1 > 0:
            raise ValueError('Q1_thelta is larger than 1, impossible!')
        if count2 > 0:
            raise ValueError('Q1_thelta is smaller than 0, impossible!')
        return q1_thelta


def simulate_Rc():
    pass


if __name__ == '__main__':
    start = time.time()
    leaf = Needle(diameter=40, thickness=1.6, xu=0.045, baseline=0.0005, albino=2, Cab=200, Cl=40, Cp=1, Cw=100)
    # tree = SpheroidTree(leaf=leaf, R=1, Ha=1, Hb=5, LAI=3.5, Omega_E=0.8, gamma_E=1, ge_choice='BRANCH', alpha_l=-1,
    #                    alpha_b=25)
    tree = ConeTree(leaf=leaf, R=1, alpha=13, Ha=1, Hb=5, LAI=3.5, Omega_E=0.8, gamma_E=1, ge_choice='BRANCH',
                    alpha_b=25, alpha_l=-1)
    domain = Domain(tree=tree, area=10000, n_tree=6000, n_quadrat=40, Fr=0.0, m2=0)
    vza = range(-80, 80, 5)
    bili_pg, bili_pt, bili_zg, bili_zt = [], [], [], []
    for i in vza:
        if i < 0:
            geovi = GeoVI(SZA=20, VZA=abs(i), phi=180)
        else:
            geovi = GeoVI(SZA=20, VZA=i, phi=0)
        aop_domain = AOPDomain(geovi=geovi, domain=domain)
        ro = aop_domain.Rc
        zg, pg, pt, zt = aop_domain.ZG, aop_domain.PG, aop_domain.PT, aop_domain.ZT
        bili_pg.append(pg), bili_pt.append(pt), bili_zg.append(zg), bili_zt.append(zt)
    vza = list(vza)
    rs = np.vstack((vza, bili_pg, bili_pt, bili_zg, bili_zt))
    rs = np.transpose(rs)
    np.save('abc_cone.npy', rs)
    end = time.time()
    print('Record time: %.5f' % (end - start))
    fig, ax = plt.subplots()
    l1, = ax.plot(vza, bili_pg)
    l2, = ax.plot(vza, bili_pt)
    l3, = ax.plot(vza, bili_zg)
    l4, = ax.plot(vza, bili_zt)
    ax.set(xlabel='vza', ylabel='areal proportions')
    ax.set_yticks([0, 0.5, 1])
    ax.legend((l1, l2, l3, l4), ('sunlit ground', 'sunlit foliage', 'shaded ground', 'shaded foliage'),
              loc='upper right', shadow=True)
    ax.grid()
    plt.show()
