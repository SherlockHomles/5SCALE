import sys, os

# tmp_path = os.path.abspath(__file__)
# tmp_path = os.path.dirname(tmp_path)
# tmp_path = os.path.join(tmp_path, 'prosail')
# sys.path.append(tmp_path)
from typing import Optional
from prosail.prospect_d import run_prospect
import numpy as np
from math import pi, sin, cos, asin, tan
import pandas as pd
from abc import abstractmethod, ABC


def tav(teta: float, ref: float):
    s = np.size(ref, 0)
    teta = np.radians(teta)
    r2 = np.square(ref)
    rp = r2 + 1
    rm = r2 - 1
    a = np.square(ref + 1) / 2
    k = -np.square(r2 - 1) / 4
    ds = sin(teta)
    k2 = np.square(k)
    rm2 = np.square(rm)
    if 0 == teta:
        f = 4 * ref / (a * 2)
    else:
        if pi / 2 == teta:
            b1 = np.zeros([s, 1])
        else:
            b1 = np.sqrt(np.square(np.square(ds) - rp / 2) + k)
        b2 = np.square(ds) - rp / 2
        b = b1 - b2
        ts = (k2 / (6 * np.power(b, 3)) + k / b - b / 2) - (k2 / (6 * np.power(a, 3)) + k / a - a / 2)
        tp1 = -2 * r2 * (b - a) / np.square(rp)
        tp2 = -2 * r2 * rp * np.log(b / a) / rm2
        tp3 = r2 * (np.power(b, -1) - np.power(a, -1)) / 2
        tp4 = 16 * np.power(r2, 2) * (np.power(r2, 2) + 1) * np.log((2 * rp * b - rm2) / (2 * rp * a - rm2)) / (
                np.power(rp, 3) * rm2)
        tp5 = 16 * np.power(r2, 3) * (np.power(2 * rp * b - rm2, -1) - np.power(2 * rp * a - rm2, -1)) / np.power(
            rp, 3)
        tp = tp1 + tp2 + tp3 + tp4 + tp5
        f = (ts + tp) / (2 * np.power(ds, 2))
    return f


def LIBERTY(D: float, xu: float, thickness: float, baseline: float, albino: float, c_factor: float, w_factor: float,
            l_factor: float, p_factor: float):
    '''
    Main body of the LIBERTY model
    Reference: Dawson, T. P., Curran, P. J., & Plummer, S. E. (1998).
    LIBERTY—Modeling the effects of leaf biochemical concentration
    on reflectance spectra. Remote sensing of environment, 65(1), 50-60.
    Args:
        D: cell diameter
        xu: cell scattering factor
        baseline: used to calculate refractive index
        albino: denotes albino effects
        c_factor: chlorophyll content
        w_factor: water content
        l_factor: lignin content
        p_factor: protein content
    Returns:
        wv: wavelengths
        refl: leaf directional-hemispherical reflectance
        trans: leaf directional-hemispherical transmittance
    '''
    R = D / 2
    xd = xu
    i = np.arange(0, 421).reshape([421, 1])
    n = 1.4891 - baseline * i
    T12 = tav(90, n)
    T21 = T12 / np.square(n)
    me = 1 - T12
    mi = 1 - T21
    #mi = cal_mi(n)
    dat_path = os.path.dirname(os.path.abspath(__file__))
    k_a_f = os.path.join(dat_path, 'ALBINO.DAT')
    k_l_f = os.path.join(dat_path, 'LIGCELL.DAT')
    k_c_f = os.path.join(dat_path, 'PIGMENT.DAT')
    k_p_f = os.path.join(dat_path, 'PROTEIN.DAT')
    k_w_f = os.path.join(dat_path, 'WATER.DAT')
    k_a = np.array(pd.read_csv(k_a_f, header=None).values.tolist())
    k_l = np.array(pd.read_csv(k_l_f, header=None).values.tolist())
    k_c = np.array(pd.read_csv(k_c_f, header=None).values.tolist())
    k_p = np.array(pd.read_csv(k_p_f, header=None).values.tolist())
    k_w = np.array(pd.read_csv(k_w_f, header=None).values.tolist())
    l1 = np.size(k_w, 0)
    l2 = np.size(k_a, 0)
    tail = np.zeros([l1 - l2, 1], dtype=k_a.dtype)
    k_a = np.concatenate((k_a, tail), axis=0)
    k_c = np.concatenate((k_c, tail), axis=0)
    k1 = k_c * c_factor + k_w * w_factor + k_l * l_factor + k_a * albino + k_p * p_factor + baseline

    '''calcualte t---the transmittance of the cell model'''
    kd = k1 * R * 2
    M = 2 * (1 - (kd + 1) * np.exp(-kd)) / np.square(kd)
    t = (1 - mi) * M / (1 - mi * M)

    '''calcualte x---the reflectance of single cell layer'''
    xa = 1 - xu - xd
    x = xu / (1 - xa * t)

    '''clacualte R_inf----the reflectance of infinite cell layers with 2xme not in numerator'''
    a = x * (1 - 2 * x * me) * t
    b = a * me
    c = me + (1 - x) * (1 - me) * t
    d = 2 * x * me
    A = c
    B = -(b + d * c + 1)
    C = a + d
    R_inf = (-B - np.sqrt(np.square(B) - 4 * A * C)) / (2 * A)

    '''calcualte transmittance and reflectance: the equations used here are Eq.3,5,51,53-55
    Reference: Benford, 1946, Radiation in a diffusing medium'''
    Rs = 2 * x * me + x * (1 - 2 * x * me) * t
    Ts = np.sqrt((R_inf - Rs) * (1 - R_inf * Rs) / R_inf)
    layers = np.int64(thickness)
    f = thickness - layers
    numerator = np.power(Ts, 1 + f) * np.power(np.square(1 + Ts) - np.square(Rs), 1 - f)
    denominator1 = np.power(1 + Ts, 2 * (1 - f)) - np.square(Rs)
    denominator2 = 1 + (64 / 3) * f * (f - 0.5) * (f - 1) * 0.001
    denominator = denominator1 * denominator2
    T1f = numerator / denominator
    R1f = (1 + np.square(Rs) - np.square(Ts) - np.sqrt(
        np.square(1 + np.square(Rs) - np.square(Ts)) - 4 * np.square(Rs) * (1 - np.square(T1f)))) / (2 * Rs)
    Ti = np.ones([l1, 1], dtype=Rs.dtype)
    Ri = np.zeros([l1, 1], dtype=Rs.dtype)
    # if layers >1 else????????????????
    if layers >= 2:
        for i in range(1, layers):
            next_Ti = (Ti * Ts) / (1 - Ri * Rs)
            next_Ri = Ri + (np.square(Ti) * Rs) / (1 - Rs * Ri)
            Ti = next_Ti
            Ri = next_Ri
    trans = (Ti * T1f) / (1 - Ri * R1f)
    refl = Ri + (np.square(Ti) * R1f) / (1 - R1f * Ri)
    wv = np.arange(400, 2501, 5).tolist()
    refl = refl.tolist()
    trans = trans.tolist()
    return wv, refl, trans


#def cal_mi(ref):
#    n = ref.size
#    mi = np.zeros_like(ref)
#    tetac = np.degrees(np.arcsin(1 / ref))
#    width = pi / 180
#    for i in range(n):
#        mint = 0
#        for j in range(1, np.int64(np.ceil(tetac[i][0]))):
#            alpha = j * pi / 180
#            beta = asin(ref[i] * sin(alpha))
#            plus = alpha + beta
#            dif = alpha - beta
#            refl = 0.5 * (((sin(dif) * sin(dif)) / (sin(plus) * sin(plus))) + (
#                    (tan(dif) * tan(dif)) / (tan(plus) * tan(plus))))
#            mint = mint + (refl * sin(alpha) * cos(alpha) * width)
#        mi[i] = 1 - np.square(sin(tetac[i] * pi / 180)) + 2 * mint
#    return mi


class LeafSpecies(ABC):
    '''
    this abstract class stores species-specific leaf biochemical and biophysical traits
    Attributes:
        DHR: leaf directional-hemispherical reflectance(DHR)
        DHT: leaf directional-hemispherical transmittance(DHT)
        wv: wavelengths
    Methods:
        _spectra_simulation: simulate DHR and DHT with leaf optical properties models
    Notes:
        DHR and DHT are regarded as inherent optical properties of leaves since they are independent of view or
    illumination geometries, they are calculated by leaf optical properties model
    '''

    @property
    @abstractmethod
    def DHR(self):
        '''
        directional-hemispherical reflectance
        '''
        pass

    @property
    @abstractmethod
    def DHT(self):
        '''
        directional-hemispherical transmittance
        '''
        pass

    @property
    @abstractmethod
    def wv(self):
        '''
        wavelengths, its length should be the same with those of DHR and DHT
        '''
        pass

    @abstractmethod
    def _spectra_simulation(self):
        pass


class Leaf(object):
    '''
    this class stores leaf biochemical and biophysical traits that is independent of leaf species
    Attributes:
        Cab: chlorophyll
        Cw: leaf water content
        Cm: dry matter, default: None
        Car: carotenoid, default: None
        Cbrown: brown pigment, default: None
        Cl: lignin, default: None
        Cp: protein, default: None
        ant: anthocyanin, default: None
        thickness: leaf thickness, default: None
        RATIO: leaf thickness/width ratio, its value should lie in 0-1
        ws: typical foliage element width
        cp: coefficient determined by optical properties of foliage elements (Eq.57), cp == 1 if foliages have
            lambertian surfaces used in Eq.57
        lambdas: selected wavelengths
    Methods:
        spectra_simulation: simulate DHR and DHT with leaf optical properties models
    Notes:
        DHR and DHT are regarded as inherent optical properties of leaves since they are independent of view or
    illumination geometries, they are calculated by leaf optical properties model
    '''

    def __init__(self, Cab: float, Cw: float, Cm: Optional[float] = None, Car: Optional[float] = None,
                 Cbrown: Optional[float] = None, Cl: Optional[float] = None, Cp: Optional[float] = None,
                 ant: Optional[float] = None, thickness: Optional[float] = None,
                 RATIO: float = 0.2, ws: float = 0.4, cp: float = 1.0) -> None:
        self.Cab = Cab
        self.Cw = Cw
        self.Cm = Cm
        self.Car = Car
        self.Cbrown = Cbrown
        self.Cl = Cl
        self.Cp = Cp
        self.ant = ant
        self.thickness = thickness
        self.RATIO = RATIO
        self.ws = ws
        self.cp = cp

    @property
    def cp(self):
        return self.__cp

    @cp.setter
    def cp(self, value: float):
        if value < 0:
            raise ValueError('cp must be larger than 0')
        self.__cp = value

    @property
    def ws(self):
        return self.__ws

    @ws.setter
    def ws(self, value: float):
        if value < 0:
            raise ValueError('ws must be larger than 0')
        self.__ws = value

    @property
    def Cab(self):
        return self.__Cab

    @Cab.setter
    def Cab(self, value: float):
        if value < 0:
            raise ValueError('Cab must be larger than 0')
        self.__Cab = value

    @property
    def Cw(self):
        return self.__Cw

    @Cw.setter
    def Cw(self, value):
        if value < 0:
            raise ValueError('Cw must be larger than 0')
        self.__Cw = value

    @property
    def Car(self):
        return self.__Car

    @Car.setter
    def Car(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError('Car must be larger than 0')
        self.__Car = value

    @property
    def Cm(self):
        return self.__Cm

    @Cm.setter
    def Cm(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError('Cm must be larger than 0')
        self.__Cm = value

    @property
    def Cbrown(self):
        return self.__Cbrown

    @Cbrown.setter
    def Cbrown(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError('Cbrown must be larger than 0')
        self.__Cbrown = value

    @property
    def Cl(self):
        return self.__Cl

    @Cl.setter
    def Cl(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError('Cl must be larger than 0')
        self.__Cl = value

    @property
    def Cp(self):
        return 0

    @Cp.setter
    def Cp(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError('Cp must be larger than 0')
        self.__Cp = value

    @property
    def ant(self):
        return 0

    @ant.setter
    def ant(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError('ant must be larger than 0')
        self.__ant = value

    @property
    def thickness(self):
        return self.__thickness

    @thickness.setter
    def thickness(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError('thickness must be larger than 0')
        self.__thickness = value

    @property
    def RATIO(self):
        return self.__RATIO

    @RATIO.setter
    def RATIO(self, value: float):
        if value < 0 or value > 1:
            raise ValueError('RATIO must lie in 0-1')
        self.__RATIO = value


class Broadleaf(Leaf, LeafSpecies):
    '''
    this abstract class stores biochemical and biophysical traits of a leaf, no matter it is a broadleaf or a needle
    Attributes:
        N: structural parameter, denoting number of plates
        Cab: chlorophyll contents, in g/cm2
        Car: carotenoid contents, in g/cm2
        Cw: leaf water content, in g/cm2 or cm
        Cm: dry matter per area, in g/cm2
        Cbrown: brown pigment content, in g/cm2
        ant: anthocyanin content, in g/cm2, if PROSPECT-D is used
        prospect_version: the version of propsect assigned to simulate leaf DHR and DHT, '5': prospect5, 'D': prospect-D
        RATIO: leaf thickness/width ratio
        DHR: leaf directional-hemispherical reflectance(DHR)
        DHT: leaf directional-hemispherical transmittance(DHT)
        lambdas: selected wavelengths, if assigned, it will only caluclate DHR &DHT at these wavelengths
    Methods:
        spectra_simulation: simulate DHR and DHT with PROSPECT model
    Notes:
        DHR and DHT are regarded as inherent optical properties of leaves since they are independent of view or
    illumination geometries, they are calculated by leaf optical properties model
        In this class, leaf DHR and DHT are simulated by PROSPECT which could be downloaded from
        http://teledetection.ipgp.jussieu.fr/prosail/
    '''

    def __init__(self, N: float, Cab: float, Car: float, Cbrown: float, Cw: float, Cm: float, RATIO: float = 0.2,
                 ws: float = 0.4, cp: float = 1.0, ant: Optional[float] = None, prospect_version: str = '5',
                 lambdas=None):
        '''
        this abstract class stores biochemical and biophysical traits of a leaf, no matter it is a broadleaf or a needle
        @param N:structural parameter, denoting number of plates
        @param Cab:chlorophyll contents, in g/cm2
        @param Car:carotenoid contents, in g/cm2
        @param Cbrown:brown pigment content, in g/cm2
        @param Cw:leaf water content, in g/cm2 or cm
        @param Cm:dry matter per area, in g/cm2
        @param RATIO:leaf thickness/width ratio
        @param ant:anthocyanin content, in g/cm2, if PROSPECT-D is used
        @param prospect_version:the version of prospect assigned to simulate leaf DHR and DHT, '5': prospect5, 'D': prospect-D
        '''
        self.N = N
        self.prospect_version = prospect_version
        self.lambdas = lambdas
        Leaf.__init__(self, Cab, Cw, Cm, Car, Cbrown, ant=ant, RATIO=RATIO, ws=ws, cp=cp)
        wv, refl, trans = self._spectra_simulation()
        self._wv = wv
        self._DHR = refl
        self._DHT = trans

    @property
    def lambdas(self):
        return self.__lambdas

    @lambdas.setter
    def lambdas(self, value):
        self.__lambdas = value

    @property
    def N(self):
        return self.__N

    @N.setter
    def N(self, value: float):
        if value < 0:
            raise ValueError('N must be larger than 0')
        if value > 10:
            raise ValueError('Outrageous N!')
        self.__N = value

    @Leaf.Cab.setter
    def Cab(self, value):
        if value > 200:
            raise ValueError('Outrageous Cab!')
        Leaf.Cab.fset(self, value)

    @Leaf.Car.setter
    def Car(self, value: float):
        if value > 100:
            raise ValueError('Outrageous Car!')
        Leaf.Car.fset(self, value)

    @Leaf.Cw.setter
    def Cw(self, value: float):
        if value > 2:
            raise ValueError('Outrageous Cw!')
        Leaf.Cw.fset(self, value)

    @Leaf.Cbrown.setter
    def Cbrown(self, value: float):
        if value > 100:
            raise ValueError('Outrageous Cbrown!')
        Leaf.Cbrown.fset(self, value)

    @Leaf.Cm.setter
    def Cm(self, value: float):
        if value > 2:
            raise ValueError('Outrageous Cm!')
        Leaf.Cm.fset(self, value)

    @Leaf.ant.setter
    def ant(self, value: Optional[float]):
        if 'D' == self.prospect_version and value is None:
            raise ValueError('ant cannot be None when prospect-D is assigned to simulate leaf R&T!')
        Leaf.ant.fset(self, value)

    @property
    def prospect_version(self):
        return self.__prospect_version

    @prospect_version.setter
    def prospect_version(self, value):
        if value.upper() not in ['D', '5']:
            raise ValueError('prospect_version must be 5 or D')
        self.__prospect_version = value.upper()

    @property
    def DHR(self):
        return self._DHR

    @property
    def DHT(self):
        return self._DHT

    @property
    def wv(self):
        return self._wv

    def _spectra_simulation(self):
        '''
        this class uses PROSPECT model to simulate leaf DHR and DHT, the model can be downloaded from
        http://teledetection.ipgp.jussieu.fr/prosail/
        '''
        N = self.N
        Cab = self.Cab
        Car = self.Car
        Cbrown = self.Cbrown
        Cw = self.Cw
        Cm = self.Cm
        ant = self.ant
        lambdas = self.lambdas
        prospect_version = self.prospect_version
        if '5' == prospect_version.upper():
            wv, refl, trans = run_prospect(N, Cab, Car, Cbrown, Cw, Cm, prospect_version=prospect_version,
                                           wvls=lambdas)
        else:
            wv, refl, trans = run_prospect(N, Cab, Car, Cbrown, Cw, Cm, ant=ant, prospect_version=prospect_version,
                                           wvls=lambdas)
        return wv, refl, trans


class Needle(Leaf, LeafSpecies):
    '''
    this abstract class stores biochemical and biophysical traits of a leaf, no matter it is a broadleaf or a needle
    Attributes:
        diameter: one of the two structural parameters needed by LIBERTY, denoting cell size
        xu: the scattering parameter of cell, its another structural parameter needed by LIBERTY
        thickness: leaf thickness, dividing diameter to get the number of plates
        Cab: chlorophyll contents, dimensionless, the ratio of chloropyll to dry matter
        Cw: leaf water content, the ratio of water to dry matter
        Cl: lignin content, dimensionless, the ratio of lignin to dry matter
        Cp: protein content, dimensionless, the ratio of protein to dry matter
        albino: reflectance and transmittance of an albino leaf
        baseline: a parameter used to calculate refractive index on leaf surfaces
        DHR: leaf directional-hemispherical reflectance(DHR)
        DHT: leaf directional-hemispherical transmittance(DHT)
        lambdas: selected wavelengths, if assigned, it will only caluclate DHR &DHT at these wavelengths
    Methods:
        _spectra_simulation: simulate DHR and DHT with LIBERTY model
    Notes:
        DHR and DHT are regarded as inherent optical properties of leaves since they are independent of view or
    illumination geometries, they are calculated by leaf optical properties model
    '''

    def __init__(self, diameter: float, thickness: float, xu: float, baseline: float, albino: float, Cab: float,
                 Cl: float, Cp: float, Cw: float, RATIO: float = 0.2, ws: float = 0.4, cp: float = 1.0):
        Leaf.__init__(self, Cab, Cw, Cl=Cl, Cp=Cp, thickness=thickness, RATIO=RATIO, ws=ws, cp=cp)
        self.diameter = diameter
        self.baseline = baseline
        self.albino = albino
        self.xu = xu
        wv, refl, trans = self._spectra_simulation()
        self.__wv = wv
        self.__DHR = refl
        self.__DHT = trans

    @property
    def diameter(self):
        return self.__diameter

    @diameter.setter
    def diameter(self, value: float):
        if value < 0:
            raise ValueError('diameter must be larger than 0')
        self.__diameter = value

    @property
    def baseline(self):
        return self.__baseline

    @baseline.setter
    def baseline(self, value: float):
        if value < 0:
            raise ValueError('baseline must be larger than 0')
        self.__baseline = value

    @property
    def albino(self):
        return self.__albino

    @albino.setter
    def albino(self, value: float):
        if value < 0:
            raise ValueError('albino must be larger than 0')
        self.__albino = value

    @property
    def xu(self):
        return self.__xu

    @xu.setter
    def xu(self, value: float):
        if value < 0:
            raise ValueError('xu must be larger than 0')
        self.__xu = value

    @property
    def DHR(self):
        return self.__DHR

    @property
    def DHT(self):
        return self.__DHT

    @property
    def wv(self):
        return self.__wv

    def _spectra_simulation(self):
        diameter = self.diameter
        Cab = self.Cab
        Cl = self.Cl
        Cp = self.Cp
        Cw = self.Cw
        thickness = self.thickness
        baseline = self.baseline
        albino = self.albino
        xu = self.xu
        wv, refl, trans = LIBERTY(diameter, xu, thickness, baseline, albino, Cab, Cw, Cl, Cp)
        return wv, refl, trans
