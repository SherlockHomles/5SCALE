# 5SCALE

5SCALE is a geometrical optics model proposed by Chen, J.M., &amp; Leblanc, S.G. (1997). It is often used to simulate
canopy reflectance factor based on inputs of leaf and canopy biochemical and biophysical traits. It is the python
version of the model

Usage:
run AOPDomain.py

Notes:
The leaf class includes two subclasses: Broadleaf and Needle, corresponding to two distinct leaf types. They use
different kinds of leaf optical properties models to simulate leaf directional-hemispherical reflectance and
transmittance (DHR&DHT), PROSPECT for Broadleaf and LIBERTY for Needle by default. The code only include LIBERTY,
PROSPECT model can be downloaded from http://teledetection.ipgp.jussieu.fr/prosail/
