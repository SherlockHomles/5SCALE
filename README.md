# 5SCALE

5SCALE is a geometrical optics model proposed by Chen, J.M., &amp; Leblanc, S.G. (1997). It is often used to simulate
canopy reflectance factor based on inputs of leaf and canopy biochemical and biophysical traits. **This is a python
version** written under object-oriented coding(OOC) rule

# Usage
run AOPDomain.py

# Notes
The leaf class includes two subclasses: Broadleaf and Needle, corresponding to two major leaf types. They use
different kinds of leaf optical properties models to simulate leaf directional-hemispherical reflectance and
transmittance (DHR&DHT), PROSPECT for Broadleaf while LIBERTY for Needle by default. **The code includes LIBERTY only,
PROSPECT model can be downloaded from** http://teledetection.ipgp.jussieu.fr/prosail/
