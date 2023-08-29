# Functions to convert Longitude and Latitude to Easting and Northing, or vice versa
import math

# Ellipsoid parameters for different datums (m): semi-major axis, a, and
# semi-minor axis, b.
datum_ellipsoid = {
    # Airy 1830 ellipsoid
    'osgb36': {'a': 6.377563396e6,
               'b': 6.356256909e6
              },
    # WGS84 ellipsoid parameters
    'wgs84':  {'a': 6.378137e6,
               'b': 6.3567523141e6
              },
    }

# Transverse Mercator projection parameters: Map coordinates of true origin,
# (E0, N0), scale factor on central meridian, F0, true origin (phi0, lambda0).
N0 = -100000
E0 = 400000
F0 = 0.9996012717
phi0 = math.radians(49)
lambda0 = math.radians(-2)

def fM(phi, a, b):
    """Return the parameter M for latitude phi using ellipsoid params a, b."""

    n = (a-b)/(a+b)
    n2 = n**2
    n3 = n * n2
    dphi, sphi = phi - phi0, phi + phi0
    M = b * F0 * (
            (1 + n + 5/4 * (n2+n3)) * dphi
          - (3*n + 3*n2 + 21/8 * n3) * math.sin(dphi) * math.cos(sphi)
          + (15/8 * (n2 + n3)) * math.sin(2*dphi) * math.cos(2*sphi)
          - (35/24 * n3 * math.sin(3*dphi) * math.cos(3*sphi))
        )
    return M

def dms_pretty_print(d, m ,s, latlong, ndp=4):
    """Return a prettified string for angle d degrees, m minutes, s seconds."""

    if latlong=='latitude':
        hemi = 'N' if d>=0 else 'S'
    elif latlong=='longitude':
        hemi = 'E' if d>=0 else 'W'
    else:
        hemi = '?'
    return '{d:d}° {m:d}′ {s:.{ndp:d}f}″ {hemi:1s}'.format(
                d=abs(d), m=m, s=s, hemi=hemi, ndp=ndp)

def deg_to_dms(deg, pretty_print_latlong=None, ndp=4):
    """Convert from decimal degrees to degrees, minutes, seconds."""

    m, s = divmod(abs(deg)*3600, 60)
    d, m = divmod(m, 60)
    if deg < 0:
        d = -d
    d, m = int(d), int(m)

    if pretty_print_latlong:
        return dms_pretty_print(d, m, s, pretty_print_latlong)
    return d, m, s

def dms_to_deg(d, m, s):
    """Convert from degrees, minutes, seconds to decimal degrees."""
    return d + m/60 + s/3600

def get_prms(phi, a, F0, e2):
    """Calculate and return the parameters rho, nu, and eta2."""

    rho = a * F0 * (1-e2) * (1-e2*math.sin(phi)**2)**-1.5
    nu = a * F0 / math.sqrt(1-e2*math.sin(phi)**2)
    eta2 = nu/rho - 1
    return rho, nu, eta2

def os_to_ll(E, N, datum='osgb36'):
    """Convert from OS grid reference (E, N) to latitude and longitude.

    Latitude, phi, and longitude, lambda, are returned in degrees.

    """

    a, b = datum_ellipsoid[datum]['a'], datum_ellipsoid[datum]['b']
    e2 = (a**2 - b**2)/a**2
    M, phip = 0, phi0
    while abs(N-N0-M) >= 1.e-5:
        phip = (N - N0 - M)/(a*F0) + phip
        M = fM(phip, a, b)

    rho, nu, eta2 = get_prms(phip, a, F0, e2)

    tan_phip = math.tan(phip)
    tan_phip2 = tan_phip**2
    nu3, nu5 = nu**3, nu**5
    sec_phip = 1./math.cos(phip)

    c1 = tan_phip/2/rho/nu
    c2 = tan_phip/24/rho/nu3 * (5 + 3*tan_phip2 + eta2 * (1 - 9*tan_phip2))
    c3 = tan_phip / 720/rho/nu5 * (61 + tan_phip2*(90 + 45 * tan_phip2))
    d1 = sec_phip / nu
    d2 = sec_phip / 6 / nu3 * (nu/rho + 2*tan_phip2)
    d3 = sec_phip / 120 / nu5 * (5 + tan_phip2*(28 + 24*tan_phip2))
    d4 = sec_phip / 5040 / nu**7 *  (61 + tan_phip2*(662 + tan_phip2*
                                                    (1320 + tan_phip2*720)))
    EmE0 = E - E0
    EmE02 = EmE0**2
    phi = phip + EmE0**2 * (-c1 + EmE02*(c2 - c3*EmE02))
    lam = lambda0 + EmE0 * (d1 + EmE02*(-d2 + EmE02*(d3 - d4*EmE02)))
    return math.degrees(phi), math.degrees(lam)

def ll_to_os(phi, lam, datum='osgb36'):
    """Convert from latitude and longitude to OS grid reference (E, N).

    Latitude, phi, and longitude, lambda, are to be provided in degrees.

    """

    phi, lam = math.radians(phi), math.radians(lam)
    a, b = datum_ellipsoid[datum]['a'], datum_ellipsoid[datum]['b']
    e2 = (a**2 - b**2)/a**2
    rho, nu, eta2 = get_prms(phi, a, F0, e2)
    M = fM(phi, a, b)

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    cos_phi2 = cos_phi**2
    cos_phi3 = cos_phi2 * cos_phi
    cos_phi5 = cos_phi3 * cos_phi2
    tan_phi2 = math.tan(phi)**2
    tan_phi4 = tan_phi2 * tan_phi2
    
    a1 = M + N0
    a2 = nu/2 * sin_phi * cos_phi
    a3 = nu/24 * sin_phi * cos_phi3 * (5 - tan_phi2 + 9*eta2)
    a4 = nu/720 * sin_phi * cos_phi5 * (61 - 58*tan_phi2 + tan_phi4)
    b1 = nu * cos_phi
    b2 = nu/6 * cos_phi3 * (nu/rho - tan_phi2)
    b3 = nu/120 * cos_phi5 * (5 - 18*tan_phi2 + tan_phi4 + eta2*(14 -
                              58*tan_phi2))
    lml0 = lam - lambda0
    lml02 = lml0**2
    N = a1 + lml02 * (a2 + lml02*(a3 + a4*lml02))
    E = E0 + lml0 * (b1 + lml02*(b2 + b3*lml02))
    return E, N