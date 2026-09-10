"""Stokes/Mueller conventions and material hooks (read before adding an adapter).

Stokes order is [I,Q,U,V], Q=|E1|²-|E2|², U=2 Re(E1 E2*),
V=2 Im(E1 E2*), with exp(-i omega t) phasors. e2 = k cross e1,
where k is the PHYSICAL photon propagation direction, opposite backward travel.
The reference e1 is global +z projected normal to k, falling back to +x at poles.
All rotations below are changes of basis, not physical polarizer rotations.
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn


def basis(k):
    k=np.asarray(k,float); axis=np.broadcast_to([0.,0.,1.],k.shape).copy()
    axis[np.abs(k[...,2])>.999999]=[1.,0.,0.]
    e1=axis-np.sum(axis*k,axis=-1)[...,None]*k
    e1/=np.linalg.norm(e1,axis=-1)[...,None]
    return e1,np.cross(k,e1)


def rotation(angle):
    angle=np.asarray(angle); c=np.cos(2*angle); s=np.sin(2*angle)
    m=np.zeros((*angle.shape,4,4)); m[...,0,0]=m[...,3,3]=1
    m[...,1,1]=m[...,2,2]=c; m[...,1,2]=s; m[...,2,1]=-s
    return m


def transport(start,end):
    """Mueller map from end-ray basis to start-ray basis along a planar curve.

    Isotropic refraction parallel-transports polarization; it is NOT a polarizer.
    Do not apply Fresnel losses at artificial atmospheric integration levels.
    """
    k0,k1=-start,-end; normal=np.cross(k0,k1); norm=np.linalg.norm(normal,axis=-1)
    normal=np.divide(normal,norm[...,None],out=np.zeros_like(normal),where=norm[...,None]>1e-12)
    a0,b0=basis(k0); a1,b1=basis(k1)
    psi0=np.arctan2(np.sum(normal*b0,axis=-1),np.sum(normal*a0,axis=-1))
    psi1=np.arctan2(np.sum(normal*b1,axis=-1),np.sum(normal*a1,axis=-1))
    return rotation(np.where(norm>1e-12,psi1-psi0,0.))


def jones_diagonal(parallel,perpendicular):
    """Nondepolarizing Mueller matrix; amplitudes use the scattering-plane basis."""
    a,b=np.broadcast_arrays(parallel,perpendicular); product=a*np.conj(b)
    m=np.zeros((*a.shape,4,4)); aa=np.abs(a)**2; bb=np.abs(b)**2
    m[...,0,0]=m[...,1,1]=(aa+bb)/2
    m[...,0,1]=m[...,1,0]=(aa-bb)/2
    m[...,2,2]=m[...,3,3]=product.real
    m[...,2,3]=-product.imag; m[...,3,2]=product.imag
    return m


def rayleigh(cosine):
    """Ideal dipole Rayleigh; integral P11 dOmega = 4pi.

    Molecular depolarization is currently neglected in this phase matrix, as
    in the scalar phase. The extinction still uses the existing King factor.
    A future depolarization model MUST change all matrix elements consistently.
    """
    return 1.5*jones_diagonal(np.asarray(cosine),np.ones_like(cosine))


def mie_coefficients(size_parameter,refractive_index):
    """Homogeneous sphere Lorenz-Mie coefficients (Bohren & Huffman convention).

    HOOK: size_parameter=2*pi*radius_um/vacuum_wavelength_um. Material index is
    RELATIVE to surrounding air, n+i*k with k>=0 under exp(-i omega t).
    This is not the atmospheric index used by the ray tracer. For distributions,
    mix cross-section-weighted Mueller matrices, NOT radius-averaged indices.
    Restrict x<=100 for this direct Bessel implementation; large/absorbing spheres
    need validated logarithmic-derivative recurrences, not a larger loop bound.
    """
    x=float(size_parameter); m=complex(refractive_index)
    if not .0001<=x<=100 or m.real<=0 or m.imag<0 or abs(m*x)>1000:
        raise ValueError('Mie inputs require 1e-4<=x<=100, n>0, k>=0 and |m*x|<=1000')
    ell=np.arange(1,int(np.ceil(x+4*x**(1/3)+2))+1)
    def psi(z): return z*spherical_jn(ell,z),spherical_jn(ell,z)+z*spherical_jn(ell,z,True)
    px,dx=psi(x); pm,dm=psi(m*x)
    xi=px+1j*x*spherical_yn(ell,x)
    dxi=dx+1j*(spherical_yn(ell,x)+x*spherical_yn(ell,x,True))
    a=(m*pm*dx-px*dm)/(m*pm*dxi-xi*dm)
    b=(pm*dx-m*px*dm)/(pm*dxi-m*xi*dm)
    qsca=2/x**2*np.sum((2*ell+1)*(abs(a)**2+abs(b)**2))
    qext=2/x**2*np.sum((2*ell+1)*(a+b).real)
    return a,b,float(qsca),float(qext)


def mie_matrix(cosine,size_parameter,refractive_index):
    a,b,qsca,_=mie_coefficients(size_parameter,refractive_index)
    mu=np.asarray(cosine); previous=np.zeros_like(mu); current=np.ones_like(mu)
    s1=np.zeros_like(mu,dtype=complex); s2=s1.copy()
    for l,(an,bn) in enumerate(zip(a,b),1):
        tau=l*mu*current-(l+1)*previous; factor=(2*l+1)/(l*(l+1))
        s1+=factor*(an*current+bn*tau); s2+=factor*(an*tau+bn*current)
        previous,current=current,((2*l+1)*mu*current-(l+1)*previous)/l
    return jones_diagonal(s2,s1)*(4/(size_parameter**2*qsca))


def scattering_matrix(old_travel,new_travel,fraction,g,mie=None):
    """Map incident to outgoing reference basis, for backward-ray direction pairs.

    `mie` is an optional (x, relative_complex_index). Without measured particle
    inputs, aerosol is an explicitly DEPOLARIZING HG matrix (only P11 nonzero).
    Never infer polarization from g: infinitely many matrices share that moment.
    """
    kout,kin=-old_travel,-new_travel
    mu=np.clip(np.sum(kout*kin,axis=-1),-1,1)
    molecular=rayleigh(mu)
    if mie is None:
        aerosol=np.zeros_like(molecular)
        aerosol[...,0,0]=(1-g*g)/(1+g*g-2*g*mu)**1.5
    else:
        aerosol=mie.evaluate(mu)
    m=fraction[...,None,None]*aerosol+(1-fraction[...,None,None])*molecular
    plane_in=kout-mu[...,None]*kin; plane_out=kin-mu[...,None]*kout
    a,b=basis(kin); psi_in=np.arctan2(np.sum(plane_in*b,axis=-1),np.sum(plane_in*a,axis=-1))
    a,b=basis(kout); psi_out=np.arctan2(np.sum(plane_out*b,axis=-1),np.sum(plane_out*a,axis=-1))
    return rotation(-psi_out)@m@rotation(psi_in)


def surface_matrix(travel,normal,index):
    """Specular interface map and local Kirchhoff emission in outgoing basis."""
    reflected=travel-2*np.sum(travel*normal,axis=1)[:,None]*normal
    kout,kin=-travel,-reflected
    s=np.cross(kout,normal); norm=np.linalg.norm(s,axis=1)
    a,_=basis(kout)
    s=np.divide(s,norm[:,None],out=np.cross(kout,a),where=norm[:,None]>1e-12)
    pout=np.cross(s,kout); pin=np.cross(s,kin)
    a,b=basis(kout); out_angle=np.arctan2(np.sum(pout*b,axis=1),np.sum(pout*a,axis=1))
    a,b=basis(kin); in_angle=np.arctan2(np.sum(pin*b,axis=1),np.sum(pin*a,axis=1))
    local=fresnel_reflection(np.abs(np.sum(travel*normal,axis=1)),index)
    mapping=rotation(-out_angle)@local@rotation(in_angle)
    emissivity=np.zeros((len(travel),4)); emissivity[:,0]=1-local[:,0,0]; emissivity[:,1]=-local[:,0,1]
    emissivity=np.einsum('nij,nj->ni',rotation(-out_angle),emissivity)
    return reflected,mapping,emissivity


def fresnel_reflection(cosine,index):
    """Smooth dielectric/absorbing interface in p/s basis, no roughness or coating.

    HOOK: index is substrate/air relative n+i*k at the actual wavelength.
    Do not label a constant demo index as measured seawater. Real sensor optics
    may require multilayer Jones matrices, incidence distributions, temperature,
    contamination, and polarization-dependent emission (Kirchhoff's law).
    """
    mu=np.asarray(cosine); m=complex(index); transmitted=np.sqrt(m*m-(1-mu*mu)+0j)
    rs=(mu-transmitted)/(mu+transmitted)
    rp=(m*m*mu-transmitted)/(m*m*mu+transmitted)
    return jones_diagonal(rp,rs)


class MiePhase:
    """Piecewise-linear normalized phase table with its exact sampling CDF.

    Sampling and evaluating DIFFERENT interpolants biases Monte Carlo. This
    class keeps both together. Its angular-grid convergence is distinct from
    photon-count convergence. Increase grid points for large size parameters.
    """
    def __init__(self,x,index,points=4097):
        self.mu=-np.cos(np.linspace(0,np.pi,points))
        self.matrix=mie_matrix(self.mu,x,index)
        area=np.trapezoid(self.matrix[:,0,0],self.mu)/2
        self.matrix/=area
        pdf=self.matrix[:,0,0]/2
        self.cdf=np.r_[0,np.cumsum(np.diff(self.mu)*(pdf[:-1]+pdf[1:])/2)]
        self.cdf[-1]=1.
        self.g=float(np.trapezoid(self.mu*pdf,self.mu))

    def evaluate(self,mu):
        return np.stack([np.interp(mu,self.mu,self.matrix[:,i,j]) for i in range(4) for j in range(4)],axis=-1).reshape((*np.shape(mu),4,4))

    def sample(self,rng,n):
        u=rng.random(n); j=np.clip(np.searchsorted(self.cdf,u,side='right')-1,0,len(self.mu)-2)
        left=self.matrix[j,0,0]/2; right=self.matrix[j+1,0,0]/2
        slope=(right-left)/(self.mu[j+1]-self.mu[j]); mass=u-self.cdf[j]
        # Stable quadratic inverse, also valid when the density is constant.
        offset=2*mass/(left+np.sqrt(np.maximum(left*left+2*slope*mass,0)))
        return self.mu[j]+offset
