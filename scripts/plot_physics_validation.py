"""Plot external physics checks. Run: python scripts/plot_physics_validation.py."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mwir_atmos.optics.continuum import _mt_ckd_coefficients, _load_coefficients
from mwir_atmos.optics.rayleigh import _rayleigh_cross_section_cm2
from mwir_atmos.types import SpectralGrid
from mwir_atmos.constants import C2_RAD, LOSCHMIDT_CM3


def main():
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    with np.load(ROOT/'tests/reference_data/mt_ckd_4_3_fortran.npz') as data:
        for case,color in [(2,'#3178a7'),(5,'#c16527')]:
            d={n:data[n][data['case']==case] for n in data.files}
            nu=d['wavenumber_cm1']; p,t,vmr=(d[n][0] for n in ['pressure_mbar','temperature_k','h2o_vmr'])
            grid=SpectralGrid(nu,2.5)
            actual=sum(_mt_ckd_coefficients(grid,p,t,vmr))
            reference=d['self_cm2']+d['foreign_cm2']
            label=f'{t:.0f} K, {p:.0f} mbar, H2O={vmr:g}'
            axes[0,0].semilogy(nu,actual,color=color,label=label)
            axes[0,0].plot(nu[::40],reference[::40],'o',mfc='none',color=color,ms=4)
            # Historical implementation at commit 6bba840, for before/after comparison.
            c=_load_coefficients(); native=c['wavenumber_cm1']
            sr,fr,ex=(np.interp(nu,native,c[n]) for n in ['self_absco_ref','for_absco_ref','self_texp'])
            rho=p/c['ref_press_mbar']*c['ref_temp_k']/t
            old=(sr*(c['ref_temp_k']/t)**ex*vmr+fr*(1-vmr))*rho*nu*np.tanh(C2_RAD*nu/(2*t))
            axes[0,1].plot(nu,100*(old/reference-1),color=color,label=label)
            axes[0,1].plot(nu,100*(actual/reference-1),'--',color=color,lw=.8)
    axes[0,0].set(title='Water continuum: model lines, AER Fortran circles',xlabel='Wavenumber (cm$^{-1}$)',ylabel='Self + foreign cross-section (cm$^2$/molecule)')
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(title='Continuum error: previous solid, corrected dashed',xlabel='Wavenumber (cm$^{-1}$)',ylabel='Difference from AER (%)')
    axes[0,1].axhline(0,color='black',lw=.5)
    w=np.array([.5,.55,1,1.5,2,2.2,2.4,2.6,2.8,3,3.5,4])
    reference=np.array([6.6451e-27,4.4995e-27,4.0030e-28,7.8549e-29,2.4796e-29,1.6927e-29,1.1947e-29,8.6714e-30,6.4453e-30,4.8900e-30,2.6386e-30,1.5463e-30])
    new=_rayleigh_cross_section_cm2(w)
    from mwir_atmos.optics.rayleigh import _refractive_index_minus_1
    n=1+_refractive_index_minus_1(w)
    old=24*np.pi**3*(n*n-1)**2/((w*1e-4)**4*LOSCHMIDT_CM3**2*(n*n+2)**2)*1.0455
    axes[1,0].semilogy(w,reference,'o',mfc='none',color='black',label='Tomasi et al. (2005), moist air')
    axes[1,0].semilogy(w,new,color='#3178a7',label='Corrected model, dry air')
    axes[1,0].semilogy(w,old,'--',color='#c16527',label='Previous model')
    axes[1,0].set(title='Rayleigh cross-section',xlabel='Wavelength (micrometers)',ylabel='Cross-section (cm$^2$/molecule)')
    axes[1,0].legend(fontsize=8)
    axes[1,1].plot(w,100*(old/reference-1),'o--',color='#c16527',label='Previous model')
    axes[1,1].plot(w,100*(new/reference-1),'o-',color='#3178a7',label='Corrected model')
    axes[1,1].axhspan(-.3,.3,color='#3178a7',alpha=.12,label='Comparison tolerance: +/-0.3%')
    axes[1,1].set(title='Rayleigh difference from published moist-air table',xlabel='Wavelength (micrometers)',ylabel='Difference (%)')
    axes[1,1].legend(fontsize=8)
    fig.suptitle('External checks of continuum and Rayleigh components',fontsize=15)
    dest=ROOT/'docs/physics_reference_checks.png';dest.parent.mkdir(exist_ok=True)
    fig.savefig(dest,dpi=160);plt.close(fig)
    print(dest)

if __name__=='__main__':
    main()
