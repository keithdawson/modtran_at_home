! Calls the original EXABIN, AEREXT and AERPRF routines (single precision).
! Compile with the unmodified routines extracted by extract_lowtran_aerosols.py.
program check_aerosol
  common /CARD2/ IHAZE,ISEASN,IVULCN,ICSTL,ICLD,IVSA,VIS,WSS,WHH,RAINRT
  common /CARD2D/ IREG(4),ALTB(4),IREGC(4)
  common RELHUM(34),HSTOR(34),ICH(4),VH(17),TX(63),W(63)
  common WPATH(68,63),TBBY(68)
  common ABSC(5,47),EXTC(5,47),ASYM(5,47),VX0(47),AWCCON(5)
  common /AER/ EXTV(5),ABSV(5),ASYV(5)
  integer, parameter :: codes(4) = [1,5,4,6]
  real, parameter :: rhs(7) = [0.,35.,70.,75.,80.,90.,99.]
  real, parameter :: extras(8) = [.4,1.2,3.2,3.8,4.0,4.7,5.3,13.]
  real, parameter :: visibilities(7) = [2.,5.,10.,17.,23.,35.,50.]
  real :: wavelengths(48), haze
  integer :: it, ir, il, iz, label
  IREG=0
  IREGC=0
  ICLD=0
  IVULCN=0
  W=0
  ICH=[1,6,11,20]
  open(10,file='lowtran_spectra.csv',status='replace')
  write(10,'(A)') 'code,rh_pct,wavelength_um,region,extinction,absorption,g'
  do it=1,4
    IHAZE=codes(it)
    ICH(1)=IHAZE
    do ir=1,size(rhs)
      W(15)=rhs(ir)
      call EXABIN
      wavelengths(:40)=VX0(:40)
      wavelengths(41:)=extras
      do il=1,size(wavelengths)
        call AEREXT(1.e4/wavelengths(il))
        do iz=1,4
          write(10,'(I0,2(",",ES16.8),",",I0,3(",",ES16.8))') &
            IHAZE,W(15),wavelengths(il),iz,EXTV(iz),ABSV(iz),ASYV(iz)
        end do
      end do
    end do
  end do
  close(10)
  open(11,file='lowtran_profiles.csv',status='replace')
  write(11,'(A)') 'season,visibility_km,node,extinction_per_km'
  do ISEASN=1,2
    do it=1,size(visibilities)
      VIS=visibilities(it)
      do iz=1,33
        call AERPRF(iz,VIS,haze,1,ISEASN,0,label)
        write(11,'(I0,",",ES16.8,",",I0,",",ES16.8)') ISEASN,VIS,iz,haze
      end do
    end do
  end do
  close(11)
end program

! Link guard: AEREXT references microwave GAMFOG even when that branch is
! outside the 0.2-40 um validation domain. Fail if it is ever reached.
real function GAMFOG(MR,V,TAER,AWCCON)
  integer MR
  real V,TAER,AWCCON
  stop 'Microwave branch is outside this reference test'
end function
