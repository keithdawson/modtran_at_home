! Reference driver for unmodified AER MT_CKD_H2O 4.3 source.
! All coefficients include the radiation term, units cm^2/molecule.
program reference_cases
  use mt_ckd_h2o
  implicit none
  real, allocatable :: self(:), foreign(:)
  real :: pressure(5), temperature(5), vmr(5), step
  double precision :: first, last
  integer :: k, i, n
  logical :: radiation
  pressure = [1013.0,250.0,600.0,1013.0,900.0]
  temperature = [300.0,230.0,260.0,296.0,320.0]
  vmr = [0.00990098,0.0001,0.005,0.02,0.04]
  radiation = .true.
  open(20,file='mt_ckd_4_3_fortran.csv',status='replace')
  write(20,'(a)') 'case,pressure_mbar,temperature_k,h2o_vmr,wavenumber_cm1,self_cm2,foreign_cm2'
  do k=1,5
    if (k==1) then
      first=497.0d0
      last=603.0d0
      step=1.0
    else
      first=1800.25d0
      last=4300.25d0
      step=2.5
    endif
    n=nint((last-first)/step)+1
    allocate(self(n),foreign(n))
    self=0.0
    foreign=0.0
    call mt_ckd_h2o_absco(pressure(k),temperature(k),vmr(k),first,last,step,self,foreign,'0',radiation)
    do i=1,n
      write(20,'(i0,6(",",es25.17))') k,pressure(k),temperature(k),vmr(k),first+(i-1)*step,self(i),foreign(i)
    enddo
    deallocate(self,foreign)
  enddo
  close(20)
end program reference_cases
