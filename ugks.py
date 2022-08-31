from math import sqrt
import taichi as ti
import numpy as np
import struct
import sys

ti.init(arch=ti.cpu, default_fp=ti.f32, device_memory_GB=4)

SMV = 1.0E-16  # small value to avoid 0/0
  
iuh = 28       # the mesh size along the u direction
ivh = 28       # the mesh size along the v direction
iwh = 28       # the mesh size along the w direction
NXM = 32       # the mesh size along the x direction
NYM = 32       # the mesh size along the y direction
NZM = 32       # the mesh size along the z direction
i_o_t = 30     # period of computation

NM = max(NXM, max(NYM, NZM)) - 1
NV = 5         # The number of conserved quantities
NPAR = 5       # The number of conserved quantities

PI = 3.14159265358979
gam_mono = 5.0 / 3.0    # the ratio of specific heats
prantle = 2.0 /3.0
omega_ref = 0.81
UP = 1.0

ZLREF = 1.0             # the reference length
XLREF = ZLREF
YLREF = ZLREF
TTI = 1.0               # Initial dimensionless temperature
DDI_ref = 1.0           # Initial dimensionless density
alf_ref = 1.0
k00 = 4.0
AAAA = 64.0 / 3.0 / np.sqrt(2.0 * PI) * 0.5 * 0.8 ** 2 / (k00 ** 5)
sound_c = 1.0           # The initial dimensionless speed of sound
R_argon = sound_c ** 2/(gam_mono * TTI)
k_ref = 3.0 * AAAA / 64.0 * np.sqrt(2 * PI) * k00 ** 5 
q_ref = np.sqrt(2.0 * k_ref)
u_ref = q_ref / np.sqrt(3.0)
lamda11 = 2.0 / k00
VISCO = (DDI_ref * u_ref * lamda11) / 30.0     # Initial dimensionless viscosity

Mach = q_ref / sound_c                         # Mach number
Kn = VISCO / (5.0 * (alf_ref + 1.0)*(alf_ref + 2.0)/((7.0 - 2.0 * omega_ref) * (4.0 * alf_ref * (5.0 - 2.0 * omega_ref))) * DDI_ref * np.sqrt(2.0 * PI * R_argon * TTI) )/(2.0 * PI) 
Re_lamda = (DDI_ref * u_ref * lamda11) / VISCO # Reynolds number
tau_t = lamda11 / u_ref /2.0 * np.sqrt(2.0 * PI) 
tau_t_i = tau_t    # 2.0*sqrt(3.0d0)/(k00*Mach) # Large vortex turnover time

dx = ti.field(float, shape=())                 

umid = 0.0       
vmid = 0.0
wmid = 0.0
ALFA = 0.0   # Angle of attack

CK = ti.field(float, shape=())
outputvel_t = ti.field(float, shape=(i_o_t))
         
XYZ = ti.Vector.field(3, float, shape=(NXM+5, NYM+5, NZM+5))  # Coordinates of mesh in physical space
XYZT = ti.Vector.field(3, float, shape=(NXM+5, NYM+5, NZM+5))

DTL = ti.field(float, shape=(NXM+4, NYM+4, NZM+4))            # time step
CELL = ti.Vector.field(3, float, shape=(NXM+5, NYM+5, NZM+5)) # Average cell length
DIJK = ti.Matrix.field(3, 3, float, shape=(3, NXM+4, NYM+4, NZM+4)) # matrix of coordinate transformation 
AREA = ti.Vector.field(3, float, shape=(NXM+4, NYM+4, NZM+4)) # area
VOL = ti.field(float, shape=(NXM+4, NYM+4, NZM+4))            # volume
VOLMIN = ti.field(float, shape=())              
uvw0 = ti.Vector.field(3, float, shape=(NXM+4, NYM+4, NZM+4)) # Initial random velocity
kine_ene = ti.field(float, shape=())                          # The turbulent kinetic energy
kine_dis = ti.field(float, shape=())                          # Dissipation rate of turbulent kinetic energy

dis_u = ti.field(float, shape=(iuh, ivh, iwh))                # coordinates in velocity space along u
dis_v = ti.field(float, shape=(iuh, ivh, iwh))                # coordinates in velocity space along v
dis_w = ti.field(float, shape=(iuh, ivh, iwh))                # coordinates in velocity space along w
weight = ti.field(float, shape=(iuh, ivh, iwh))               # The integral weight
ctr_w = ti.Vector.field(5, float, shape=(NXM+4, NYM+4, NZM+4))# Macroscopic conserved quantity
af_ctr_w = ti.Vector.field(5, float, shape=(NXM+5, NYM+5, NZM+5))
d_ctr_w = ti.Vector.field(5, float, shape=(NXM+4, NYM+4, NZM+4))
ctr_w_old = ti.Vector.field(5, float, shape=(NXM+4, NYM+4, NZM+4))    # The flux of the macroscopic conserved quantity
dis_f_h = ti.field(float, shape=(iuh, ivh, iwh, NXM+4, NYM+4, NZM+4)) # Distribution function
af_dis_f_h = ti.field(float, shape=(iuh, ivh, iwh, NXM+5, NYM+5, NZM+5)) # The flux of the distribution function
d_dis_f_h = ti.field(float, shape=(iuh, ivh, iwh, NXM+4, NYM+4, NZM+4)) # The flux of the distribution function

NMOM = 6 
NDIM = 3

EPSL = 1.0E-7 * DDI_ref
UP_f = 1.0

PRAN_BGK     = 0
PRAN_SHAKHOV = 1
MPRAN = PRAN_SHAKHOV

#--------------------------------------------------
# module intef
#--------------------------------------------------
IBOUND = ti.field(float, shape=())
IC = ti.field(float, shape=(13, 27))

#--------------------------------------------------
# module time
#--------------------------------------------------
CFL = ti.field(float, shape=())
umax = ti.field(float, shape=())
vmax = ti.field(float, shape=())
wmax = ti.field(float, shape=())
DT_MIN = ti.field(float, shape=())
DT_MAX = ti.field(float, shape=())

# file
RSAV_MPI = r"rsave.dat"
RSAV1_MPI = r"rsave1.dat"
GRD_MPI = r"grid.dat"
OUT_MPI = r"out.dat"
CON_MPI = r"cont.dat"
vel_mpi = r"veli.dat"

#--------------------------------------------------
#SUBROUTINE discrete_maxwell : equilibrium state
#--------------------------------------------------
@ti.func
def discrete_maxwell(vn1, vt1, vw1, prim):
    h = prim[0] * ((prim[4] / PI) ** 1.5) * ti.exp(-prim[4] * ((vn1 - prim[1]) ** 2 + (vt1 - prim[2]) ** 2 + (vw1 - prim[3]) ** 2))
    return h

#--------------------------------------------------
#get_primary : density,velocity,lamda
#--------------------------------------------------
@ti.func
def get_primary(w, gam):
    prim = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
    prim[0] = w[0]
    prim[1] = w[1] / w[0]
    prim[2] = w[2] / w[0]
    prim[3] = w[3] / w[0]
    prim[4] = 0.5 * w[0] / (gam - 1.0) / (w[4] - 0.5 * (w[1] ** 2 + w[2] ** 2 + w[3] ** 2) / w[0])
    return prim

#--------------------------------------------------
# get heat flux
# param[in] h             :distribution function
# param[in] vn1,vt1,vw1   :normal and tangential velocity
# param[in] prim          :primary variables
# return get_heat_flux    :heat flux in normal and tangential direction
#--------------------------------------------------
@ti.func
def get_heat_flux(h,vn1,vt1,vw1,prim):
    heat_flux = ti.Vector([0.0, 0.0, 0.0])
    heat_flux[0] = 0.5 * (ti.sum(weight * (vn1 - prim[1]) * ((vn1 - prim[1]) ** 2 + (vt1 - prim[2]) ** 2 + (vw1 - prim[3]) ** 2) * h)) 
    heat_flux[1] = 0.5 * (ti.sum(weight * (vt1 - prim[2]) * ((vn1 - prim[1]) ** 2 + (vt1 - prim[2]) ** 2 + (vw1 - prim[3]) ** 2) * h))
    heat_flux[2] = 0.5 * (ti.sum(weight * (vw1 - prim[3]) * ((vn1 - prim[1]) ** 2 + (vt1 - prim[2]) ** 2 + (vw1 - prim[3]) ** 2) * h))
    return heat_flux

#--------------------------------------------------
# get stress
# param[in] h             :distribution function
# param[in] vn1,vt1,vw1   :normal and tangential velocity
# param[in] prim          :primary variables
# return tauij            :stress in normal and tangential direction
#--------------------------------------------------
@ti.func
def get_tauij(h,vn1,vt1,vw1,prim):
    tauij = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pre = ti.sum(weight * ((vn1 - prim[1]) ** 2 + (vt1 - prim[2]) ** 2 + (vw1 - prim[3]) ** 2) * h) / 3.0
    tauij[0] = pre - ti.sum(weight * (vn1 - prim[1]) ** 2 * h) 
    tauij[1] = -ti.sum(weight * (vn1 - prim[1]) * (vt1 - prim[2]) * h) 
    tauij[2] = -ti.sum(weight * (vn1 - prim[1]) * (vw1 - prim[3]) * h) 
    tauij[3] = pre - ti.sum(weight * (vt1 - prim[2]) ** 2 * h) 
    tauij[4] = -ti.sum(weight * (vt1 - prim[2]) * (vw1 - prim[3]) * h) 
    tauij[5] = pre - ti.sum(weight * (vw1 - prim[3]) ** 2 * h) 
    return tauij

#--------------------------------------------------
# stress calculated by NS equation
#--------------------------------------------------
@ti.func
def get_tauij_ns(i,j,k):
    get_tauij_ns = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) #heat flux in normal and tangential direction
    t1 = ti.Vector([0.0, 0.0, 0.0])
    t3 = ti.Vector([0.0, 0.0, 0.0])
    t4 = ti.Vector([0.0, 0.0, 0.0])
    t5 = ti.Vector([0.0, 0.0, 0.0])
    t6 = ti.Vector([0.0, 0.0, 0.0])
    t7 = ti.Vector([0.0, 0.0, 0.0])
    prim = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])

    prim = get_primary(ctr_w[i,j,k],gam_mono)
    miu = VISCK(1.0/(2.0*R_argon*prim[4]))
    t2 = ti.Vector([prim[1], prim[2], prim[3]])
    t1 = uvw0[i - 1, j, k]
    t3 = uvw0[i + 1, j, k]
    t4 = uvw0[i, j - 1, k]
    t5 = uvw0[i, j + 1, k]
    t6 = uvw0[i, j, k - 1]
    t7 = uvw0[i, j, k + 1]

    dudx=(t3[0]-t1[0])/(2.0*dx)
    dudy=(t5[0]-t4[0])/(2.0*dx)
    dudz=(t7[0]-t6[0])/(2.0*dx)

    dvdx=(t3[1]-t1[1])/(2.0*dx)
    dvdy=(t5[1]-t4[1])/(2.0*dx)
    dvdz=(t7[1]-t6[1])/(2.0*dx)

    dwdx=(t3[2]-t1[2])/(2.0*dx)
    dwdy=(t5[2]-t4[2])/(2.0*dx)
    dwdz=(t7[2]-t6[2])/(2.0*dx)

    div=dudx+dvdy+dwdz

    get_tauij_ns[0]=miu*(dudx+dudx-2.0/3.0*div)
    get_tauij_ns[1]=miu*(dudy+dvdx)
    get_tauij_ns[2]=miu*(dudz+dwdx)
    get_tauij_ns[3]=miu*(dvdy+dvdy-2.0/3.0*div)
    get_tauij_ns[4]=miu*(dvdz+dwdy)
    get_tauij_ns[5]=miu*(dwdz+dwdz-2.0/3.0*div)
    
    return get_tauij_ns

@ti.func
def discrete_ce(H, vn1, vt1, vw1, prim, i, j, k):
    tauij = -get_tauij_ns(i, j, k)
    H_ce = 2.0*prim[4]**2/prim[0]*((vn1-prim[1])*(vn1-prim[1])*tauij[0]+(vn1-prim[1])*(vt1-prim[2])*tauij[1]+(vn1-prim[1])*(vw1-prim[3])*tauij[2] +(vt1-prim[2])*(vt1-prim[2])*tauij[3]
    +(vn1-prim[1])*(vt1-prim[2])*tauij[1]+(vt1-prim[2])*(vw1-prim[3])*tauij[4]+(vw1-prim[3])*(vw1-prim[3])*tauij[5]+(vn1-prim[1])*(vw1-prim[3])*tauij[2]+(vt1-prim[2])*(vw1-prim[3])*tauij[4])*H   
    return H_ce                          

#--------------------------------------------------
# calculate the Shakhov part H^+
# param[in]  H             :Maxwellian distribution function
# param[in]  vn1,vt1,vw1   :normal and tangential velocity
# param[in]  qf            :heat flux
# param[in]  prim          :primary variables
# param[out] H_plus        :Shakhov part
#--------------------------------------------------
@ti.func
def shakhov_part(H,vn1,vt1,vw1,qf,prim):
    H_plus = 0.8 * (1.0 - prantle) * prim(5) ** 2 / prim(1) * ((vn1 - prim(2)) * qf(1) + (vt1 - prim(3)) * qf(2)+(vw1 - prim(4)) * qf(3)) * (2 * prim(5) * ((vn1 - prim(2)) ** 2 + (vt1 - prim(3)) ** 2 + (vw1 - prim(4)) ** 2) + CK[None] - 5.0) * H
    return H_plus

#--------------------------------------------------
# get lambda
#--------------------------------------------------
@ti.func
def EIG(DN):
    SUM = DN[1] * DN[1] + DN[2] * DN[2] + DN[3] * DN[3] 
    EIG0 = 0.25 * (CK[None] + NDIM) * DN[0] / (DN[4] - 0.5 * SUM / DN[0])
    return EIG0

#--------------------------------------------------
# get collision time tau
#--------------------------------------------------
@ti.func
def COLLISION_TIME_T(DN0,EIG0,DNL,EIGL,DNR,EIGR,DT):
    RRR = R_argon
    TEM = 1.0 / 2.0 / EIG0 / RRR
    VISC = VISCK(TEM)

    T1 = 2.0 * EIG0 * VISC / DN0  # miu/pressure
    T2 = DNL / EIGL
    T3 = DNR / EIGR

    T1 = T1 # + 1.0 * abs((T2 - T3)/(T2 + T3)) * DT + DT * 0.01
    return T1

#--------------------------------------------------
# get collision time tau
#--------------------------------------------------
@ti.func
def get_tau(prim,mu_ref):
    omega = omega_ref
    RRR = R_argon
    TEM = 1.0 / 2.0 / prim[4] / RRR
    VISC = VISCK(TEM)
    tau = VISC * 2.0 * prim[4] / prim[0]   #miu/pressure
    return tau
        
@ti.func
def get_ck(GAM):
    ck = (5.0 - 3.0 * GAM) / (GAM - 1.0) 
    return ck      

#----------------------------------------------------------------
# Subroutine to transform the velocities to the local coordinates
#----------------------------------------------------------------
@ti.func
def CRTT(GLOBAL, DXYZ):
    LOCAL = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
    LOCAL[0] = GLOBAL[0]
    LOCAL[1] = DXYZ[0, 0] * GLOBAL[1] + DXYZ[1, 0] * GLOBAL[2] + DXYZ[2, 0] * GLOBAL[3]
    LOCAL[2] = DXYZ[0, 1] * GLOBAL[1] + DXYZ[1, 1] * GLOBAL[2] + DXYZ[2, 1] * GLOBAL[3]
    LOCAL[3] = DXYZ[0, 2] * GLOBAL[1] + DXYZ[1, 2] * GLOBAL[2] + DXYZ[2, 2] * GLOBAL[3]
    LOCAL[4] = GLOBAL[4]
    return LOCAL

#--------------------------------------------------------------------------------------
# Subroutine to transform the flux from the local coordinates to the global coordinates
#--------------------------------------------------------------------------------------
@ti.func
def ICRTT(LOCAL1, LOCAL2, LOCAL3, DXYZ):
    GLOBAL = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
    GLOBAL[1] = DXYZ[0, 0] * LOCAL1 + DXYZ[0, 1] * LOCAL2 + DXYZ[0, 2] * LOCAL3
    GLOBAL[2] = DXYZ[1, 0] * LOCAL1 + DXYZ[1, 1] * LOCAL2 + DXYZ[1, 2] * LOCAL3
    GLOBAL[3] = DXYZ[2, 0] * LOCAL1 + DXYZ[2, 1] * LOCAL2 + DXYZ[2, 2] * LOCAL3
    return GLOBAL

#-----------------------------------------------------------------------------------
# Subroutine to calculate area and the local coordinates vectors of a cell interface
#-----------------------------------------------------------------------------------       
@ti.kernel 
def CAREA(IJKD: ti.i32, I: ti.i32, J: ti.i32, K: ti.i32, DXYZ: ti.template()):
    I0=I
    J0=J
    K0=K
    IFT=0
    if IJKD == 1:
        DX1 = XYZ[I0, J0 + 1, K0 + 1][0] - XYZ[I0, J0, K0][0]
        DY1 = XYZ[I0, J0 + 1, K0 + 1][1] - XYZ[I0, J0, K0][1]
        DZ1 = XYZ[I0, J0 + 1, K0 + 1][2] - XYZ[I0, J0, K0][2]
        DX2 = XYZ[I0, J0, K0 + 1][0] - XYZ[I0, J0 + 1, K0][0]
        DY2 = XYZ[I0, J0, K0 + 1][1] - XYZ[I0, J0 + 1, K0][1]
        DZ2 = XYZ[I0, J0, K0 + 1][2] - XYZ[I0, J0 + 1, K0][2]
    elif IJKD == 2:
        DX1 = XYZ[I0, J0, K0 + 1][0] - XYZ[I0 + 1, J0, K0][0]
        DY1 = XYZ[I0, J0, K0 + 1][1] - XYZ[I0 + 1, J0, K0][1]
        DZ1 = XYZ[I0, J0, K0 + 1][2] - XYZ[I0 + 1, J0, K0][2]
        DX2 = XYZ[I0 + 1, J0, K0 + 1][0] - XYZ[I0, J0, K0][0]
        DY2 = XYZ[I0 + 1, J0, K0 + 1][1] - XYZ[I0, J0, K0][1]
        DZ2 = XYZ[I0 + 1, J0, K0 + 1][2] - XYZ[I0, J0, K0][2]
    else:
        DX1 = XYZ[I0 + 1, J0 + 1, K0][0] - XYZ[I0, J0, K0][0]
        DY1 = XYZ[I0 + 1, J0 + 1, K0][1] - XYZ[I0, J0, K0][1]
        DZ1 = XYZ[I0 + 1, J0 + 1, K0][2] - XYZ[I0, J0, K0][2]
        DX2 = XYZ[I0, J0 + 1, K0][0] - XYZ[I0 + 1, J0, K0][0]
        DY2 = XYZ[I0, J0 + 1, K0][1] - XYZ[I0 + 1, J0, K0][1]
        DZ2 = XYZ[I0, J0 + 1, K0][2] - XYZ[I0 + 1, J0, K0][2]

    SIX = DY1 * DZ2 - DZ1 * DY2
    SIY = DZ1 * DX2 - DX1 * DZ2
    SIZ = DX1 * DY2 - DY1 * DX2

    AAA = ti.sqrt(SIX * SIX + SIY * SIY + SIZ * SIZ)
    AREA[I, J, K][IJKD - 1] = 0.5 * AAA

    EP = 1.1E-15
    if AAA <= EP:
        AA = ti.sqrt(DX1 * DX1 + DY1 * DY1 + DZ1 * DZ1)
        AB = ti.sqrt(DX2 * DX2 + DY2 * DY2 + DZ2 * DZ2)
        if AA >= EP or AB >= EP:
            if AA >= EP:
                if IJKD == 1:
                    I0 = I + 1
                    if I0 == NX + 3:
                        I0 = I - 1
                    DX3 = 0.25 * (XYZ[I0, J0, K0 + 1][0] + XYZ[I0, J0, K0][0] + XYZ[I0, J0 + 1, K0][0] + XYZ[I0, J0 + 1, K0 + 1][0]) - XYZ[I, J0, K0][0]
                    DY3 = 0.25 * (XYZ[I0, J0, K0 + 1][1] + XYZ[I0, J0, K0][1] + XYZ[I0, J0 + 1, K0][1] + XYZ[I0, J0 + 1, K0 + 1][1]) - XYZ[I, J0, K0][1]
                    DZ3 = 0.25 * (XYZ[I0, J0, K0 + 1][2] + XYZ[I0, J0, K0][2] + XYZ[I0, J0 + 1, K0][2] + XYZ[I0, J0 + 1, K0 + 1][2]) - XYZ[I, J0, K0][2]
                elif IJKD == 2:
                    J0 = J0 + 1
                    if J0 == NY + 3:
                        J0 = J - 1
                    DX3 = 0.25 * (XYZ[I0, J0, K0 + 1][0] + XYZ[I0, J0, K0][0] + XYZ[I0 + 1, J0, K0][0] + XYZ[I0 + 1, J0, K0 + 1][0]) - XYZ[I0 + 1, J, K0][0]
                    DY3 = 0.25 * (XYZ[I0, J0, K0 + 1][1] + XYZ[I0, J0, K0][1] + XYZ[I0 + 1, J0, K0][1] + XYZ[I0 + 1, J0, K0 + 1][1]) - XYZ[I0 + 1, J, K0][1]
                    DZ3 = 0.25 * (XYZ[I0, J0, K0 + 1][2] + XYZ[I0, J0, K0][2] + XYZ[I0 + 1, J0, K0][2] + XYZ[I0 + 1, J0, K0 + 1][2]) - XYZ[I0 + 1, J, K0][2]
                else:
                    K0 = K0 + 1
                    if K0 == NZ + 3:
                        K0 = K - 1
                    DX3 = 0.25 * (XYZ[I0 + 1, J0, K0][0] + XYZ[I0, J0, K0][0] + XYZ[I0, J0 + 1, K0][0] + XYZ[I0 + 1, J0 + 1, K0][0]) - XYZ[I0, J0, K][0]
                    DY3 = 0.25 * (XYZ[I0 + 1, J0, K0][1] + XYZ[I0, J0, K0][1] + XYZ[I0, J0 + 1, K0][1] + XYZ[I0 + 1, J0 + 1, K0][1]) - XYZ[I0, J0, K][1]
                    DZ3 = 0.25 * (XYZ[I0 + 1, J0, K0][2] + XYZ[I0, J0, K0][2] + XYZ[I0, J0 + 1, K0][2] + XYZ[I0 + 1, J0 + 1, K0][2]) - XYZ[I0, J0, K][2]
            else: 
                if IJKD == 1:
                    I0 = I + 1
                    if I0 == NX + 3:
                        I0 = I - 1
                    DX3 = 0.25 * (XYZ[I0, J0, K0 + 1][0] + XYZ[I0, J0, K0][0] + XYZ[I0, J0 + 1, K0][0] + XYZ[I0, J0 + 1, K0 + 1][0]) - XYZ[I, J0 + 1, K0][0]
                    DY3 = 0.25 * (XYZ[I0, J0, K0 + 1][1] + XYZ[I0, J0, K0][1] + XYZ[I0, J0 + 1, K0][1] + XYZ[I0, J0 + 1, K0 + 1][1]) - XYZ[I, J0 + 1, K0][1]
                    DZ3 = 0.25 * (XYZ[I0, J0, K0 + 1][2] + XYZ[I0, J0, K0][2] + XYZ[I0, J0 + 1, K0][2] + XYZ[I0, J0 + 1, K0 + 1][2]) - XYZ[I, J0 + 1, K0][2]
                elif IJKD == 2:
                    J0 = J0 + 1
                    if J0 == NY + 3:
                        J0 = J - 1
                    DX3 = 0.25 * (XYZ[I0, J0, K0 + 1][0] + XYZ[I0, J0, K0][0] + XYZ[I0 + 1, J0, K0][0] + XYZ[I0 + 1, J0, K0 + 1][0]) - XYZ[I0, J, K0][0]
                    DY3 = 0.25 * (XYZ[I0, J0, K0 + 1][1] + XYZ[I0, J0, K0][1] + XYZ[I0 + 1, J0, K0][1] + XYZ[I0 + 1, J0, K0 + 1][1]) - XYZ[I0, J, K0][1]
                    DZ3 = 0.25 * (XYZ[I0, J0, K0 + 1][2] + XYZ[I0, J0, K0][2] + XYZ[I0 + 1, J0, K0][2] + XYZ[I0 + 1, J0, K0 + 1][2]) - XYZ[I0, J, K0][2]
                else:
                    K0 = K0 + 1
                    if K0 == NZ + 3:
                        K0 = K - 1
                    DX3 = 0.25 * (XYZ[I0 + 1, J0, K0][0] + XYZ[I0, J0, K0][0] + XYZ[I0, J0 + 1, K0][0] + XYZ[I0 + 1, J0 + 1, K0][0]) - XYZ[I0 + 1, J0, K][0]
                    DY3 = 0.25 * (XYZ[I0 + 1, J0, K0][1] + XYZ[I0, J0, K0][1] + XYZ[I0, J0 + 1, K0][1] + XYZ[I0 + 1, J0 + 1, K0][1]) - XYZ[I0 + 1, J0, K][1]
                    DZ3 = 0.25 * (XYZ[I0 + 1, J0, K0][2] + XYZ[I0, J0, K0][2] + XYZ[I0, J0 + 1, K0][2] + XYZ[I0 + 1, J0 + 1, K0][2]) - XYZ[I0 + 1, J0, K][2]

                DX1 = -DX2
                DY1 = -DY2
                DZ1 = -DZ2
                AA = AB  

            if I0 == I - 1 or J0 == J - 1 or K0 == K - 1:
                DX3 = -DX3
                DY3 = -DY3
                DZ3 = -DZ3

            SIX = DY3 * DZ1 - DZ3 * DY1
            SIY = DZ3 * DX1 - DX3 * DZ1
            SIZ = DX3 * DY1 - DY3 * DX1

            AAA = ti.sqrt(SIX * SIX + SIY * SIY + SIZ * SIZ)
            AAA = 1.0 / AAA
            SIX = SIX * AAA
            SIY = SIY * AAA
            SIZ = SIZ * AAA

            AA = 1.0 / AA
            DXT = DX1 * AA
            DYT = DY1 * AA
            DZT = DZ1 * AA

            DXYZ[0, 2] = SIX
            DXYZ[1, 2] = SIY
            DXYZ[2, 2] = SIZ

            DXYZ[0, 1] = DXT
            DXYZ[1, 1] = DYT
            DXYZ[2, 1] = DZT

            DXYZ[0, 0] = SIY * DZT - SIZ * DYT
            DXYZ[1, 0] = SIZ * DXT - SIX * DZT
            DXYZ[2, 0] = SIX * DYT - SIY * DXT

        elif IFT == 0:
            if IJKD == 1:
                I0=I+1
                if I0 == NX + 3:
                    I0=I-1
            elif IJKD == 2:
                J0=J0+1
                if J0 == NY+3:
                    J0 = J-1
            else:
                K0=K0+1
                if K0 == NZ+3:
                    K0=K-1
            IFT=1

            if IJKD == 1:
                DX1 = XYZ[I0, J0 + 1, K0 + 1][0] - XYZ[I0, J0, K0][0]
                DY1 = XYZ[I0, J0 + 1, K0 + 1][1] - XYZ[I0, J0, K0][1]
                DZ1 = XYZ[I0, J0 + 1, K0 + 1][2] - XYZ[I0, J0, K0][2]
                DX2 = XYZ[I0, J0, K0 + 1][0] - XYZ[I0, J0 + 1, K0][0]
                DY2 = XYZ[I0, J0, K0 + 1][1] - XYZ[I0, J0 + 1, K0][1]
                DZ2 = XYZ[I0, J0, K0 + 1][2] - XYZ[I0, J0 + 1, K0][2]
            elif IJKD == 2:
                DX1 = XYZ[I0, J0, K0 + 1][0] - XYZ[I0 + 1, J0, K0][0]
                DY1 = XYZ[I0, J0, K0 + 1][1] - XYZ[I0 + 1, J0, K0][1]
                DZ1 = XYZ[I0, J0, K0 + 1][2] - XYZ[I0 + 1, J0, K0][2]
                DX2 = XYZ[I0 + 1, J0, K0 + 1][0] - XYZ[I0, J0, K0][0]
                DY2 = XYZ[I0 + 1, J0, K0 + 1][1] - XYZ[I0, J0, K0][1]
                DZ2 = XYZ[I0 + 1, J0, K0 + 1][2] - XYZ[I0, J0, K0][2]
            else:
                DX1 = XYZ[I0 + 1, J0 + 1, K0][0] - XYZ[I0, J0, K0][0]
                DY1 = XYZ[I0 + 1, J0 + 1, K0][1] - XYZ[I0, J0, K0][1]
                DZ1 = XYZ[I0 + 1, J0 + 1, K0][2] - XYZ[I0, J0, K0][2]
                DX2 = XYZ[I0, J0 + 1, K0][0] - XYZ[I0 + 1, J0, K0][0]
                DY2 = XYZ[I0, J0 + 1, K0][1] - XYZ[I0 + 1, J0, K0][1]
                DZ2 = XYZ[I0, J0 + 1, K0][2] - XYZ[I0 + 1, J0, K0][2]

            SIX = DY1 * DZ2 - DZ1 * DY2
            SIY = DZ1 * DX2 - DX1 * DZ2
            SIZ = DX1 * DY2 - DY1 * DX2

            AAA = ti.sqrt(SIX * SIX + SIY * SIY + SIZ * SIZ)
            AREA[I, J, K][IJKD - 1] = 0.5 * AAA

            AAA = 1.0 / AAA
            SIX = SIX * AAA
            SIY = SIY * AAA
            SIZ = SIZ * AAA

            DXYZ[0, 0] = SIX
            DXYZ[1, 0] = SIY
            DXYZ[2, 0] = SIZ

            DXT = 0.5 * (DX1 + DX2)
            DYT = 0.5 * (DY1 + DY2)
            DZT = 0.5 * (DZ1 + DZ2)
            AAA = ti.sqrt(DXT * DXT + DYT * DYT + DZT * DZT)
            AAA = 1.0 / AAA

            DXT = DXT * AAA
            DYT = DYT * AAA
            DZT = DZT * AAA

            DXYZ[0, 2] = DXT
            DXYZ[1, 2] = DYT
            DXYZ[2, 2] = DZT

            DXYZ[0, 1] = DYT * SIZ - DZT * SIY
            DXYZ[1, 1] = DZT * SIX - DXT * SIZ
            DXYZ[2, 1] = DXT * SIY - DYT * SIX

            if AAA <= EP:
                out.write('Error in CAREA! IB,ID,I,J,K:',IJKD,I,J,K)
                exit

    else:
        AAA = 1.0 / AAA
        SIX = SIX * AAA
        SIY = SIY * AAA
        SIZ = SIZ * AAA

        DXYZ[0,0] = SIX
        DXYZ[1,0] = SIY
        DXYZ[2,0] = SIZ

        DXT = 0.5 * (DX1 + DX2)
        DYT = 0.5 * (DY1 + DY2)
        DZT = 0.5 * (DZ1 + DZ2)
        AAA = ti.sqrt(DXT * DXT + DYT * DYT + DZT * DZT)
        AAA = 1.0 / AAA

        DXT = DXT * AAA
        DYT = DYT * AAA
        DZT = DZT * AAA

        DXYZ[0,2] = DXT
        DXYZ[1,2] = DYT
        DXYZ[2,2] = DZT

        DXYZ[0,1] = DYT * SIZ - DZT * SIY
        DXYZ[1,1] = DZT * SIX - DXT * SIZ
        DXYZ[2,1] = DXT * SIY - DYT * SIX

#------------------------------------------------------
#-------calculate al, ar, at
#------------------------------------------------------
@ti.func
def DXE_f(EIGG,U,V,W,G):
    M = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
    T1 = CK[None] + 3.0
    T2 = U * U + V * V + W * W + 0.5 * T1 / EIGG
    R5 = 2.0 * G[4] - T2 * G[0]
    R4 = G[3] - W * G[0]
    R3 = G[2] - V * G[0]
    R2 = G[1] - U * G[0]

    T3 = R5 - 2.0 * U * R2 - 2.0 * V * R3 - 2.0 * W * R4

    M[4] = 4.0 * EIGG * EIGG * T3 / T1
    M[3] = 2.0 * EIGG * R4 - W * M[4]
    M[2] = 2.0 * EIGG * R3 - V * M[4]
    M[1] = 2.0 * EIGG * R2 - U * M[4]
    M[0] = G[0] - U * M[1] - V * M[2] - W * M[3] - 0.5 * M[4] * T2
    return M

#-------------------------------------------------------------------
#---------calculat time derivative according to the conservation law
#-------------------------------------------------------------------
@ti.func
def MUGF(GGT,ML0,MR0,MUD0,MOI0,UFF0,UFM0,UFN0,VFF0,WFF0):  
    GGT = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
    GGT[0] = -(TUVW(ML0, UFM0, VFF0, WFF0, 1, 0, 0) +TUVW(MR0, UFN0, VFF0, WFF0, 1, 0, 0) + TUVW(MUD0, UFF0, VFF0, WFF0, 0, 1, 0) +TUVW(MOI0, UFF0, VFF0, WFF0, 0, 0, 1))
    GGT[1] = -(TUVW(ML0, UFM0, VFF0, WFF0, 2, 0, 0) +TUVW(MR0, UFN0, VFF0, WFF0, 2, 0, 0) + TUVW(MUD0, UFF0, VFF0, WFF0, 1, 1, 0) +TUVW(MOI0, UFF0, VFF0, WFF0, 1, 0, 1))
    GGT[2] = -(TUVW(ML0, UFM0, VFF0, WFF0, 1, 1, 0) +TUVW(MR0, UFN0, VFF0, WFF0, 1, 1, 0) + TUVW(MUD0, UFF0, VFF0, WFF0, 0, 2, 0) +TUVW(MOI0, UFF0, VFF0, WFF0, 0, 1, 1))
    GGT[3] = -(TUVW(ML0, UFM0, VFF0, WFF0, 1, 0, 1) +TUVW(MR0, UFN0, VFF0, WFF0, 1, 0, 1) + TUVW(MUD0, UFF0, VFF0, WFF0, 0, 1, 1) +TUVW(MOI0, UFF0, VFF0, WFF0, 0, 0, 2))
    GGT[4] = -(TUV_U2_V2_W2(ML0, UFM0, VFF0, WFF0, 1, 0, 0) + TUV_U2_V2_W2(MR0, UFN0, VFF0, WFF0, 1, 0, 0) + TUV_U2_V2_W2(MUD0, UFF0, VFF0, WFF0, 0, 1, 0) + TUV_U2_V2_W2(MOI0, UFF0, VFF0, WFF0, 0, 0, 1))
    return GGT

@ti.func
def TUVW(M,U,V,W,I1,I2,I3):
    T1 = (U[I1 + 2] * V[I2] * W[I3] + U[I1] * V[I2 + 2] * W[I3] + U[I1] * V[I2] * W[I3 + 2]) * M[4] * 0.5
    UVW = M[0] * U[I1] * V[I2] * W[I3] + M[1] * U[I1 + 1] * V[I2] * W[I3] + M[2] * U[I1] * V[I2 + 1] * W[I3] + M[3] * U[I1] * V[I2] * W[I3 + 1] + T1
    return UVW

@ti.func
def TUV_U2_V2_W2(M,U,V,W,I1,I2,I3):
    T1 = TUVW(M,U,V,W,I1+2,I2,I3) + TUVW(M,U,V,W,I1,I2+2,I3) + TUVW(M,U,V,W,I1,I2,I3+2)
    UV_U2_V2_W2 = 0.5 * T1
    return UV_U2_V2_W2

#------------------------------------------------------
#-------calculate <u^n> <0 <v^n> <0 <w^n> <0
#------------------------------------------------------
@ti.func
def COEF_INFITE_HARF_LEFT(EIGG,U):
    UF = ti.types.vector(NMOM+1, ti.f32)
    UF[0] = 0.5 * DERFC(-ti.sqrt(EIGG) * U)
    UF[1] = U * UF[0] + 0.5 * ti.exp(-EIGG * U * U) / ti.sqrt(PI * EIGG)
    for I in range(2, NMOM+1):
        UF[I] = U * UF[I - 1] + 0.5 * (I - 1.0) * UF[I-2] /EIGG
    return UF

#------------------------------------------------------
#-------calculate <u^n> >0 <v^n> >0 <w^n> >0
#------------------------------------------------------
@ti.func
def COEF_INFITE_HARF_RIGHT(EIGG,U,UF,UFM):
    UFN = ti.types.vector(NMOM+1, ti.f32)
    for I in range(0, NMOM+1):
        UFN[I] = UF[I] - UFM[I]
    return UFN

#------------------------------------------------------
#-------calculate <u^n> <v^n> <w^n>
#------------------------------------------------------
@ti.func
def COEF_INFITE(EIGG,U):
    UF = ti.types.vector(NMOM+1, ti.f32)
    UF[0] = 1.0
    UF[1] = U
    for I in range(2, NMOM+1):
        UF[I] = U * UF[I-1] + 0.5 * (I-1) * UF[I-2] / EIGG
    return UF

#---------------------------------------------------------------------        
#-------convert macroscopic conserved variables to primitive variables
#--------------------------------------------------------------------- 
@ti.func
def PHYSICAL_PAR(U):
    D = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
    D[0] = U[0]
    for K in range(1, 4):
        D[K] = U[K] / D[0]
    D[4] = U[4]
    return D

#---------------------------------------------------------------------        
#-------calculate equilibrium macroscopic flux
#--------------------------------------------------------------------- 
@ti.func
def fMUVSF(DN0,RT,ML0,MR0,MUD0,MOI0,MT0,UFM0,UFN0,UFF0,VFF0,WFF0,I1,I2,I3):
    F = ti.Vector([0.0, 0.0, 0.0])
    F[0] = DN0 * UFF0[I1] * VFF0[I2] * WFF0[I3]
    F[1] = (TUVW(ML0 ,UFM0, VFF0, WFF0, I1 + 1, I2, I3) + TUVW(MR0 ,UFN0, VFF0, WFF0, I1 + 1, I2, I3) + TUVW(MUD0, UFN0, VFF0, WFF0, I1, I2 + 1, I3) + TUVW(MOI0, UFN0, VFF0, WFF0, I1, I2, I3 + 1)) * DN0
    F[2] = DN0 * TUVW(MT0, UFF0, VFF0, WFF0, I1, I2, I3)
    fMUVSF = 0.0
    for I in range(3):
        fMUVSF = fMUVSF + RT(I) * F(I)
    return fMUVSF

#---------------------------------------------------------------------        
#-------calculate time integral
#--------------------------------------------------------------------- 
@ti.func
def CPDTT(DT,TAU):
    R = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
    ET = ti.exp(-DT / TAU)
    R[0] = DT - TAU * (1.0 - ET)
    R[1] = (-DT + 2.0 * TAU * (1.0 - ET) - DT * ET) * TAU
    R[2] = 0.5 * DT * DT - TAU * DT + TAU * TAU * (1.0 - ET)
    R[3] = (1.0 - ET) * TAU
    R[4] = (DT * ET - TAU * (1.0 - ET)) * TAU
    return R

#---------------------------------------------------------------------  
#-------This function is to caculate the ERFC function
#-------where b : integral boundary
#---------------------------------------------------------------------  
@ti.func
def DERFC(XX):
    PI_derfc = 3.1415926535897932
    EP = 1.0E-15
    B = ti.abs(XX)
    if B >= 6.0:
        DERFC = 0.0
    elif B < 1.0:
        X = B
        BI = 3.0
        AI = 1.0
        AC = -1.0
        AX = X*X
        AB = AX*X/BI
        ER = X-AB
        while AB >= EP or AB < -EP:
            AI = AI+1.0
            BI = BI+2.0
            AC = -AC
            AB = AB / BI * AX * (BI - 2.0) / AI
            if AC >= 0.0:
                ER = ER + AB
            else:
                ER = ER - AB
        ER = ER * 2.0 / ti.sqrt(PI_derfc)
        DERFC = 1.0 - ER
    else:
        V = 0.5 / B / B
        XE = B * ti.exp(B * B)
        P0 = 1.0
        Q0 = 1.0
        P1 = 1.0
        ER0 = 1.0
        ER = 5.0
        N = 2
        Q1 = 1.0 + V
        while ti.abs(ER - ER0) >= EP:
            RN = N
            AN = RN * V
            PN = P1 + AN * P0
            QN = Q1 + AN * Q0
            P0 = P1
            Q0 = Q1
            P1 = PN
            Q1 = QN
            N = N + 1
            ER0 = ER
            ER = PN / QN
        DERFC = ER / XE / ti.sqrt(PI_derfc)
      
    if XX < 0.0:
        DERFC = 2.0 - DERFC
    return DERFC

#********************************************************************        
# Subroutine to calculate the timestep
#********************************************************************
@ti.kernel
def CDTL():              
    DT_MIN[None]=100.0
    DT_MAX[None]=0.0
    for I, J, K in ti.ndrange((2, NX + 2), (2, NY + 2), (2, NZ + 2)):
        DTL[I, J, K] = CFL[None] / (ti.max(umax[None], vmax[None], wmax[None]) / ti.min(CELL[I, J, K][0], CELL[I, J, K][1], CELL[I, J, K][2]))

        DT_MIN[None] = ti.atomic_min(DT_MIN[None], DTL[I, J, K])
        DT_MAX[None] = ti.atomic_max(DT_MAX[None], DTL[I, J, K])

#********************************************************************
#     calculate the kinetic viscosity
#********************************************************************
@ti.func
def VISCK(TEM):   
    TTI_ref = TTI
    VISCK = VISCO * (TEM / TTI_ref) ** omega_ref
    return VISCK

#********************************************************************
#      velocity discreting space & weigth of integration
#********************************************************************
@ti.kernel
def velocity_grids():         
    vcoords = ti.Vector([ -0.5392407922630E+01, -0.4628038787602E+01, -0.3997895360339E+01, -0.3438309154336E+01, 
    -0.2926155234545E+01, -0.2450765117455E+01, -0.2007226518418E+01, -0.1594180474269E+01, -0.1213086106429E+01, 
    -0.8681075880846E+00, -0.5662379126244E+00, -0.3172834649517E+00, -0.1331473976273E+00, -0.2574593750171E-01, 
    +0.2574593750171E-01, +0.1331473976273E+00, +0.3172834649517E+00, +0.5662379126244E+00, +0.8681075880846E+00, 
    +0.1213086106429E+01, +0.1594180474269E+01, +0.2007226518418E+01, +0.2450765117455E+01, +0.2926155234545E+01, 
    +0.3438309154336E+01, +0.3997895360339E+01, +0.4628038787602E+01, +0.5392407922630E+01 ])

    weights = ti.Vector([ +0.2070921821819E-12, +0.3391774320172E-09, +0.6744233894962E-07, +0.3916031412192E-05,
    +0.9416408715712E-04, +0.1130613659204E-02, +0.7620883072174E-02, +0.3130804321888E-01, +0.8355201801999E-01, 
    +0.1528864568113E+00, +0.2012086859914E+00, +0.1976903952423E+00, +0.1450007948865E+00, +0.6573088665062E-01,
    +0.6573088665062E-01, +0.1450007948865E+00, +0.1976903952423E+00, +0.2012086859914E+00, +0.1528864568113E+00, 
    +0.8355201801999E-01, +0.3130804321888E-01, +0.7620883072174E-02, +0.1130613659204E-02, +0.9416408715712E-04,
    +0.3916031412192E-05, +0.6744233894962E-07, +0.3391774320172E-09, +0.2070921821819E-12 ])
            
    umax[None] = umid + ti.max(vcoords)
    vmax[None] = vmid + ti.max(vcoords)
    wmax[None] = wmid + ti.max(vcoords)

    for i, j, k in ti.static(range(iuh, ivh, iwh)):            
        dis_u[i, j, k] = umid + vcoords[i]
        dis_v[i, j, k] = vmid + vcoords[j]
        dis_w[i, j, k] = wmid + vcoords[k]
        weight[i, j, k] = (weights[i] * ti.exp(vcoords[i] ** 2)) *(weights[j] * ti.exp(vcoords[j] ** 2)) * (weights[k] * ti.exp(vcoords[k] ** 2))   


#********************************************************************
#      calculate volume of cells
#********************************************************************
@ti.kernel
def CVOL():      
    VOLMIN[None] = 1.0E7
    for I, J, K in ti.ndrange((1, NX + 1), (1, NY + 1), (1, NZ + 1)):
        DX1 = XYZ[I, J + 1, K + 1][0] - XYZ[I, J, K][0]
        DY1 = XYZ[I, J + 1, K + 1][1] - XYZ[I, J, K][1]
        DZ1 = XYZ[I, J + 1, K + 1][2] - XYZ[I, J, K][2]
        DX2 = XYZ[I, J, K + 1][0] - XYZ[I, J + 1, K][0]
        DY2 = XYZ[I, J, K + 1][1] - XYZ[I, J + 1, K][1]
        DZ2 = XYZ[I, J, K + 1][2] - XYZ[I, J + 1, K][2]
        SIX = 0.5 * (DY1 * DZ2 - DZ1 * DY2)
        SIY = 0.5 * (DZ1 * DX2 - DX1 * DZ2)
        SIZ = 0.5 * (DX1 * DY2 - DY1 * DX2)

        DX1 = XYZ[I + 1, J, K + 1][0] - XYZ[I, J, K][0]
        DY1 = XYZ[I + 1, J, K + 1][1] - XYZ[I, J, K][1]
        DZ1 = XYZ[I + 1, J, K + 1][2] - XYZ[I, J, K][2]
        DX2 = XYZ[I + 1, J, K][0] - XYZ[I, J, K + 1][0]
        DY2 = XYZ[I + 1, J, K][1] - XYZ[I, J, K + 1][1]
        DZ2 = XYZ[I + 1, J, K][2] - XYZ[I, J, K + 1][2]
        SJX = 0.5 * (DY1 * DZ2 - DZ1 * DY2)
        SJY = 0.5 * (DZ1 * DX2 - DX1 * DZ2)
        SJZ = 0.5 * (DX1 * DY2 - DY1 * DX2)

        DX1 = XYZ[I + 1, J + 1, K][0] - XYZ[I, J, K][0]
        DY1 = XYZ[I + 1, J + 1, K][1] - XYZ[I, J, K][1]
        DZ1 = XYZ[I + 1, J + 1, K][2] - XYZ[I, J, K][2]
        DX2 = XYZ[I, J + 1, K][0] - XYZ[I + 1, J, K][0]
        DY2 = XYZ[I, J + 1, K][1] - XYZ[I + 1, J, K][1]
        DZ2 = XYZ[I, J + 1, K][2] - XYZ[I + 1, J, K][2]
        SKX = 0.5 * (DY1 * DZ2 - DZ1 * DY2)
        SKY = 0.5 * (DZ1 * DX2 - DX1 * DZ2)
        SKZ = 0.5 * (DX1 * DY2 - DY1 * DX2)

        SX = SIX + SJX + SKX
        SY = SIY + SJY + SKY
        SZ = SIZ + SJZ + SKZ
        XT1 = XYZ[I + 1, J + 1, K + 1][0] - XYZ[I, J, K][0]
        YT1 = XYZ[I + 1, J + 1, K + 1][1] - XYZ[I, J, K][1]
        ZT1 = XYZ[I + 1, J + 1, K + 1][2] - XYZ[I, J, K][2]
        VOL[I, J, K] = (XT1 * SX + YT1 * SY + ZT1 * SZ) / 3.0

        if VOL[I, J, K] <= 0.0:
            out.write('NEGATIVE VOLUME')
            out.write('I,J,K',I,J,K)
            exit
     
        if VOL[I, J, K] <= VOLMIN[None]:
            VOLMIN[None] = VOL[I, J, K]

#********************************************************************
#---- calculate interface length of cell
#********************************************************************
@ti.kernel
def CCELL():
    VOL_T = ti.types.matrix(3, NX + 2, NY + 2, NZ + 2, ti.f32)
    for I, J, K in ti.ndrange(NX + 5, NY + 5, NZ + 5):             
        if I != NX + 4: 
            XT1 = XYZ[I + 1, J, K][0] - XYZ[I, J, K][0]
            YT1 = XYZ[I + 1, J, K][1] - XYZ[I, J, K][1]
            ZT1 = XYZ[I + 1, J, K][2] - XYZ[I, J, K][2]
            CELL[I, J, K][0] = ti.sqrt(XT1 * XT1 + YT1 * YT1 + ZT1 * ZT1)
        if J != NY + 4:
            XT1 = XYZ[I, J + 1, K][0] - XYZ[I, J, K][0]
            YT1 = XYZ[I, J + 1, K][1] - XYZ[I, J, K][1]
            ZT1 = XYZ[I, J + 1, K][2] - XYZ[I, J, K][2]
            CELL[I, J, K][1] = ti.sqrt(XT1 * XT1 + YT1 * YT1 + ZT1 * ZT1)
        if K != NZ + 4:
            XT1 = XYZ[I, J, K + 1][0] - XYZ[I, J, K][0]
            YT1 = XYZ[I, J, K + 1][1] - XYZ[I, J, K][1]
            ZT1 = XYZ[I, J, K + 1][2] - XYZ[I, J, K][2]
            CELL[I, J, K][2] = ti.sqrt(XT1 * XT1 + YT1 * YT1 + ZT1 * ZT1)
      
    for I, J, K in ti.ndrange(NX + 4, NY + 4, NZ + 4): 
        VOL_T[I, J, K][0] = 0.25 * (CELL[I, J, K][0] + CELL[I, J + 1, K][0] + CELL[I, J + 1, K + 1][0] + CELL[I, J, K + 1][0])
        VOL_T[I, J, K][1] = 0.25 * (CELL[I, J, K][1] + CELL[I + 1, J, K][1] + CELL[I + 1, J, K + 1][1] + CELL[I, J, K + 1][1])
        VOL_T[I, J, K][2] = 0.25 * (CELL[I, J, K][2] + CELL[I, J + 1, K][2] + CELL[I + 1, J + 1, K][2] + CELL[I + 1, J, K][2])
     
    for I, J, K in ti.ndrange(NX + 4, NY + 4, NZ + 4): 
        CELL[I, J, K][0] = VOL_T[I, J, K][0]  
        CELL[I, J, K][1] = VOL_T[I, J, K][1]  
        CELL[I, J, K][2] = VOL_T[I, J, K][2]  

#********************************************************************
#---- calculate interface flux along x
#********************************************************************
@ti.kernel
def flux_x():
    for IX, IY, IZ in ti.ndrange((2, NX+3), (2, NY+2), (2, NZ+2)):
        ULC = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        URC = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN_U = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN_D = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN_O = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN_I = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        RT = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        DN0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UFF0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        VFF0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        WFF0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        ML0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        MR0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        MUD0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        MOI0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        MT0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        GGT = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        FLUX_EQ = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        FLUX_UN = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        FLUX_Q = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        Q = ti.Vector([0.0, 0.0, 0.0])
        GGL = ti.Vector([0.0, 0.0, 0.0])
        GGR = ti.Vector([0.0, 0.0, 0.0])
        DIST = ti.Vector([0.0, 0.0])

#------ time step
        DELTT = dt_all
        DLL = AREA(1, IX, IY, IZ) / DELTT

        DIST[0] = 0.5 * dx
        DIST[1] = 0.5 * dx

#-------macroscopic conserved variable of left
        ULC = ctr_w[IX - 1, IY, IZ]

#-------macroscopic conserved variable of right
        URC = ctr_w[IX, IY, IZ]     

#-------macroscopic conserved variable of up
        UN_U = 0.5*(ctr_w[IX - 1, IY + 1, IZ] + ctr_w[IX, IY + 1, IZ])

#-------macroscopic conserved variable of down
        UN_D = 0.5*(ctr_w[IX - 1, IY - 1, IZ] + ctr_w[IX, IY - 1, IZ])

#-------macroscopic conserved variable of outside        
        UN_O = 0.5*(ctr_w[IX - 1, IY, IZ + 1] + ctr_w[IX, IY, IZ + 1])

#-------macroscopic conserved variable of inside
        UN_I = 0.5*(ctr_w[IX - 1, IY, IZ - 1] + ctr_w[IX, IY, IZ - 1])

#-------macroscopic conserved variable at interface
        UN0 = 0.5*(ULC+URC)

#-------Lambda
        EIG0 = EIG(UN0)
        EIGL = EIG(ULC)
        EIGR = EIG(URC)
             
#-------collision time
        TAU = COLLISION_TIME_T(UN0[0], EIG0, ULC[0], EIGL, URC[0], EIGR, DELTT)
#-------time integral
        RT = CPDTT(DELTT, TAU)
        
#-------convert macroscopic conserved variables to primitive variables
        DN0 = PHYSICAL_PAR(UN0)
     
#-------calculate <u^n> <v^n> <w^n>
        UFF0 = COEF_INFITE(EIG0, DN0[1])
        VFF0 = COEF_INFITE(EIG0, DN0[2])
        WFF0 = COEF_INFITE(EIG0, DN0[3])
        
#-------calculate <u^n> >0, <u^n> <0
        UFM0 = COEF_INFITE_HARF_LEFT(EIG0, DN0[1])
        UFN0 = COEF_INFITE_HARF_RIGHT(EIG0, DN0[1], UFF0, UFM0)
        
#-------calculate al=ML0
        DLNG = DIST[0] * UN0[0]
        GGT = (UN0 - ULC) / DLNG
        ML0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)
 
#-------calculate ar=MR0
        DLNG = DIST[1] * UN0[0]
        GGT = (URC - UN0) / DLNG
        MR0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)
        
#-------calculate aud=MUDO for multidimensional
        DLNG = DIST[0] * UN0[0]
        GGT = (UN_U - UN_D) / (4.0 * DLNG)
        MUD0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)
        
#-------calculate aoi=MUDO for multidimensional
        DLNG = DIST[0] * UN0[0]
        GGT = (UN_O - UN_I) / (4.0 * DLNG)
        MOI0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)

#-------calculate At=MT0
        GGT = MUGF(ML0, MR0, MUD0, MOI0, UFF0, UFM0, UFN0, VFF0, WFF0)
        MT0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)

#-------Equilibrium macroscopic flux
        FLUX_EQ[0] = fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 0, 0)
        FXI = fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 2, 0, 0)
        FYI = fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 1, 0)
        FZI = fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 0, 1)
        FLUX_EQ[4] = (fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 3, 0, 0)         
        + fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 2, 0)         
        + fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 0, 2))*0.5

        FLUX_EQ[1:3] = ICRTT(FXI, FYI, FZI, DIJK[:,:,1,IX,IY,IZ])
        FLUX_EQ = FLUX_EQ * DLL

        FXI = 0.0
        FYI = 0.0
        FZI = 0.0
        FLUX_UN = 0.0
        Q = 0.0

        GAML=gam_mono
        CK[None] = get_ck(GAML)

        for iu, iv, iw in ti.static(range(iuh, ivh, iwh)):
#------------------------------------------------------------------------------
#------------------------ x ---------------------------------------------------
#------------------------------------------------------------------------------
            UN  = dis_u[iu,iv,iw]
            VNT = dis_v[iu,iv,iw]
            WT  = dis_w[iu,iv,iw]
           
            CX  = UN - DN0[1]
            CY  = VNT - DN0[2]
            CZ  = WT - DN0[3]
            UV2 = CX * CX + CY * CY + CZ * CZ

#----------g0
            G0 = ti.exp(-EIG0 * UV2 + (CK[None] + NDIM) * ti.log(EIG0 / PI) * 0.5) * DN0[0]

            if UN >= 0:
                H1 = 1.0
                H2 = 0.0
            else:
                H1 = 0.0
                H2 = 1.0

#----------------------------------------------------------
#----------Equilibrium microscopic flux
#----------------------------------------------------------
            T1 = ML0[0] + ML0[1] * UN + ML0[2] * VNT + ML0[3] * WT
            T2 = MR0[0] + MR0[1] * UN + MR0[2] * VNT + MR0[3] * WT
            T3 = MT0[0] + MT0[1] * UN + MT0[2] * VNT + MT0[3] * WT
            T4 = (UN * UN + VNT * VNT + WT * WT) * 0.5
            T5 = T1 + ML0[4] * T4
            T6 = T2 + MR0[4] * T4
            T56_1 = MUD0[0] + MUD0[1] * UN + MUD0[2] * VNT + MUD0[3] * WT + MUD0[4] * T4
            T56_2 = MOI0[0] + MOI0[1] * UN + MOI0[2] * VNT + MOI0[3] * WT + MOI0[4] * T4
            T7 = T3 + MT0[4] * T4
            af_dis_f_h[iu, iv, iw, IX, IY, IZ] = UN * G0 * (RT[0] + RT[1] * UN * (T5 * H1 + T6 * H2) + RT[1] * VNT * T56_1 + RT[1] * WT * T56_2 + RT[2] * T7)

#--------------------------------------------------------------
#----------Calculate the slope of the gas distribution function
#--------------------------------------------------------------
            GGL[0] = (dis_f_h[iu, iv, iw, IX, IY, IZ] - dis_f_h[iu, iv, iw, IX - 1, IY, IZ]) / dx
            GGL[1] = (dis_f_h[iu, iv, iw, IX - 1, IY + 1, IZ] + dis_f_h[iu, iv, iw, IX, IY + 1, IZ]  
            - dis_f_h[iu, iv, iw, IX - 1, IY - 1, IZ] - dis_f_h[iu, iv, iw, IX, IY - 1, IZ]) / (4 * dx)
            GGL[2] = (dis_f_h[iu, iv, iw, IX - 1, IY, IZ + 1] + dis_f_h[iu, iv, iw, IX, IY, IZ + 1] 
            - dis_f_h[iu, iv, iw, IX - 1, IY, IZ - 1] - dis_f_h[iu, iv, iw, IX, IY, IZ - 1]) / (4 * dx)
            GGR[0] = GGL[0]
            GGR[1] = GGL[1]
            GGR[2] = GGL[2]
            GFL = 0.5 * (dis_f_h[iu, iv, iw, IX - 1, IY, IZ] + dis_f_h[iu, iv, iw, IX, IY, IZ])
            GFR = GFL

            WXY = weight[iu,iv,iw]

#-----------------------------------------------------------------------
#----------unequilibrium microscopic flux
#-----------------------------------------------------------------------
            T11 = RT[3] * GFL + RT[4] * UN * GGL[0]
            T12 = RT[3] * GFR + RT[4] * UN * GGR[0]
            T1 = T11 * H1 + T12 * H2 + RT[4] * 0.5 * (VNT * (GGL[1] + GGR[1]) + WT * (GGL[2] + GGR[2]))
            af_dis_f_h[iu, iv, iw, IX, IY, IZ] = (af_dis_f_h[iu, iv, iw, IX, IY, IZ] + UN * T1) * DLL
           
#-----------------------------------------------------------------------
#----------unquilibrium macroscopic flux
#-----------------------------------------------------------------------
            FLUX_UN[0] = FLUX_UN[0] + WXY * UN * T1
            FXI = FXI + WXY * UN * UN * T1
            FYI = FYI + WXY * UN * VNT * T1
            FZI = FZI + WXY * UN * WT * T1
            FLUX_UN[4] = FLUX_UN[4] + 0.5 * WXY * UN * T1 * (UN * UN + VNT * VNT + WT * WT)
          
#---------heat flux = FLUX(6:8)       
            if MPRAN == PRAN_SHAKHOV and EIG0 > EPSL:
                if UN >= 0:
                    QGH =UV2*GFL*WXY
                else:
                    QGH =UV2*GFR*WXY
                Q[0] = Q[0] + 0.5 * CX * QGH
                Q[1] = Q[1] + 0.5 * CY * QGH
                Q[2] = Q[2] + 0.5 * CZ * QGH

        FLUX_UN[1:3] = ICRTT(FXI,FYI,FZI,DIJK[:,:,1,IX,IY,IZ]) 
        FLUX_UN = FLUX_UN * DLL

#--------heat flux 
        TX1 = Q[0]
        TX2 = Q[1]
        TX3 = Q[2]
 
        TX0 = 0.8 * (1.0 - prantle) * EIG0 * EIG0 / UN0[0]

        FXI=0.0
        FYI=0.0
        FZI=0.0
        FLUX_Q=0.0
         
        for iu, iv, iw in ti.static(range(iuh, ivh, iwh)):
#----------------------------------------------------------------
#------------------------- x ------------------------------------
#----------------------------------------------------------------
            UN  = dis_u[iu, iv, iw]
            VNT = dis_v[iu, iv, iw]
            WT  = dis_w[iu, iv, iw]
         
            CX  = UN - DN0[1]
            CY  = VNT - DN0[2]
            CZ  = WT - DN0[3]
            UV2 = CX * CX + CY * CY + CZ * CZ
            CQ  = CX * TX1 + CY * TX2 + CZ * TX3
      
            G0 = DN0[0] * ti.exp(-EIG0 * UV2 + (CK[None] + NDIM) * ti.log(EIG0 / PI) * 0.5)

            T4 = 2.0 * EIG0 * UV2
            GG = (T4 + CK[None] - 5.0) * TX0 * CQ * G0 * RT[0]
        
            af_dis_f_h[iu, iv, iw, IX, IY, IZ] = af_dis_f_h[iu, iv, iw, IX, IY, IZ] + UN * GG * DLL
   
            WXY = weight[iu, iv, iw]
            
            FLUX_Q[0] = FLUX_Q[0] + WXY * UN * GG
            FXI = FXI + WXY * UN * UN * GG
            FYI = FYI + WXY * UN * VNT * GG
            FZI = FZI + WXY * UN * WT * GG
            FLUX_Q[4] = FLUX_Q[4] + 0.5 * WXY * UN * GG * (UN * UN + VNT * VNT + WT * WT)
            
        FLUX_Q[1:3] = ICRTT(FXI,FYI,FZI,DIJK[:,:,1,IX,IY,IZ])
            
        if EIG0 <= EPSL:
            FLUX_Q = 0.0
        
        FLUX_Q = FLUX_Q * DLL
        af_ctr_w[IX, IY, IZ] = FLUX_EQ + FLUX_UN + FLUX_Q

#********************************************************************
#---- calculate interface flux along y
#********************************************************************
@ti.kernel
def flux_y():
    for IX, IY, IZ in ti.ndrange((2, NX+2), (2, NY+3), (2, NZ+2)):
        ULC = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        URC = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN_U = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN_D = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN_O = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN_I = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        RT = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        DN0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UFF0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        VFF0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        WFF0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        ML0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        MR0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        MUD0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        MOI0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        MT0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        GGT = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        FLUX_EQ = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        FLUX_UN = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        FLUX_Q = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        Q = ti.Vector([0.0, 0.0, 0.0])
        GGL = ti.Vector([0.0, 0.0, 0.0])
        GGR = ti.Vector([0.0, 0.0, 0.0])
        DIST = ti.Vector([0.0, 0.0])

#------ time step
        DELTT = dt_all
        DLL = AREA(2, IX, IY, IZ) / DELTT

        DIST[0] = 0.5 * dx
        DIST[1] = 0.5 * dx

#-------macroscopic conserved variable of left
        ULC = ctr_w[IX, IY - 1, IZ]
        ULC[1], ULC[2] = ULC[2], -ULC[1]

#-------macroscopic conserved variable of right
        URC = ctr_w[IX, IY, IZ] 
        URC[1], URC[2] = URC[2], -URC[1]    

#-------macroscopic conserved variable of up
        UN_U = 0.5*(ctr_w[IX - 1, IY - 1, IZ] + ctr_w[IX - 1, IY, IZ])
        UN_U[1], UN_U[2] = UN_U[2], -UN_U[1]

#-------macroscopic conserved variable of down
        UN_D = 0.5*(ctr_w[IX + 1, IY - 1, IZ] + ctr_w[IX + 1, IY, IZ])
        UN_D[1], UN_D[2] = UN_D[2], -UN_D[1]

#-------macroscopic conserved variable of outside        
        UN_O = 0.5*(ctr_w[IX, IY - 1, IZ + 1] + ctr_w[IX, IY, IZ + 1])
        UN_O[1], UN_O[2] = UN_O[2], -UN_O[1]

#-------macroscopic conserved variable of inside
        UN_I = 0.5*(ctr_w[IX, IY - 1, IZ - 1] + ctr_w[IX, IY, IZ - 1])
        UN_I[1], UN_I[2] = UN_I[2], -UN_I[1]

#-------macroscopic conserved variable at interface
        UN0 = 0.5*(ULC+URC)

#-------Lambda
        EIG0 = EIG(UN0)
        EIGL = EIG(ULC)
        EIGR = EIG(URC)
             
#-------collision time
        TAU = COLLISION_TIME_T(UN0[0], EIG0, ULC[0], EIGL, URC[0], EIGR, DELTT)
#-------time integral
        RT = CPDTT(DELTT, TAU)
        
#-------convert macroscopic conserved variables to primitive variables
        DN0 = PHYSICAL_PAR(UN0)
     
#-------calculate <u^n> <v^n> <w^n>
        UFF0 = COEF_INFITE(EIG0, DN0[1])
        VFF0 = COEF_INFITE(EIG0, DN0[2])
        WFF0 = COEF_INFITE(EIG0, DN0[3])
        
#-------calculate <u^n> >0, <u^n> <0
        UFM0 = COEF_INFITE_HARF_LEFT(EIG0, DN0[1])
        UFN0 = COEF_INFITE_HARF_RIGHT(EIG0, DN0[1], UFF0, UFM0)
        
#-------calculate al=ML0
        DLNG = DIST[0] * UN0[0]
        GGT = (UN0 - ULC) / DLNG
        ML0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)
 
#-------calculate ar=MR0
        DLNG = DIST[1] * UN0[0]
        GGT = (URC - UN0) / DLNG
        MR0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)
        
#-------calculate aud=MUDO for multidimensional
        DLNG = DIST[0] * UN0[0]
        GGT = (UN_U - UN_D) / (4.0 * DLNG)
        MUD0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)
        
#-------calculate aoi=MUDO for multidimensional
        DLNG = DIST[0] * UN0[0]
        GGT = (UN_O - UN_I) / (4.0 * DLNG)
        MOI0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)

#-------calculate At=MT0
        GGT = MUGF(ML0, MR0, MUD0, MOI0, UFF0, UFM0, UFN0, VFF0, WFF0)
        MT0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)

#-------Equilibrium macroscopic flux
        FLUX_EQ[0] = fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 0, 0)
        FXI = fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 2, 0, 0)
        FYI = fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 1, 0)
        FZI = fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 0, 1)
        FLUX_EQ[4] = (fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 3, 0, 0)         
        + fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 2, 0)         
        + fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 0, 2))*0.5

        FLUX_EQ[1:3] = ICRTT(FXI, FYI, FZI, DIJK[:,:,2,IX,IY,IZ])
        FLUX_EQ = FLUX_EQ * DLL

        FXI = 0.0
        FYI = 0.0
        FZI = 0.0
        FLUX_UN = 0.0
        Q = 0.0

        GAML=gam_mono
        CK[None] = get_ck(GAML)

        for iu, iv, iw in ti.static(range(iuh, ivh, iwh)):
#------------------------------------------------------------------------------
#------------------------ y ---------------------------------------------------
#------------------------------------------------------------------------------
            UN  = dis_v[iu,iv,iw]
            VNT = -dis_u[iu,iv,iw]
            WT  = dis_w[iu,iv,iw]
           
            CX  = UN - DN0[1]
            CY  = VNT - DN0[2]
            CZ  = WT - DN0[3]
            UV2 = CX * CX + CY * CY + CZ * CZ

#----------g0
            G0 = ti.exp(-EIG0 * UV2 + (CK[None] + NDIM) * ti.log(EIG0 / PI) * 0.5) * DN0[0]

            if UN >= 0:
                H1 = 1.0
                H2 = 0.0
            else:
                H1 = 0.0
                H2 = 1.0

#----------------------------------------------------------
#----------Equilibrium microscopic flux
#----------------------------------------------------------
            T1 = ML0[0] + ML0[1] * UN + ML0[2] * VNT + ML0[3] * WT
            T2 = MR0[0] + MR0[1] * UN + MR0[2] * VNT + MR0[3] * WT
            T3 = MT0[0] + MT0[1] * UN + MT0[2] * VNT + MT0[3] * WT
            T4 = (UN * UN + VNT * VNT + WT * WT) * 0.5
            T5 = T1 + ML0[4] * T4
            T6 = T2 + MR0[4] * T4
            T56_1 = MUD0[0] + MUD0[1] * UN + MUD0[2] * VNT + MUD0[3] * WT + MUD0[4] * T4
            T56_2 = MOI0[0] + MOI0[1] * UN + MOI0[2] * VNT + MOI0[3] * WT + MOI0[4] * T4
            T7 = T3 + MT0[4] * T4
            af_dis_f_h[iu, iv, iw, IX, IY, IZ] = UN * G0 * (RT[0] + RT[1] * UN * (T5 * H1 + T6 * H2) + RT[1] * VNT * T56_1 + RT[1] * WT * T56_2 + RT[2] * T7)

#--------------------------------------------------------------
#----------Calculate the slope of the gas distribution function
#--------------------------------------------------------------
            GGL[0] = (dis_f_h[iu, iv, iw, IX, IY, IZ] - dis_f_h[iu, iv, iw, IX - 1, IY, IZ]) / dx
            GGL[1] = (dis_f_h[iu, iv, iw, IX - 1, IY + 1, IZ] + dis_f_h[iu, iv, iw, IX, IY + 1, IZ]  
            - dis_f_h[iu, iv, iw, IX - 1, IY - 1, IZ] - dis_f_h[iu, iv, iw, IX, IY - 1, IZ]) / (4 * dx)
            GGL[2] = (dis_f_h[iu, iv, iw, IX - 1, IY, IZ + 1] + dis_f_h[iu, iv, iw, IX, IY, IZ + 1] 
            - dis_f_h[iu, iv, iw, IX - 1, IY, IZ - 1] - dis_f_h[iu, iv, iw, IX, IY, IZ - 1]) / (4 * dx)
            GGR[0] = GGL[0]
            GGR[1] = GGL[1]
            GGR[2] = GGL[2]
            GFL = 0.5 * (dis_f_h[iu, iv, iw, IX - 1, IY, IZ] + dis_f_h[iu, iv, iw, IX, IY, IZ])
            GFR = GFL

            WXY = weight[iu,iv,iw]

#-----------------------------------------------------------------------
#----------unequilibrium microscopic flux
#-----------------------------------------------------------------------
            T11 = RT[3] * GFL + RT[4] * UN * GGL[0]
            T12 = RT[3] * GFR + RT[4] * UN * GGR[0]
            T1 = T11 * H1 + T12 * H2 + RT[4] * 0.5 * (VNT * (GGL[1] + GGR[1]) + WT * (GGL[2] + GGR[2]))
            af_dis_f_h[iu, iv, iw, IX, IY, IZ] = (af_dis_f_h[iu, iv, iw, IX, IY, IZ] + UN * T1) * DLL
           
#-----------------------------------------------------------------------
#----------unquilibrium macroscopic flux
#-----------------------------------------------------------------------
            FLUX_UN[0] = FLUX_UN[0] + WXY * UN * T1
            FXI = FXI + WXY * UN * UN * T1
            FYI = FYI + WXY * UN * VNT * T1
            FZI = FZI + WXY * UN * WT * T1
            FLUX_UN[4] = FLUX_UN[4] + 0.5 * WXY * UN * T1 * (UN * UN + VNT * VNT + WT * WT)
          
#---------heat flux = FLUX(6:8)       
            if MPRAN == PRAN_SHAKHOV and EIG0 > EPSL:
                if UN >= 0:
                    QGH =UV2*GFL*WXY
                else:
                    QGH =UV2*GFR*WXY
                Q[0] = Q[0] + 0.5 * CX * QGH
                Q[1] = Q[1] + 0.5 * CY * QGH
                Q[2] = Q[2] + 0.5 * CZ * QGH

        FLUX_UN[1:3] = ICRTT(FXI, FYI, FZI, DIJK[:,:,2,IX,IY,IZ]) 
        FLUX_UN = FLUX_UN * DLL

#--------heat flux 
        TX1 = Q[0]
        TX2 = Q[1]
        TX3 = Q[2]
 
        TX0 = 0.8 * (1.0 - prantle) * EIG0 * EIG0 / UN0[0]

        FXI=0.0
        FYI=0.0
        FZI=0.0
        FLUX_Q=0.0
         
        for iu, iv, iw in ti.static(range(iuh, ivh, iwh)):
#----------------------------------------------------------------
#------------------------- y ------------------------------------
#----------------------------------------------------------------
            UN  = dis_v[iu, iv, iw]
            VNT = -dis_u[iu, iv, iw]
            WT  = dis_w[iu, iv, iw]
         
            CX  = UN - DN0[1]
            CY  = VNT - DN0[2]
            CZ  = WT - DN0[3]
            UV2 = CX * CX + CY * CY + CZ * CZ
            CQ  = CX * TX1 + CY * TX2 + CZ * TX3
      
            G0 = DN0[0] * ti.exp(-EIG0 * UV2 + (CK[None] + NDIM) * ti.log(EIG0 / PI) * 0.5)

            T4 = 2.0 * EIG0 * UV2
            GG = (T4 + CK[None] - 5.0) * TX0 * CQ * G0 * RT[0]
        
            af_dis_f_h[iu, iv, iw, IX, IY, IZ] = af_dis_f_h[iu, iv, iw, IX, IY, IZ] + UN * GG * DLL
   
            WXY = weight[iu, iv, iw]
            
            FLUX_Q[0] = FLUX_Q[0] + WXY * UN * GG
            FXI = FXI + WXY * UN * UN * GG
            FYI = FYI + WXY * UN * VNT * GG
            FZI = FZI + WXY * UN * WT * GG
            FLUX_Q[4] = FLUX_Q[4] + 0.5 * WXY * UN * GG * (UN * UN + VNT * VNT + WT * WT)
            
        FLUX_Q[1:3] = ICRTT(FXI,FYI,FZI,DIJK[:,:,2,IX,IY,IZ])
            
        if EIG0 <= EPSL:
            FLUX_Q = 0.0
        
        FLUX_Q = FLUX_Q * DLL
        af_ctr_w[IX, IY, IZ] = FLUX_EQ + FLUX_UN + FLUX_Q

#********************************************************************
#---- calculate interface flux along z
#********************************************************************
@ti.kernel
def flux_z():
    for IX, IY, IZ in ti.ndrange((2, NX+2), (2, NY+2), (2, NZ+3)):
        ULC = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        URC = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN_U = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN_D = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN_O = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN_I = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UN0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        RT = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        DN0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        UFF0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        VFF0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        WFF0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        ML0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        MR0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        MUD0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        MOI0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        MT0 = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        GGT = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        FLUX_EQ = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        FLUX_UN = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        FLUX_Q = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        Q = ti.Vector([0.0, 0.0, 0.0])
        GGL = ti.Vector([0.0, 0.0, 0.0])
        GGR = ti.Vector([0.0, 0.0, 0.0])
        DIST = ti.Vector([0.0, 0.0])

#------ time step
        DELTT = dt_all
        DLL = AREA(3, IX, IY, IZ) / DELTT

        DIST[0] = 0.5 * dx
        DIST[1] = 0.5 * dx

#-------macroscopic conserved variable of left
        ULC[0] = ctr_w[IX, IY, IZ - 1][0]
        ULC[1] = ctr_w[IX, IY, IZ - 1][3]
        ULC[2] = ctr_w[IX, IY, IZ - 1][1]
        ULC[3] = ctr_w[IX, IY, IZ - 1][2]
        ULC[4] = ctr_w[IX, IY, IZ - 1][4]

#-------macroscopic conserved variable of right
        URC[0] = ctr_w[IX, IY, IZ][0]
        URC[1] = ctr_w[IX, IY, IZ][3]
        URC[2] = ctr_w[IX, IY, IZ][1]
        URC[3] = ctr_w[IX, IY, IZ][2]
        URC[4] = ctr_w[IX, IY, IZ][4]     

#-------macroscopic conserved variable of up
        UN_U[0] = 0.5*(ctr_w[IX + 1, IY, IZ - 1][0] + ctr_w[IX + 1, IY, IZ][0])
        UN_U[1] = 0.5*(ctr_w[IX + 1, IY, IZ - 1][3] + ctr_w[IX + 1, IY, IZ][3])
        UN_U[2] = 0.5*(ctr_w[IX + 1, IY, IZ - 1][1] + ctr_w[IX + 1, IY, IZ][1])
        UN_U[3] = 0.5*(ctr_w[IX + 1, IY, IZ - 1][2] + ctr_w[IX + 1, IY, IZ][2])
        UN_U[4] = 0.5*(ctr_w[IX + 1, IY, IZ - 1][4] + ctr_w[IX + 1, IY, IZ][4])

#-------macroscopic conserved variable of down
        UN_D[0] = 0.5*(ctr_w[IX - 1, IY, IZ - 1][0] + ctr_w[IX - 1, IY, IZ][0])
        UN_D[1] = 0.5*(ctr_w[IX - 1, IY, IZ - 1][3] + ctr_w[IX - 1, IY, IZ][3])
        UN_D[2] = 0.5*(ctr_w[IX - 1, IY, IZ - 1][1] + ctr_w[IX - 1, IY, IZ][1])
        UN_D[3] = 0.5*(ctr_w[IX - 1, IY, IZ - 1][2] + ctr_w[IX - 1, IY, IZ][2])
        UN_D[4] = 0.5*(ctr_w[IX - 1, IY, IZ - 1][4] + ctr_w[IX - 1, IY, IZ][4])

#-------macroscopic conserved variable of outside        
        UN_O[0] = 0.5*(ctr_w[IX, IY + 1, IZ - 1][0] + ctr_w[IX, IY + 1, IZ][0])
        UN_O[1] = 0.5*(ctr_w[IX, IY + 1, IZ - 1][3] + ctr_w[IX, IY + 1, IZ][3])
        UN_O[2] = 0.5*(ctr_w[IX, IY + 1, IZ - 1][1] + ctr_w[IX, IY + 1, IZ][1])
        UN_O[3] = 0.5*(ctr_w[IX, IY + 1, IZ - 1][2] + ctr_w[IX, IY + 1, IZ][2])
        UN_O[4] = 0.5*(ctr_w[IX, IY + 1, IZ - 1][4] + ctr_w[IX, IY + 1, IZ][4])

#-------macroscopic conserved variable of inside
        UN_I[0] = 0.5*(ctr_w[IX, IY - 1, IZ - 1][0] + ctr_w[IX, IY - 1, IZ][0])
        UN_I[1] = 0.5*(ctr_w[IX, IY - 1, IZ - 1][3] + ctr_w[IX, IY - 1, IZ][3])
        UN_I[2] = 0.5*(ctr_w[IX, IY - 1, IZ - 1][1] + ctr_w[IX, IY - 1, IZ][1])
        UN_I[3] = 0.5*(ctr_w[IX, IY - 1, IZ - 1][2] + ctr_w[IX, IY - 1, IZ][2])
        UN_I[4] = 0.5*(ctr_w[IX, IY - 1, IZ - 1][4] + ctr_w[IX, IY - 1, IZ][4])

#-------macroscopic conserved variable at interface
        UN0 = 0.5*(ULC+URC)

#-------Lambda
        EIG0 = EIG(UN0)
        EIGL = EIG(ULC)
        EIGR = EIG(URC)
             
#-------collision time
        TAU = COLLISION_TIME_T(UN0[0], EIG0, ULC[0], EIGL, URC[0], EIGR, DELTT)
#-------time integral
        RT = CPDTT(DELTT, TAU)
        
#-------convert macroscopic conserved variables to primitive variables
        DN0 = PHYSICAL_PAR(UN0)
     
#-------calculate <u^n> <v^n> <w^n>
        UFF0 = COEF_INFITE(EIG0, DN0[1])
        VFF0 = COEF_INFITE(EIG0, DN0[2])
        WFF0 = COEF_INFITE(EIG0, DN0[3])
        
#-------calculate <u^n> >0, <u^n> <0
        UFM0 = COEF_INFITE_HARF_LEFT(EIG0, DN0[1])
        UFN0 = COEF_INFITE_HARF_RIGHT(EIG0, DN0[1], UFF0, UFM0)
        
#-------calculate al=ML0
        DLNG = DIST[0] * UN0[0]
        GGT = (UN0 - ULC) / DLNG
        ML0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)
 
#-------calculate ar=MR0
        DLNG = DIST[1] * UN0[0]
        GGT = (URC - UN0) / DLNG
        MR0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)
        
#-------calculate aud=MUDO for multidimensional
        DLNG = DIST[0] * UN0[0]
        GGT = (UN_U - UN_D) / (4.0 * DLNG)
        MUD0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)
        
#-------calculate aoi=MUDO for multidimensional
        DLNG = DIST[0] * UN0[0]
        GGT = (UN_O - UN_I) / (4.0 * DLNG)
        MOI0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)

#-------calculate At=MT0
        GGT = MUGF(ML0, MR0, MUD0, MOI0, UFF0, UFM0, UFN0, VFF0, WFF0)
        MT0 = DXE_f(EIG0, DN0[1], DN0[2], DN0[3], GGT)

#-------Equilibrium macroscopic flux
        FLUX_EQ[0] = fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 0, 0)
        FXI = fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 2, 0, 0)
        FYI = fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 1, 0)
        FZI = fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 0, 1)
        FLUX_EQ[4] = (fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 3, 0, 0)         
        + fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 2, 0)         
        + fMUVSF(DN0[0], RT, ML0, MR0, MUD0, MOI0, MT0, UFM0, UFN0, UFF0, VFF0, WFF0, 1, 0, 2))*0.5

        FLUX_EQ[1:3] = ICRTT(FXI, FYI, FZI, DIJK[:,:,3,IX,IY,IZ])
        FLUX_EQ = FLUX_EQ * DLL

        FXI = 0.0
        FYI = 0.0
        FZI = 0.0
        FLUX_UN = 0.0
        Q = 0.0

        GAML=gam_mono
        CK[None] = get_ck(GAML)

        for iu, iv, iw in ti.static(range(iuh, ivh, iwh)):
#------------------------------------------------------------------------------
#------------------------ z ---------------------------------------------------
#------------------------------------------------------------------------------
            UN  = dis_w[iu,iv,iw]
            VNT = dis_u[iu,iv,iw]
            WT  = dis_v[iu,iv,iw]
           
            CX  = UN - DN0[1]
            CY  = VNT - DN0[2]
            CZ  = WT - DN0[3]
            UV2 = CX * CX + CY * CY + CZ * CZ

#----------g0
            G0 = ti.exp(-EIG0 * UV2 + (CK[None] + NDIM) * ti.log(EIG0 / PI) * 0.5) * DN0[0]

            if UN >= 0:
                H1 = 1.0
                H2 = 0.0
            else:
                H1 = 0.0
                H2 = 1.0

#----------------------------------------------------------
#----------Equilibrium microscopic flux
#----------------------------------------------------------
            T1 = ML0[0] + ML0[1] * UN + ML0[2] * VNT + ML0[3] * WT
            T2 = MR0[0] + MR0[1] * UN + MR0[2] * VNT + MR0[3] * WT
            T3 = MT0[0] + MT0[1] * UN + MT0[2] * VNT + MT0[3] * WT
            T4 = (UN * UN + VNT * VNT + WT * WT) * 0.5
            T5 = T1 + ML0[4] * T4
            T6 = T2 + MR0[4] * T4
            T56_1 = MUD0[0] + MUD0[1] * UN + MUD0[2] * VNT + MUD0[3] * WT + MUD0[4] * T4
            T56_2 = MOI0[0] + MOI0[1] * UN + MOI0[2] * VNT + MOI0[3] * WT + MOI0[4] * T4
            T7 = T3 + MT0[4] * T4
            af_dis_f_h[iu, iv, iw, IX, IY, IZ] = UN * G0 * (RT[0] + RT[1] * UN * (T5 * H1 + T6 * H2) + RT[1] * VNT * T56_1 + RT[1] * WT * T56_2 + RT[2] * T7)

#--------------------------------------------------------------
#----------Calculate the slope of the gas distribution function
#--------------------------------------------------------------
            GGL[0] = (dis_f_h[iu, iv, iw, IX, IY, IZ] - dis_f_h[iu, iv, iw, IX - 1, IY, IZ]) / dx
            GGL[1] = (dis_f_h[iu, iv, iw, IX - 1, IY + 1, IZ] + dis_f_h[iu, iv, iw, IX, IY + 1, IZ]  
            - dis_f_h[iu, iv, iw, IX - 1, IY - 1, IZ] - dis_f_h[iu, iv, iw, IX, IY - 1, IZ]) / (4 * dx)
            GGL[2] = (dis_f_h[iu, iv, iw, IX - 1, IY, IZ + 1] + dis_f_h[iu, iv, iw, IX, IY, IZ + 1] 
            - dis_f_h[iu, iv, iw, IX - 1, IY, IZ - 1] - dis_f_h[iu, iv, iw, IX, IY, IZ - 1]) / (4 * dx)
            GGR[0] = GGL[0]
            GGR[1] = GGL[1]
            GGR[2] = GGL[2]
            GFL = 0.5 * (dis_f_h[iu, iv, iw, IX - 1, IY, IZ] + dis_f_h[iu, iv, iw, IX, IY, IZ])
            GFR = GFL

            WXY = weight[iu,iv,iw]

#-----------------------------------------------------------------------
#----------unequilibrium microscopic flux
#-----------------------------------------------------------------------
            T11 = RT[3] * GFL + RT[4] * UN * GGL[0]
            T12 = RT[3] * GFR + RT[4] * UN * GGR[0]
            T1 = T11 * H1 + T12 * H2 + RT[4] * 0.5 * (VNT * (GGL[1] + GGR[1]) + WT * (GGL[2] + GGR[2]))
            af_dis_f_h[iu, iv, iw, IX, IY, IZ] = (af_dis_f_h[iu, iv, iw, IX, IY, IZ] + UN * T1) * DLL
           
#-----------------------------------------------------------------------
#----------unquilibrium macroscopic flux
#-----------------------------------------------------------------------
            FLUX_UN[0] = FLUX_UN[0] + WXY * UN * T1
            FXI = FXI + WXY * UN * UN * T1
            FYI = FYI + WXY * UN * VNT * T1
            FZI = FZI + WXY * UN * WT * T1
            FLUX_UN[4] = FLUX_UN[4] + 0.5 * WXY * UN * T1 * (UN * UN + VNT * VNT + WT * WT)
          
#---------heat flux = FLUX(6:8)       
            if MPRAN == PRAN_SHAKHOV and EIG0 > EPSL:
                if UN >= 0:
                    QGH =UV2*GFL*WXY
                else:
                    QGH =UV2*GFR*WXY
                Q[0] = Q[0] + 0.5 * CX * QGH
                Q[1] = Q[1] + 0.5 * CY * QGH
                Q[2] = Q[2] + 0.5 * CZ * QGH

        FLUX_UN[1:3] = ICRTT(FXI,FYI,FZI,DIJK[:,:,3,IX,IY,IZ]) 
        FLUX_UN = FLUX_UN * DLL

#--------heat flux 
        TX1 = Q[0]
        TX2 = Q[1]
        TX3 = Q[2]
 
        TX0 = 0.8 * (1.0 - prantle) * EIG0 * EIG0 / UN0[0]

        FXI=0.0
        FYI=0.0
        FZI=0.0
        FLUX_Q=0.0
         
        for iu, iv, iw in ti.static(range(iuh, ivh, iwh)):
#----------------------------------------------------------------
#------------------------- z ------------------------------------
#----------------------------------------------------------------
            UN  = dis_w[iu, iv, iw]
            VNT = dis_u[iu, iv, iw]
            WT  = dis_v[iu, iv, iw]
         
            CX  = UN - DN0[1]
            CY  = VNT - DN0[2]
            CZ  = WT - DN0[3]
            UV2 = CX * CX + CY * CY + CZ * CZ
            CQ  = CX * TX1 + CY * TX2 + CZ * TX3
      
            G0 = DN0[0] * ti.exp(-EIG0 * UV2 + (CK[None] + NDIM) * ti.log(EIG0 / PI) * 0.5)

            T4 = 2.0 * EIG0 * UV2
            GG = (T4 + CK[None] - 5.0) * TX0 * CQ * G0 * RT[0]
        
            af_dis_f_h[iu, iv, iw, IX, IY, IZ] = af_dis_f_h[iu, iv, iw, IX, IY, IZ] + UN * GG * DLL
   
            WXY = weight[iu, iv, iw]
            
            FLUX_Q[0] = FLUX_Q[0] + WXY * UN * GG
            FXI = FXI + WXY * UN * UN * GG
            FYI = FYI + WXY * UN * VNT * GG
            FZI = FZI + WXY * UN * WT * GG
            FLUX_Q[4] = FLUX_Q[4] + 0.5 * WXY * UN * GG * (UN * UN + VNT * VNT + WT * WT)
            
        FLUX_Q[1:3] = ICRTT(FXI,FYI,FZI,DIJK[:,:,3,IX,IY,IZ])
            
        if EIG0 <= EPSL:
            FLUX_Q = 0.0
        
        FLUX_Q = FLUX_Q * DLL
        af_ctr_w[IX, IY, IZ] = FLUX_EQ + FLUX_UN + FLUX_Q
        
@ti.kernel
def sum_flux_x():
    for IX, IY, IZ in ti.ndrange((2, NX+2), (2, NY+2), (2, NZ+2)):
        d_ctr_w[IX, IY, IZ] += af_ctr_w[IX, IY, IZ] - af_ctr_w[IX + 1, IY, IZ]
        for iu, iv, iw in ti.static(range(iuh, ivh, iwh)):
            d_dis_f_h[iu, iv, iw, IX, IY, IZ] += af_dis_f_h[iu, iv, iw, IX, IY, IZ] - [iu, iv, iw, IX+1, IY, IZ]
    
@ti.kernel
def sum_flux_y():
    for IX, IY, IZ in ti.ndrange((2, NX+2), (2, NY+2), (2, NZ+2)):
        d_ctr_w[IX, IY, IZ] += af_ctr_w[IX, IY, IZ] - af_ctr_w[IX, IY + 1, IZ]
        for iu, iv, iw in ti.static(range(iuh, ivh, iwh)):
            d_dis_f_h[iu, iv, iw, IX, IY, IZ] += af_dis_f_h[iu, iv, iw, IX, IY, IZ] - [iu, iv, iw, IX, IY+1, IZ]

@ti.kernel
def sum_flux_z():
    for IX, IY, IZ in ti.ndrange((2, NX+2), (2, NY+2), (2, NZ+2)):
        d_ctr_w[IX, IY, IZ] += af_ctr_w[IX, IY, IZ] - af_ctr_w[IX, IY, IZ + 1]
        for iu, iv, iw in ti.static(range(iuh, ivh, iwh)):
            d_dis_f_h[iu, iv, iw, IX, IY, IZ] += af_dis_f_h[iu, iv, iw, IX, IY, IZ] - [iu, iv, iw, IX, IY, IZ+1]

#------------------------------------------------------------!
#-------------------------  update  -------------------------!
#------------------------------------------------------------! 
@ti.kernel
def update():     
    for I, J, K in ti.ndrange((2, NX + 2), (2, NY + 2), (2, NZ + 2)):
        w_old = ctr_w[I, J, K]
        ctr_w_old[I, J, K] = w_old          
              
        VOL_N = VOL[I, J, K]
        DLL = DTL[I, J, K]

        ctr_w[I, J, K] = ctr_w[I, J, K] + d_ctr_w[I, J, K] * DLL / VOL_N
                                  
#-----------------------------------------------------------------------
# Shakhov part
#-----------------------------------------------------------------------
        prim_old = get_primary(w_old, gam_mono)            
        tau_old = get_tau(prim_old, VISCO) 
        
        T1 = 0.0
        T2 = 0.0
        T3 = 0.0
        for iu, iv, iw in ti.ndrange(iuh, ivh, iwh):
            T1 = T1 + (weight[iu, iv, iw] * (dis_u[iu, iv, iw] - prim_old[1]) * ((dis_u[iu, iv, iw] - prim_old[1]) ** 2 + (dis_v[iu, iv, iw] - prim_old[2]) ** 2 + (dis_w[iu, iv, iw] - prim_old[3]) ** 2) * dis_f_h[iu, iv, iw, I, J, K])  
            T2 = T2 + (weight[iu, iv, iw] * (dis_v[iu, iv, iw] - prim_old[2]) * ((dis_u[iu, iv, iw] - prim_old[1]) ** 2 + (dis_v[iu, iv, iw] - prim_old[2]) ** 2 + (dis_w[iu, iv, iw] - prim_old[3]) ** 2) * dis_f_h[iu, iv, iw, I, J, K])
            T3 = T3 + (weight[iu, iv, iw] * (dis_w[iu, iv, iw] - prim_old[3]) * ((dis_u[iu, iv, iw] - prim_old[1]) ** 2 + (dis_v[iu, iv, iw] - prim_old[2]) ** 2 + (dis_w[iu, iv, iw] - prim_old[3]) ** 2) * dis_f_h[iu, iv, iw, I, J, K])
        qf = ti.Vector([T1 * 0.5, T2 * 0.5, T3 * 0.5])
        
        prim_new = get_primary(w_new, gam_mono)
        tau_new = get_tau(prim_new, VISCO)
                                          
#----------------------------------------------------------------------------
# Shakhov part
#--------------------------------------------------    
        for iu, iv, iw in ti.ndrange(iuh, ivh, iwh):
            vn = dis_u[iu, iv, iw]
            vt = dis_v[iu, iv, iw]
            vw = dis_w[iu, iv, iw]
            h_old_ugks = discrete_maxwell(vn, vt, vw, prim_old)
            H_plus_ugks = shakhov_part(h_old_ugks, vn, vt, vw, qf, prim_old)
            h_old_ugks = h_old_ugks + H_plus_ugks

            h_new_ugks = discrete_maxwell(vn, vt, vw, prim_new)
            H_plus_ugks = shakhov_part(h_new_ugks, vn, vt, vw, qf, prim_new)
            h_new_ugks = h_new_ugks + H_plus_ugks

            dis_f_h[iu, iv, iw, I, J, K] = (dis_f_h[iu, iv, iw, I, J, K] + DLL * d_dis_f_h[iu, iv, iw, I, J, K] / VOL_N + 0.5 * DLL * (h_new_ugks / tau_new + (h_old_ugks - dis_f_h[iu, iv, iw, I, J, K]) / tau_old)) / (1.0 + 0.5 * DLL / tau_new)    
                
#----------------------------------------------------------------
#---------------------- output ----------------------------------
#----------------------------------------------------------------
@ti.kernel
def SDINS_sa():
    UT = ti.types.matrix(NX + 4, NY + 4, NZ + 4, ti.f32)
    VT = ti.types.matrix(NX + 4, NY + 4, NZ + 4, ti.f32)
    WT = ti.types.matrix(NX + 4, NY + 4, NZ + 4, ti.f32)
    kine_miu = ti.types.matrix(NX + 4, NY + 4, NZ + 4, ti.f32)

    GAM = gam_mono
    RRR = R_argon
    for I, J, K in ti.ndrange(NX+4, NY+4, NZ+4):
        UT[I,J,K] = ctr_w[I,J,K][1] / ctr_w[I,J,K][0]
        VT[I,J,K] = ctr_w[I,J,K][2] / ctr_w[I,J,K][0]
        WT[I,J,K] = ctr_w[I,J,K][3] / ctr_w[I,J,K][0]
        DT = ctr_w[I,J,K][0]
        ET = ctr_w[I,J,K][4]
        UA = ti.sqrt(UT[I,J,K] ** 2 + VT[I,J,K] ** 2 + WT[I,J,K]**2)
        PT = (GAM - 1.0) * (ET - 0.5 * DT * UA * UA)
        TT = abs(PT / (DT * RRR))
        kine_miu[I,J,K] = VISCK(TT) / ctr_w[I,J,K][0]

    kine_ene[None] = 0.0
    kine_dis[None] = 0.0
    for I, J, K in ti.ndrange((2, NX+2), (2, NY+2), (2, NZ+2)):
        dudx = (UT[I + 1, J, K] - UT[I - 1, J, K]) / (2.0 * dx)
        dudy = (UT[I, J + 1, K] - UT[I, J - 1, K]) / (2.0 * dx)
        dudz = (UT[I, J, K + 1] - UT[I, J, K - 1]) / (2.0 * dx)
                
        dvdx = (VT[I + 1, J, K] - VT[I - 1, J, K]) / (2.0 * dx)
        dvdy = (VT[I, J + 1, K] - VT[I, J - 1, K]) / (2.0 * dx)
        dvdz = (VT[I, J, K + 1] - VT[I, J, K - 1]) / (2.0 * dx)
               
        dwdx = (WT[I + 1, J, K] - WT[I - 1, J, K]) / (2.0 * dx)
        dwdy = (WT[I, J + 1, K] - WT[I, J - 1, K]) / (2.0 * dx)
        dwdz = (WT[I, J, K + 1] - WT[I, J, K - 1]) / (2.0 * dx)
              
        kine_ene[None] = kine_ene[None] + UT[I,J,K]**2 + VT[I,J,K]**2 + WT[I,J,K]**2
        kine_dis[None] = kine_dis[None] + kine_miu[I,J,K]*(dudx**2+dudy**2+dudz**2+dvdx**2+dvdy**2+dvdz**2+dwdx**2+dwdy**2+dwdz**2)

    kine_dis[None] = kine_dis[None] * 2.0
    kine_ene[None] = kine_ene[None]/(NX*NY*NZ)
    kine_dis[None] = kine_dis[None]/(NX*NY*NZ)

def SDINS(F_NAME):   
    heat_flux = ti.types.matrix(3, NX + 2, NY + 2, NZ + 2, float)
    tauij = ti.types.matrix(6, NX + 2, NY + 2, NZ + 2, float)
    prim = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
    GAM=gam_mono          
    
    for I, J, K in ti.ndrange((2, NX+2), (2, NY+2), (2, NZ+2)):
        w = ctr_w[I, J, K]
        prim = get_primary(w, GAM)     
        heat_flux[I, J, K] = get_heat_flux(dis_f_h[:, :, :, I, J, K], dis_u, dis_v, dis_w, prim)
        tauij[I, J, K] = get_tauij(dis_f_h[:, :, :, I, J, K], dis_u, dis_v, dis_w, prim)        

    rh = open(F_NAME, "wr")
    rh.write(ctr_w)
    rh.write(heat_flux)
    rh.write(tauij)
    rh.close()
    
def SDINS_con(T_NAME):  
    rs = open(T_NAME, "wr")
    rs.write(ITL[None])
    rs.write(TIME[None])
    rs.write(TIMEC[None])
    rs.write(ctr_w)
    rs.write(dis_f_h)
    rs.close() 
                              
#------------------------------------------------------------!
#--------------------   basic parameters  -------------------!
#------------------------------------------------------------!
CFL[None] = 0.5
GAM = gam_mono
CK[None] = (5.0 - 3.0 * GAM) / (GAM - 1.0) 
DDI = DDI_ref
RRR = R_argon
PPI = DDI * RRR * TTI   

TIME = 0.0
TIMEC = 0.0
DT_MIN[None] = 0.0     

nt_rsave=1
out = open(OUT_MPI, 'w') 

#------------------------------------------------------------!
#------------------------  grid load  -----------------------!
#------------------------------------------------------------!
con = open(CON_MPI, "r")
IB1 = con.readline()
IB, IBOUND[None] = map(int, IB1.split())
IB1 = con.readline() 
NX, NY, NZ = map(int, IB1.split())
   
NX = NX - 1
NY = NY - 1
NZ = NZ - 1      
NM1 = max(max(NX, NY), NZ)

IYEND=NY
IXEND=NX
IZEND=NZ
              
grd = open(GRD_MPI, "rb")
NN = con.read(4)
NM = con.read(12)
IT, JT, KT = struct.unpack("3l", NM)
NN = con.read(4)
if IT != NX or JT != NY or KT != NZ:
    out.write('The mesh number is NOT consistent!')
    sys.exit(1)
else:
    for m in range(3):
        NN= con.read(4)
        for i in range(NX + 5):
            for j in range(NY + 5):
                for k in range(NZ + 5):
                    NM=  con.read(8)
                    NM = struct.unpack("d", NM)
                    XYZ[i, j, k][m] = NM
        NN= con.read(4)
    grd.close()

for i, j, k in range(NX + 5, NY + 5, NZ + 5):
    for m in range(3):
        XYZ[i, j, k][m] = XYZ[i, j, k][m] * XLREF

for i, j, k in range(NX + 4, NY + 4, NZ + 4):
    XYZT[i, j, k][0] = XYZ[i + 1, j, k][0] - XYZ[i, j, k][0] 
    XYZT[i, j, k][1] = XYZ[i, j + 1, k][1] - XYZ[i, j, k][1] 
    XYZT[i, j, k][2] = XYZ[i, j, k + 1][2] - XYZ[i, j, k][2] 
dx = XYZT[0, 0, 0][0]

velocity_grids() 
out.write('begin velocity_grids')
#------------------------------------------------------------!
#---------------------     initialize    --------------------!
#------------------------------------------------------------! 
DDEN1=DDI
TT=PPI/(DDI*RRR)
GAM=gam_mono
V1N=Mach*ti.sqrt(GAM*RRR*TTI)
V2N=Mach*ti.sqrt(GAM*RRR*TTI) 

joby = open('joby.dat', "r")
IB1 = joby.readline()
ICONT, IRMEAN, nt_rsave = map(int, IB1.split())
IB1 = joby.readline() 
TSTOP = map(float, IB1.split())
joby.close()
    
out.write(ICONT, IRMEAN, TSTOP)   

if ICONT == 0:
    ITL=0
    vel = open(vel_mpi, "rb") 
    NN= vel.read(4)   
    for k in range(NZ + 5):
        for j in range(NY + 5):
            for i in range(NX + 5):
                NM=  vel.read(24)
                NM = struct.unpack("3d", NM)
                uvw0[i, j, k] = NM
    NN= vel.read(4)
    vel.close()

    for i, j, k in range(NX + 4, NY + 4, NZ + 4):
        ctr_w[i, j, k][0] = DDI
        ctr_w[i, j, k][1] = uvw0[i, j, k][0] * DDI
        ctr_w[i, j, k][2] = uvw0[i, j, k][1] * DDI
        ctr_w[i, j, k][3] = uvw0[i, j, k][2] * DDI
        ctr_w[i, j, k][4] = 0.5 / DDEN1 * (ctr_w[i, j, k][1] ** 2 + ctr_w[i, j, k][2] ** 2 + ctr_w[i, j, k][3] ** 2) + PPI / (GAM - 1.0)
        w_new = ctr_w[i, j, k]              
        
    for i, j, k in range(NX + 4, NY + 4, NZ + 4):
        prim_new = get_primary(ctr_w[i, j, k], gam_mono)
        for iu, iv, iw in ti.ndrange(iuh, ivh, iwh):
            vn = dis_u[iu, iv, iw]
            vt = dis_v[iu, iv, iw]
            vw = dis_w[iu, iv, iw]
            dis_f_h[iu, iv, iw, i, j, k] = discrete_maxwell(vn, vt, vw, prim_new)
            h_new = discrete_ce(dis_f_h[iu, iv, iw, i, j, k], vn, vt, vw, prim_new, i, j, k)
            dis_f_h[iu, iv, iw, i, j, k] = dis_f_h[iu, iv, iw, i, j, k] + h_new                    

else:
    if nt_rsave == 0:
        rsav = open(RSAV_MPI, "rb")
    elif nt_rsave == 1:
        rasv = open(RSAV1_MPI, "rb")
    
    NN= rsav.read(4) 
    NM= rsav.read(4) 
    ITL = struct.unpack("i", NM)
    NN= rsav.read(4)  
    NN= rsav.read(4) 
    NM= rsav.read(8) 
    TIME = struct.unpack("d", NM)
    NN= rsav.read(4)
    NN= rsav.read(4) 
    NM= rsav.read(8) 
    TIMEC = struct.unpack("d", NM)
    NN= rsav.read(4)
    NN= rsav.read(4)
    for i, j, k in range(NX + 4, NY + 4, NZ + 4):
        NM=  rsav.read(8)
        NM = struct.unpack("d", NM)
        ctr_w[i, j, k][0] = NM
    NN= rsav.read(4)
    NN= rsav.read(4)
    for i, j, k in range(NX + 4, NY + 4, NZ + 4):
        NM=  rsav.read(8)
        NM = struct.unpack("d", NM)
        ctr_w[i, j, k][1] = NM
    NN= rsav.read(4)
    NN= rsav.read(4)
    for i, j, k in range(NX + 4, NY + 4, NZ + 4):
        NM=  rsav.read(8)
        NM = struct.unpack("d", NM)
        ctr_w[i, j, k][2] = NM
    NN= rsav.read(4)
    NN= rsav.read(4)
    for i, j, k in range(NX + 4, NY + 4, NZ + 4):
        NM=  rsav.read(8)
        NM = struct.unpack("d", NM)
        ctr_w[i, j, k][3] = NM
    NN= rsav.read(4)
    NN= rsav.read(4)
    for i, j, k in range(NX + 4, NY + 4, NZ + 4):
        NM=  rsav.read(8)
        NM = struct.unpack("d", NM)
        ctr_w[i, j, k][4] = NM
    NN= rsav.read(4)
    NN= rsav.read(4)
    for i, j, k in range(NX + 4, NY + 4, NZ + 4):
        for iu, iv, iw in range(iuh, ivh, iwh):
            NM=  rsav.read(8)
            NM = struct.unpack("d", NM)
            dis_f_h[iu, iv, iw, i, j, k] = NM
    NN= rsav.read(4)
    rsav.close()

    out.write('TIME = ',TIME) 

out.write('begin part_ini') 
TIME_C = TIMEC
CCELL()
CVOL()
out.write('begin CCELL CVOL') 

out.write('begin compution') 
MAX_STEP = 0.5 * VOLMIN[None] / VISCO
VISM = VISCO
REG_MIN = 0.6
REG_MAX = 1.0
RE_INF = DDI * Mach / 2.0 * ti.sqrt(GAM * RRR * TTI) / VISCK(TTI)
out.write('Reynolds number: ',RE_INF) 
out.write('Minimal cell Reynolds number: ', RE_INF*(VOLMIN[None])**(1.0/3.0)) 
out.write('Kn=: ',Kn,"Mach=",Mach,"Rel=",Re_lamda,"taut=",tau_t_i) 
out.write('miu=: ',VISCO,"p=",PPI,"R=",R_argon) 

istp = 0 

outputvel=0
if ICONT == 0:
    out.write("this is the",ICONT," th wtitting")  
    CH_MYID_wr = str(outputvel)
    rh_mpi ="rhp/rhp_"//CH_MYID_wr//".dat"
    SDINS(rh_mpi)

for i_o in range(i_o_t):
    outputvel_t[i_o] = 0.1 * tau_t_i * i_o

skew = open('skew.dat', 'w')
times = open('times.dat', 'w')

#-----------------------------------------------------------------
#---------------calculate area------------------------------------
#----------------------------------------------------------------- 
for i, j, k in range((2, NX + 3), (2, NY + 3), (2, NZ + 3)):
    CAREA(1, i, j, k, DIJK[0, i, j, k])
    CAREA(2, i, j, k, DIJK[1, i, j, k])
    CAREA(3, i, j, k, DIJK[2, i, j, k])

#------------------------------------------------------------!
#--------------------- start calculation --------------------!
#------------------------------------------------------------!  
while TIME <= TSTOP:
    istp = istp + 1

    if istp==1:
        out.write("start calculation") 
    
    out.write("NX,NY,NZ=",NX,NY,NZ," iuh,ivh,iwh=",iuh,ivh,iwh)     

    if istp == 1 or istp % 10 == 0 or outputvel != 0:
        SDINS_sa()
        times.write(TIME/tau_t_i,kine_ene[None],kine_dis[None])

#------------------------------------------------------------!
#--------------------      time step       ------------------!
#------------------------------------------------------------!
    DT_MIN[None] = MAX_STEP
    DT_MAX[None] = 0.0
          
    CDTL(DT_MIN[None],DT_MAX[None])
    dt_all=DT_MIN[None]
    
    for i, j, k in range(NX + 4, NY + 5, NZ + 4): 
        DTL[i, j, k] = DT_MIN[None]

    if istp==1:
        out.write("time step")    

    for i_o in range(i_o_t):
        if TIME < outputvel_t(i_o) and (TIME + DT_MIN[None]) >= outputvel_t(i_o):
            DT_MIN[None] = outputvel_t(i_o) - TIME
            DTL = DT_MIN[None]
            outputvel=i_o

    if istp % 10 == 0:
        print(istp, outputvel, (TIME+DT_MIN[None])/tau_t_i, kine_ene[None])

#----------------------------------------------------------------------
#----------------------- flux -----------------------------------------
#----------------------------------------------------------------------
    d_ctr_w = 0.0
    d_dis_f_h = 0.0
    flux_x()
    sum_flux_x()
    flux_y()
    sum_flux_y
    flux_z()
    sum_flux_z()

#----------------------------------------------------------------------
#----------------------- update ---------------------------------------
#----------------------------------------------------------------------
    update()

    if istp == 1:
        out.write("update")

    ITL = ITL + 1 
    TIME = TIME + DT_MIN[None]
    if outputvel != 0: 
        out.write("this is the",outputvel," th wtitting")  
        CH_MYID_wr = str(outputvel)
        rh_mpi ="rhp/rhp_"//CH_MYID_wr//".dat"
        SDINS(rh_mpi)
        outputvel=0

    if istp % 1000 ==0:
        nt_rsave = (nt_rsave + 1) % 2
        if nt_rsave == 0:
            SDINS_con(ITL, TIME, TIMEC, RSAV_MPI)
        elif nt_rsave == 1:
            SDINS_con(ITL, TIME, TIMEC, RSAV1_MPI)

        joby = open('joby.dat','w')
        joby.write(ICONT, IRMEAN, nt_rsave)
        joby.write(TSTOP)
        joby.close()

    if TIME / tau_t_i >= 3.0:
        skew.close()
        times.close()
        sys.exit()
